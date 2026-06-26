#!/usr/bin/env python3
"""harvest_strawberry_action_ik.py

Diff-IK로 딸기 접근 → 파지 → 바구니 이동 → 릴리즈 전체 시퀀스 (PPO 없음).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
입력 토픽:

  /call_quadruped  (std_msgs/String) ← straight.py
    └─ 트리거 전용 (좌표 불필요, 수신 여부만 사용)
    └─ 수신 시: _busy=True, COLLECT_TIME 동안 딸기 좌표 수집 시작
    └─ _busy=True 일 때는 무시

  /harvest_request (std_msgs/String) ← yolo_detector_node_4.py
    └─ JSON: {"x": float, "y": float, "z": float}
    └─ _collecting=True(수집 중) 일 때만 큐에 추가
    └─ 수집 완료 후에는 무시

  /dsr01/joint_states (sensor_msgs/JointState) ← 로봇팔 드라이버
    └─ 매 프레임 현재 관절 각도·속도 갱신

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
출력 토픽:

  /robot1/arm/joint_command (sensor_msgs/JointState) → Isaac Sim OmniGraph
    └─ 매 50Hz 스텝마다 발행 → ArticulationController가 관절 구동

  /robot1/gripper/command (std_msgs/Float64) → 그리퍼 드라이버
    └─ 딸기 도달 시 0.75 rad (닫기)
    └─ 바구니 도달 시 0.0 rad (열기)
    └─ /tmp/gripper_cmd.txt 에도 동시 기록 (Isaac Sim 파일 IPC)

  /harvest_done (std_msgs/Bool) → straight.py
    └─ 큐의 모든 딸기 수확 완료 후 True 발행

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
동작 흐름:

  1. /call_quadruped 수신
       → _busy=True, _collecting=True
       → COLLECT_TIME(3초) 동안 /harvest_request 좌표를 _strawberry_queue에 수집

  2. COLLECT_TIME 경과 (_on_collect_done)
       → _collecting=False
       → 큐에서 첫 번째 딸기 꺼내서 수확 시작

  3. 딸기 하나 수확 (Phase 0 → Phase 1)
       [Phase 0] READY_POSE → 3초 대기 → 50Hz Diff-IK 루프로 딸기 접근
                 → dist < 0.05m → 그리퍼 닫기 → Phase 1
       [Phase 1] BASKET_POS로 50Hz Diff-IK 이동
                 → dist < 0.05m → 그리퍼 열기 → _process_next_from_queue

  4. _process_next_from_queue
       → 큐에 딸기 남아있으면: 다음 딸기 수확 시작 (3번 반복)
       → 큐가 비면: /harvest_done True 발행 → _busy=False
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json

import numpy as np

import rclpy
import rclpy.duration
import rclpy.time
import tf2_ros
import tf2_geometry_msgs  # noqa: F401
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float64, Bool
from std_srvs.srv import SetBool

# ── 그리퍼 상수 ─────────────────────────────────────────────────────────────
GRIPPER_CMD_FILE  = "/tmp/gripper_cmd.txt"  # Isaac Sim 파일 IPC 경로
GRIPPER_CLOSE_VAL = 0.75                    # 닫힘 각도 (rad)
GRIPPER_OPEN_VAL  = 0.0                     # 열림 각도 (rad)

# ── 바구니 위치 (EE 좌표, arm base 프레임 기준) ─────────────────────────────
# TODO: 실제 바구니 위치로 수정 필요
BASKET_POS = np.array([0.3, 0.0, 0.4], dtype=np.float64)

# ── 프레임 이름 ──────────────────────────────────────────────────────────────
CAMERA_FRAME   = "World_kitkit2_finalllll_carter_warehouse_navigation_Nova_Carter_ROS_chassis_link_sensors_left_hawk_left"
ARM_BASE_FRAME = "World_kitkit2_finalllll_carter_warehouse_navigation_m0609_rg2_01_base_link"

# ── 관절 상수 ────────────────────────────────────────────────────────────────
READY_POSE = np.array([-1.6773, 0.0, 0.7871, 0.0, 0.6981, -1.5708], dtype=np.float64)

CONTROL_HZ    = 50.0   # 제어 주기 (Hz)
IK_SCALE      = 0.01   # 스텝당 최대 이동 거리 (m)
ARRIVE_THRESH = 0.05   # 도달 판정 거리 (m)
MAX_STEPS     = 300    # 최대 스텝 수
COLLECT_TIME      = 3.0    # /call_quadruped 수신 후 딸기 좌표 수집 시간 (초)
DEDUP_DIST        = 0.05   # 이 거리(m) 이내의 좌표는 같은 딸기로 간주해 큐에서 제외
WAREHOUSE_ENABLED = True  # /go_return_warehouse 서버 구현 전까지 False로 유지

# ── M0609 FK / Diff-IK ───────────────────────────────────────────────────────
_J_CFG = [
    ([0, 0, 0.1345],  [0,      0,      0     ]),
    ([0, 0.0062, 0],  [0,     -1.571, -1.571 ]),
    ([0.411, 0, 0],   [0,      0,      1.571 ]),
    ([0, -0.368, 0],  [1.571,  0,      0     ]),
    ([0, 0, 0],       [-1.571, 0,      0     ]),
    ([0, -0.121, 0],  [1.571,  0,      0     ]),
]
_EE_OFFSET = np.array([0.02705, -0.00953, 0.11387])


def _rpy(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
        [  -sp,          cp*sr,          cp*cr],
    ])


def m0609_fk(q) -> np.ndarray:
    """관절 각도 q(6) → 엔드이펙터 위치 (3,)"""
    T = np.eye(4)
    for i, (xyz, rpy) in enumerate(_J_CFG):
        cq, sq = np.cos(q[i]), np.sin(q[i])
        Ti = np.eye(4)
        Ti[:3, :3] = _rpy(*rpy) @ np.array([[cq,-sq,0],[sq,cq,0],[0,0,1]])
        Ti[:3, 3] = xyz
        T = T @ Ti
    return (T[:3, :3] @ _EE_OFFSET + T[:3, 3]).astype(np.float64)


def m0609_diff_ik(q, dp, damping=1e-3) -> np.ndarray:
    """현재 관절각 q + 목표 변위 dp → 다음 관절각 (Jacobian 수치 미분)"""
    eps = 1e-5
    p0 = m0609_fk(q)
    J = np.zeros((3, 6))
    for i in range(6):
        dq = q.copy(); dq[i] += eps
        J[:, i] = (m0609_fk(dq) - p0) / eps
    return q + J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), dp)


# ── 메인 노드 ────────────────────────────────────────────────────────────────
class HarvestStrawberryIKNode(Node):
    def __init__(self):
        super().__init__("harvest_strawberry_ik")

        self._joint_state = None
        self._busy        = False

        # 제어 루프 상태
        self._phase       = 0      # 0: 딸기 접근(IK), 1: 바구니 이동(IK)
        self._target      = None
        self._step        = 0
        self._ctrl_timer  = None   # 50Hz 제어 타이머
        self._ready_timer = None   # READY_POSE 대기 원샷 타이머

        self._basket_count      = 0      # 바구니에 넣은 딸기 수
        self._basket_full       = False  # 창고 이동 요청 보냈는지 (중복 방지)
        self._waiting_warehouse = False  # 창고 응답 대기 중 (True면 수확 차단)
        self._warehouse_cli     = self.create_client(SetBool, "/go_return_warehouse")

        # 딸기 좌표 수집 상태
        self._collecting       = False  # /harvest_request 수집 중 여부
        self._collect_timer    = None   # 수집 종료 타이머
        self._strawberry_queue = []     # 수집된 딸기 좌표 큐 [(x,y,z), ...]

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._joint_cmd_pub    = self.create_publisher(JointState, "/robot1/arm/joint_command", 10)
        self._gripper_pub      = self.create_publisher(Float64, "/robot1/gripper/command", 10)
        self._harvest_done_pub = self.create_publisher(Bool, "/harvest_done", 10)
        self._joint_sub        = self.create_subscription(JointState, "/dsr01/joint_states", self._joint_cb, 10)
        self._harvest_sub      = self.create_subscription(String, "/call_quadruped", self._harvest_cb, 10)
        self._request_sub      = self.create_subscription(String, "/harvest_request", self._harvest_request_cb, 10)

        self.get_logger().info("HarvestStrawberry IK 대기 중... (/call_quadruped, /harvest_request)")

    # ── 관절 상태 수신 ────────────────────────────────────────────────────────
    def _joint_cb(self, msg: JointState):
        self._joint_state = msg

    # ── /call_quadruped 수신 (트리거만, 좌표 사용 안 함) ─────────────────────
    def _harvest_cb(self, _msg: String):
        if self._busy or self._waiting_warehouse:
            return
        self._busy             = True
        self._collecting       = True
        self._strawberry_queue = []
        self.get_logger().info(
            f"  🔔 수확 트리거 수신 → {COLLECT_TIME}초 동안 딸기 좌표 수집 시작"
        )
        self._collect_timer = self.create_timer(COLLECT_TIME, self._on_collect_done)

    # ── /harvest_request 수신 ({"x":..,"y":..,"z":..}) ───────────────────────
    def _harvest_request_cb(self, msg: String):
        if not self._collecting:
            return  # 수집 기간이 아니면 무시
        try:
            data = json.loads(msg.data)
            x = float(data["x"])
            y = float(data["y"])
            z = float(data["z"]) - 3.0
        except Exception as e:
            self.get_logger().error(f"파싱 오류: {e}")
            return
        # 같은 딸기가 여러 프레임에서 반복 감지되므로 DEDUP_DIST 이내면 무시
        new_pos = np.array([x, y, z], dtype=np.float64)
        for qx, qy, qz in self._strawberry_queue:
            if np.linalg.norm(new_pos - np.array([qx, qy, qz], dtype=np.float64)) < DEDUP_DIST:
                return
        self._strawberry_queue.append((x, y, z))
        self.get_logger().info(
            f"  🍓 새 딸기 등록 ({len(self._strawberry_queue)}개): "
            f"({x:.3f}, {y:.3f}, {z:.3f})"
        )

    # ── COLLECT_TIME 경과 → 큐 처리 시작 ─────────────────────────────────────
    def _on_collect_done(self):
        self._collect_timer.cancel()
        self._collect_timer = None
        self._collecting    = False
        self.get_logger().info(
            f"  수집 완료: 총 {len(self._strawberry_queue)}개 고유 딸기 → 순차 수확 시작"
        )
        self._process_next_from_queue()

    # ── 큐에서 다음 딸기 처리 또는 완료 발행 ──────────────────────────────────
    def _process_next_from_queue(self):
        
        if self._waiting_warehouse:
            # 바구니 가득 참 → 창고 응답 올 때까지 루프 중단
            self.get_logger().info("  ⏸ 바구니 가득 참 → 창고 응답 대기 중 (수확 중단)")
            self._ctrl_timer and self._ctrl_timer.cancel()
            self._ctrl_timer = None
            self._phase = 0
            self._busy  = False
            return
        
        if self._strawberry_queue:
            x, y, z = self._strawberry_queue.pop(0)
            self.get_logger().info(
                f"  🤖 남은 딸기: {len(self._strawberry_queue)}개 대기 중, "
                f"현재 목표: ({x:.3f}, {y:.3f}, {z:.3f})"
            )
            self._start_harvest(x, y, z)
        else:
            self.get_logger().info("[DONE] 모든 딸기 수확 완료 → /harvest_done True 발행")
            done_msg = Bool()
            done_msg.data = True
            self._harvest_done_pub.publish(done_msg)
            self._phase = 0
            self._busy  = False

    # ── 수확 시작 공통 처리 ───────────────────────────────────────────────────
    def _start_harvest(self, x: float, y: float, z: float):
        self._target = self._transform_to_arm_base(x, y, z)
        self.get_logger().info(
            f"Ripe 감지! 카메라:({x:.3f},{y:.3f},{z:.3f}) "
            f"→ arm base:({self._target[0]:.3f},{self._target[1]:.3f},{self._target[2]:.3f})"
        )
        self._phase = 0
        self._step  = 0
        self._pub_cmd(READY_POSE)
        self._ready_timer = self.create_timer(3.0, self._on_ready)

    # ── READY 대기 완료 → 제어 타이머 시작 ───────────────────────────────────
    def _on_ready(self):
        self._ready_timer.cancel()
        self._ready_timer = None
        self._ctrl_timer = self.create_timer(1.0 / CONTROL_HZ, self._control_step)

    # ── 50Hz 제어 스텝 ────────────────────────────────────────────────────────
    def _control_step(self):
        q    = self._get_joint_pos()
        ee   = m0609_fk(q)
        diff = self._target - ee
        dist = float(np.linalg.norm(diff))

        if self._phase == 0:
            # ── Phase 0: Diff-IK로 딸기 접근 ─────────────────────────────────
            self.get_logger().info(
                f"[P0/{self._step:3d}] EE=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})  dist={dist:.4f}m"
            )
            if dist < ARRIVE_THRESH:
                self.get_logger().info("[GRASP] 딸기 도달 → 그리퍼 닫기")
                self._close_gripper()
                self._phase  = 1
                self._step   = 0
                self._target = BASKET_POS.copy()
                return

            step_vec = diff / dist * IK_SCALE
            self._pub_cmd(m0609_diff_ik(q, step_vec))
            self._step += 1
            if self._step >= MAX_STEPS:
                self.get_logger().warning("[TIMEOUT] phase0 max_steps 초과")
                self._stop_control()

        else:
            # ── Phase 1: Diff-IK로 바구니 이동 ───────────────────────────────
            self.get_logger().info(
                f"[P1/{self._step:3d}] EE=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})  dist={dist:.4f}m"
            )
            if dist < ARRIVE_THRESH:
                self.get_logger().info("[PLACE] 바구니 도달 → 그리퍼 열기")
                self._open_gripper()
                self._pub_cmd(READY_POSE)
                # 제어 루프 종료 후 큐에서 다음 딸기 처리
                if self._ctrl_timer is not None:
                    self._ctrl_timer.cancel()
                    self._ctrl_timer = None
                self._process_next_from_queue()
                return

            step_vec = diff / dist * IK_SCALE
            self._pub_cmd(m0609_diff_ik(q, step_vec))
            self._step += 1
            if self._step >= MAX_STEPS:
                self.get_logger().warning("[TIMEOUT] phase1 max_steps 초과")
                self._stop_control()

    # ── 그리퍼 닫기 ──────────────────────────────────────────────────────────
    def _close_gripper(self):
        self._basket_count += 1
        msg = Float64()
        msg.data = GRIPPER_CLOSE_VAL
        self._gripper_pub.publish(msg)
        with open(GRIPPER_CMD_FILE, "w") as f:
            f.write(f"{GRIPPER_CLOSE_VAL:.6f}")
        self.get_logger().info(
            f"[GRIPPER] 닫기 → {GRIPPER_CLOSE_VAL:.3f} rad  (바구니 딸기: {self._basket_count}개)"
        )
        if WAREHOUSE_ENABLED and self._basket_count >= 6 and not self._basket_full:
            self._basket_full       = True
            self._waiting_warehouse = True
            req = SetBool.Request()
            req.data = True
            future = self._warehouse_cli.call_async(req)
            future.add_done_callback(self._on_warehouse_response)
            self.get_logger().info("[BASKET] 딸기 6개 달성 → /go_return_warehouse True 요청 (응답 대기 중)")

    # ── /go_return_warehouse 응답 수신 ───────────────────────────────────────
    def _on_warehouse_response(self, future):
        self._waiting_warehouse = False
        self._basket_full       = False
        self._basket_count      = 0
        if self._strawberry_queue:
            # Case 1: 큐에 딸기 남아있음 → 수확 재개
            self.get_logger().info(
                f"[WAREHOUSE] 창고 복귀 완료 → 남은 딸기 {len(self._strawberry_queue)}개 수확 재개"
            )
            self._busy = True
            self._process_next_from_queue()
        else:
            # Case 2: 큐 비어있음 → 화분 수확 완료
            self.get_logger().info("[WAREHOUSE] 창고 복귀 완료 → 모든 딸기 수확 완료")
            done_msg = Bool()
            done_msg.data = True
            self._harvest_done_pub.publish(done_msg)
            self._busy = False

    # ── 그리퍼 열기 ──────────────────────────────────────────────────────────
    def _open_gripper(self):
        msg = Float64()
        msg.data = GRIPPER_OPEN_VAL
        self._gripper_pub.publish(msg)
        with open(GRIPPER_CMD_FILE, "w") as f:
            f.write(f"{GRIPPER_OPEN_VAL:.6f}")
        self.get_logger().info(f"[GRIPPER] 열기 → {GRIPPER_OPEN_VAL:.3f} rad")

    # ── 제어 종료 및 상태 초기화 ──────────────────────────────────────────────
    def _stop_control(self):
        if self._ctrl_timer is not None:
            self._ctrl_timer.cancel()
            self._ctrl_timer = None
        if self._collect_timer is not None:
            self._collect_timer.cancel()
            self._collect_timer = None
        self._collecting       = False
        self._strawberry_queue = []
        self._phase            = 0
        self._busy             = False

    # ── TF2 좌표 변환: 카메라 프레임 → 팔 base 프레임 ────────────────────────
    def _transform_to_arm_base(self, x, y, z) -> np.ndarray:
        pt = PointStamped()
        pt.header.frame_id = CAMERA_FRAME
        pt.header.stamp = rclpy.time.Time().to_msg()
        pt.point.x, pt.point.y, pt.point.z = x, y, z
        try:
            out = self._tf_buffer.transform(
                pt, ARM_BASE_FRAME,
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
            return np.array([out.point.x, out.point.y, out.point.z], dtype=np.float64)
        except Exception as e:
            self.get_logger().warning(f"TF 변환 실패 ({e}), 원본 좌표 사용")
            return np.array([x, y, z], dtype=np.float64)

    # ── 현재 관절 위치 읽기 ───────────────────────────────────────────────────
    def _get_joint_pos(self) -> np.ndarray:
        if self._joint_state is None:
            return READY_POSE.copy()
        n2i = {n: i for i, n in enumerate(self._joint_state.name)}
        q = READY_POSE.copy()
        for k, name in enumerate([f"joint_{i}" for i in range(1, 7)]):
            idx = n2i.get(name)
            if idx is not None:
                q[k] = self._joint_state.position[idx]
        return q

    # ── 관절 명령 발행 ────────────────────────────────────────────────────────
    def _pub_cmd(self, q6):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = [f"joint_{i}" for i in range(1, 7)]
        msg.position = [float(v) for v in q6]
        self._joint_cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HarvestStrawberryIKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
