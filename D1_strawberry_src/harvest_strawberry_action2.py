#!/usr/bin/env python3
"""harvest_strawberry_action2.py

Diff-IK + PPO 하이브리드로 딸기 수확 전체 시퀀스를 수행하는 노드.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
제어 전략 (IK → PPO 하이브리드)

  타겟이 PPO 훈련 범위(x±0.4, y -0.9~-0.6, z 0.05~0.30) 안에 있거나
  EE와 타겟 거리가 IK_HANDOFF_DIST(0.15m) 이하면 PPO 제어.
  그 외에는 Diff-IK로 타겟 방향으로 1cm/step 접근.

  카메라(OpenCV) → arm base 변환: CAM_TO_ARM_R, CAM_TO_ARM_T 하드코딩.
  Isaac Sim에서 joint state 피드백이 없으므로 dead-reckoning(_current_q) 사용.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
입력 토픽:

  /call_quadruped  (std_msgs/String) ← straight.py
    └─ 수확 트리거. _busy 또는 _waiting_warehouse 이면 무시.
    └─ 수신 시 COLLECT_TIME(3초) 동안 딸기 좌표 수집 시작.

  /harvest_request (std_msgs/String) ← yolo_detector_node_4.py
    └─ JSON {"x", "y", "z"}: YOLO 감지 딸기의 카메라 프레임 3D 좌표.
    └─ _collecting 중일 때만 수신. DEDUP_DIST(5cm) 이내 중복 제거.
    └─ z에 -0.05m 오프셋 적용 (딸기 5cm 앞에서 정지).

  /robot1/arm/joint_states (sensor_msgs/JointState) ← Isaac Sim OmniGraph
    └─ 실제 관절 각도 수신 → _current_q 보정에 사용 (설정 시).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
출력 토픽:

  /robot1/arm/joint_command (sensor_msgs/JointState) → Isaac Sim OmniGraph
    └─ 50Hz로 관절 목표각 발행 → ArticulationController 구동.

  /robot1/gripper/command (std_msgs/Float64) → 그리퍼
    └─ 파지: 0.75 rad / 릴리즈: 0.0 rad.
    └─ /tmp/gripper_cmd.txt 동시 기록 (Isaac Sim IPC).

  /harvest_done (std_msgs/Bool) → straight.py
    └─ 큐의 모든 딸기 수확 완료 후 True 발행.

서비스 클라이언트:

  /go_return_warehouse (std_srvs/SetBool) → robot2
    └─ 바구니 6개 달성 시 창고 복귀 요청. 응답 전까지 수확 차단.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
동작 흐름:

  1. /call_quadruped 수신
       → COLLECT_TIME 동안 /harvest_request 좌표 수집
       → 수집 완료 시 cam_z 오름차순 정렬 (가까운 딸기 우선)

  2. 딸기 하나 수확 (Phase 0 → Phase 1)
       [Phase 0] READY_POSE → 3초 대기 → 50Hz 제어 루프
                 IK 접근(dist > 0.15m) → PPO 정밀 접근(dist ≤ 0.15m)
                 → dist < 0.05m → 그리퍼 닫기 → Phase 1

       [Phase 1] BASKET_POS 타겟으로 동일한 IK→PPO 제어
                 → dist < 0.05m → 그리퍼 열기 → 다음 딸기 또는 완료

  3. 큐 비면 /harvest_done True 발행 → _busy=False

  4. 바구니 6개 달성 시 /go_return_warehouse 요청 → 창고 복귀 후 재개

로그: /tmp/harvest_log_YYYYMMDD_HHMMSS.csv (EE위치·타겟·액션·관절각 기록)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
from pathlib import Path

import numpy as np
import csv
import datetime
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float64, Bool
from std_srvs.srv import SetBool

# ── 그리퍼 상수 ─────────────────────────────────────────────────────────────
GRIPPER_CMD_FILE  = "/tmp/gripper_cmd.txt"  # Isaac Sim 파일 IPC 경로
GRIPPER_CLOSE_VAL = 0.75                    # 닫힘 각도 (rad)
GRIPPER_OPEN_VAL  = 0.0                     # 열림 각도 (rad)

# ── 바구니 위치 (EE 좌표, arm base 프레임 기준) ─────────────────────────────
# TODO: 실제 바구니 위치로 수정 필요
BASKET_POS = np.array([0.3, 0.0, 0.4], dtype=np.float32)

# ── 카메라→arm 하드코딩 변환 ─────────────────────────────────────────────────
# Camera(OpenCV: X=right,Y=down,Z=forward) → Arm base frame
# Verified against Isaac Sim USD stage: fruit_02(-0.617,-0.789,-0.007), fruit_04(-0.103,-0.983,0.019)
CAM_TO_ARM_R = np.array([[-2., 0., 0.],
                          [ 0., 0.,-2.],
                          [ 0.,-2., 0.]], dtype=np.float32)
CAM_TO_ARM_T = np.array([0.1138, -0.2748, -0.2016], dtype=np.float32)

# ── 관절 상수 ────────────────────────────────────────────────────────────────
READY_POSE = np.array([-1.6773, 0.0, 0.7871, 0.0, 0.6981, -1.5708], dtype=np.float32)
# deg:       (  -96.1, 0.0, 45.1, 0.0, 40.0, -90.0 )
DEFAULT_JOINT_POS = np.concatenate([READY_POSE, np.zeros(2, dtype=np.float32)])  # (8,)

IK_SCALE          = 0.01
CONTROL_HZ        = 50.0
ARRIVE_THRESH     = 0.05
MAX_STEPS         = 1000
IK_HANDOFF_DIST   = 0.15   # IK → PPO 전환 거리: EE가 target에서 이 거리 이내면 PPO 시작
# PPO 훈련 범위 (arm base frame)
PPO_TGT_X_RANGE = (-0.4,  0.4)
PPO_TGT_Y_RANGE = (-0.9, -0.6)
PPO_TGT_Z_RANGE = (0.05,  0.30)
COLLECT_TIME      = 3.0    # /call_quadruped 수신 후 딸기 좌표 수집 시간 (초)
DEDUP_DIST        = 0.05   # 이 거리(m) 이내의 좌표는 같은 딸기로 간주해 큐에서 제외
WAREHOUSE_ENABLED = True  # /go_return_warehouse 서버 구현 전까지 False로 유지
DEFAULT_POLICY = str(Path(__file__).parent / "resource" / "grap_str_model1.pt")

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


def m0609_fk(q):
    T = np.eye(4)
    for i, (xyz, rpy) in enumerate(_J_CFG):
        cq, sq = np.cos(q[i]), np.sin(q[i])
        Ti = np.eye(4)
        Ti[:3, :3] = _rpy(*rpy) @ np.array([[cq,-sq,0],[sq,cq,0],[0,0,1]])
        Ti[:3, 3] = xyz
        T = T @ Ti
    return (T[:3, :3] @ _EE_OFFSET + T[:3, 3]).astype(np.float32)


def m0609_fk_full(q):
    """pos + EE 회전행렬 반환 (body→base 변환용)"""
    T = np.eye(4)
    for i, (xyz, rpy) in enumerate(_J_CFG):
        cq, sq = np.cos(q[i]), np.sin(q[i])
        Ti = np.eye(4)
        Ti[:3, :3] = _rpy(*rpy) @ np.array([[cq,-sq,0],[sq,cq,0],[0,0,1]])
        Ti[:3, 3] = xyz
        T = T @ Ti
    pos = (T[:3, :3] @ _EE_OFFSET + T[:3, 3]).astype(np.float32)
    return pos, T[:3, :3].copy()


def _R_to_quat_wxyz(R):
    """회전행렬 → quaternion (w,x,y,z)"""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


# HOME_POSE 접근 방향 quat (w,x,y,z) — lift_env_cfg.py roll=-1.4753, pitch=0.0, yaw=3.0347
TARGET_APPROACH_QUAT = _R_to_quat_wxyz(_rpy(-1.4753, 0.0, 3.0347)).astype(np.float32)


def m0609_diff_ik(q, dp, damping=1e-3):
    eps = 1e-5
    p0 = m0609_fk(q).astype(np.float64)
    J = np.zeros((3, 6))
    for i in range(6):
        dq = q.copy(); dq[i] += eps
        J[:, i] = (m0609_fk(dq).astype(np.float64) - p0) / eps
    return (q + J.T @ np.linalg.solve(J@J.T + damping**2*np.eye(3), dp)).astype(np.float32)


# ── 정책 모델 ────────────────────────────────────────────────────────────────
class ActorMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(26, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 64),  torch.nn.ELU(),
            torch.nn.Linear(64, 3),
        )
    def forward(self, x): return self.mlp(x)


def load_policy(path, device="cpu"):
    path = Path(path)
    jit = path.parent / "exported" / "policy.pt"
    if jit.exists():
        return torch.jit.load(str(jit), map_location=device)
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model = ActorMLP().to(device)
    model.mlp.load_state_dict(
        {k.replace("mlp.", "", 1): v
         for k, v in ckpt["actor_state_dict"].items() if k.startswith("mlp.")}
    )
    model.eval()
    return model


# ── 메인 노드 ────────────────────────────────────────────────────────────────
class HarvestStrawberryNode(Node):
    def __init__(self):
        super().__init__("harvest_strawberry_action")

        self._policy      = load_policy(DEFAULT_POLICY)
        self._joint_state = None
        self._busy        = False
        self._current_q   = READY_POSE.copy().astype(np.float64)  # 누적 관절각 (Isaac Sim은 feedback 없음)

        # CSV 로그 초기화
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = f"/tmp/harvest_log_{ts}.csv"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "phase", "step",
            "ee_x", "ee_y", "ee_z",
            "tgt_x", "tgt_y", "tgt_z",
            "dist",
            "act_x", "act_y", "act_z",
            "dp_base_x", "dp_base_y", "dp_base_z",
            "q1", "q2", "q3", "q4", "q5", "q6",
        ])
        self.get_logger().info(f"CSV 로그: {self._csv_path}")

        # 제어 루프 상태
        self._phase            = 0     # 0: 딸기 접근(PPO), 1: 바구니 이동(PPO)
        self._target           = None
        self._step             = 0
        self._last_action      = np.zeros(3, dtype=np.float32)  # dx,dy,dz
        self._ctrl_timer       = None
        self._ready_timer      = None

        self._basket_count      = 0      # 바구니에 넣은 딸기 수
        self._basket_full       = False  # 창고 이동 요청 보냈는지 (중복 방지)
        self._waiting_warehouse = False  # 창고 응답 대기 중 (True면 수확 차단)
        self._warehouse_cli     = self.create_client(SetBool, "/go_return_warehouse")

        # 딸기 좌표 수집 상태
        self._collecting       = False  # /harvest_request 수집 중 여부
        self._collect_timer    = None   # 수집 종료 타이머
        self._strawberry_queue = []     # 수집된 딸기 좌표 큐 [(x,y,z), ...]


        self._joint_cmd_pub  = self.create_publisher(JointState, "/robot1/arm/joint_command", 10)
        self._gripper_pub    = self.create_publisher(Float64, "/robot1/gripper/command", 10)
        self._harvest_done_pub = self.create_publisher(Bool, "/harvest_done", 10)
        self._joint_sub      = self.create_subscription(JointState, "/dsr01/joint_states", self._joint_cb, 10)
        self._harvest_sub    = self.create_subscription(String, "/call_quadruped", self._harvest_cb, 10)
        self._request_sub    = self.create_subscription(String, "/harvest_request", self._harvest_request_cb, 10)

        self.get_logger().info("HarvestStrawberry 대기 중... (/call_quadruped, /harvest_request)")

    # ── 관절 상태 수신 ────────────────────────────────────────────────────────
    def _joint_cb(self, msg):
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
            z = float(data["z"]) - 0.05  # 5cm 앞에서 정지
        except Exception as e:
            self.get_logger().error(f"파싱 오류: {e}")
            return
        # 같은 딸기가 여러 프레임에서 반복 감지되므로 DEDUP_DIST 이내면 무시
        new_pos = np.array([x, y, z], dtype=np.float32)
        for qx, qy, qz in self._strawberry_queue:
            if np.linalg.norm(new_pos - np.array([qx, qy, qz], dtype=np.float32)) < DEDUP_DIST:
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
        # cam_z(깊이) 오름차순 정렬 → 가까운 딸기부터 수확
        self._strawberry_queue.sort(key=lambda p: p[2])
        self.get_logger().info(
            f"  수집 완료: 총 {len(self._strawberry_queue)}개 → 가까운 순 정렬"
        )
        for i, (x, y, z) in enumerate(self._strawberry_queue):
            self.get_logger().info(f"    [{i+1}] cam_z={z:.3f}m  ({x:.3f},{y:.3f},{z:.3f})")
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
            self._publish_harvest_done()
            self._phase = 0
            self._busy  = False

    # ── 수확 시작 공통 처리 ───────────────────────────────────────────────────
    def _start_harvest(self, x: float, y: float, z: float):
        self._target = self._transform_to_arm_base(x, y, z)
        self.get_logger().info(
            f"Ripe 감지! 카메라:({x:.3f},{y:.3f},{z:.3f}) "
            f"→ arm base:({self._target[0]:.3f},{self._target[1]:.3f},{self._target[2]:.3f})"
        )
        self._current_q   = READY_POSE.copy().astype(np.float64)
        self._busy        = True
        self._phase       = 0
        self._step        = 0
        self._last_action = np.zeros(3, dtype=np.float32)
        self._pub_cmd(READY_POSE)
        self._ready_timer = self.create_timer(3.0, self._on_ready)

    # ── READY 대기 완료 → 제어 타이머 시작 ───────────────────────────────────
    def _on_ready(self):
        self._ready_timer.cancel()
        self._ready_timer = None
        self._ctrl_timer = self.create_timer(1.0 / CONTROL_HZ, self._control_step)

    # ── 50Hz 제어 스텝 ────────────────────────────────────────────────────────
    def _control_step(self):
        # Isaac Sim은 /dsr01/joint_states를 발행하지 않으므로 누적 관절각 사용
        q_arm     = self._current_q                                         # (6,) float64
        joint_pos = np.concatenate([q_arm.astype(np.float32),
                                    np.zeros(2, dtype=np.float32)])         # (8,)
        joint_vel = np.zeros(8, dtype=np.float32)
        ee        = m0609_fk(q_arm)
        dist      = float(np.linalg.norm(ee - self._target))

        if self._phase == 0:
            # ── Phase 0: IK 접근 → 근거리(IK_HANDOFF_DIST)에서 PPO 전환 ─────
            tgt_in_ppo_range = (
                PPO_TGT_X_RANGE[0] <= self._target[0] <= PPO_TGT_X_RANGE[1] and
                PPO_TGT_Y_RANGE[0] <= self._target[1] <= PPO_TGT_Y_RANGE[1] and
                PPO_TGT_Z_RANGE[0] <= self._target[2] <= PPO_TGT_Z_RANGE[1]
            )
            use_ppo = tgt_in_ppo_range or (dist <= IK_HANDOFF_DIST)

            if use_ppo:
                # ── PPO 제어 ──────────────────────────────────────────────────
                obs = torch.tensor(np.concatenate([
                    joint_pos - DEFAULT_JOINT_POS,
                    joint_vel,
                    np.concatenate([self._target, TARGET_APPROACH_QUAT]),
                    self._last_action,
                ]), dtype=torch.float32).unsqueeze(0)
                with torch.inference_mode():
                    action = self._policy(obs).squeeze(0).cpu().numpy()
                action = np.clip(action, -1.0, 1.0)   # 발산 방지
                _, R_ee = m0609_fk_full(q_arm)
                dp_base = R_ee @ (action.astype(np.float64) * IK_SCALE)
                mode_str = "PPO"
            else:
                # ── IK 접근 (target이 PPO 훈련 범위 밖이고 아직 멀 때) ───────
                direction = (self._target.astype(np.float64) - ee.astype(np.float64))
                direction /= np.linalg.norm(direction)
                dp_base = direction * IK_SCALE
                action = np.zeros(3, dtype=np.float32)   # CSV용 더미
                mode_str = "IK "

            new_q = m0609_diff_ik(q_arm, dp_base)
            self._current_q = new_q
            self._pub_cmd(new_q)
            self._csv_writer.writerow([
                0, self._step,
                round(float(ee[0]),4), round(float(ee[1]),4), round(float(ee[2]),4),
                round(float(self._target[0]),4), round(float(self._target[1]),4), round(float(self._target[2]),4),
                round(dist,4),
                round(float(action[0]),4), round(float(action[1]),4), round(float(action[2]),4),
                round(float(dp_base[0]),5), round(float(dp_base[1]),5), round(float(dp_base[2]),5),
                *[round(float(v),4) for v in new_q],
            ])
            self._csv_file.flush()
            self.get_logger().info(
                f"[P0/{self._step:3d}|{mode_str}] EE=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})  dist={dist:.4f}m"
            )

            if dist < ARRIVE_THRESH:
                self.get_logger().info("[GRASP] 딸기 도달 → 그리퍼 닫기")
                self._close_gripper()
                self._phase       = 1
                self._step        = 0
                self._last_action = np.zeros(3, dtype=np.float32)
                self._target      = BASKET_POS.copy()
                return

            self._last_action = action.astype(np.float32)
            self._step += 1
            if self._step >= MAX_STEPS:
                self.get_logger().warning(f"[TIMEOUT] phase0 max_steps 초과")
                self._stop_control()

        else:
            # ── Phase 1: IK→PPO로 바구니 이동 ────────────────────────────────
            tgt_in_ppo_range = (
                PPO_TGT_X_RANGE[0] <= self._target[0] <= PPO_TGT_X_RANGE[1] and
                PPO_TGT_Y_RANGE[0] <= self._target[1] <= PPO_TGT_Y_RANGE[1] and
                PPO_TGT_Z_RANGE[0] <= self._target[2] <= PPO_TGT_Z_RANGE[1]
            )
            use_ppo = tgt_in_ppo_range or (dist <= IK_HANDOFF_DIST)

            if use_ppo:
                obs = torch.tensor(np.concatenate([
                    joint_pos - DEFAULT_JOINT_POS,
                    joint_vel,
                    np.concatenate([self._target, TARGET_APPROACH_QUAT]),
                    self._last_action,
                ]), dtype=torch.float32).unsqueeze(0)
                with torch.inference_mode():
                    action = self._policy(obs).squeeze(0).cpu().numpy()
                action = np.clip(action, -1.0, 1.0)   # 발산 방지
                _, R_ee = m0609_fk_full(q_arm)
                dp_base = R_ee @ (action.astype(np.float64) * IK_SCALE)
                mode_str = "PPO"
            else:
                direction = (self._target.astype(np.float64) - ee.astype(np.float64))
                direction /= np.linalg.norm(direction)
                dp_base = direction * IK_SCALE
                action = np.zeros(3, dtype=np.float32)
                mode_str = "IK "

            new_q = m0609_diff_ik(q_arm, dp_base)
            self._current_q = new_q
            self._pub_cmd(new_q)
            self._csv_writer.writerow([
                1, self._step,
                round(float(ee[0]),4), round(float(ee[1]),4), round(float(ee[2]),4),
                round(float(self._target[0]),4), round(float(self._target[1]),4), round(float(self._target[2]),4),
                round(dist,4),
                round(float(action[0]),4), round(float(action[1]),4), round(float(action[2]),4),
                round(float(dp_base[0]),5), round(float(dp_base[1]),5), round(float(dp_base[2]),5),
                *[round(float(v),4) for v in new_q],
            ])
            self._csv_file.flush()
            self.get_logger().info(
                f"[P1/{self._step:3d}|{mode_str}] EE=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})  dist={dist:.4f}m"
            )

            if dist < ARRIVE_THRESH:
                self.get_logger().info("[PLACE] 바구니 도달 → 그리퍼 열기")
                self._open_gripper()
                self._pub_cmd(READY_POSE)
                if self._ctrl_timer is not None:
                    self._ctrl_timer.cancel()
                    self._ctrl_timer = None
                self._process_next_from_queue()
                return

            self._last_action = action.astype(np.float32)
            self._step += 1
            if self._step >= MAX_STEPS:
                self.get_logger().warning(f"[TIMEOUT] phase1 max_steps 초과")
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
            

    # ── /go_return_warehouse 응답 수신 ─────────────────────────────────────────
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
            self._publish_harvest_done()
            self._busy = False

    # ── 그리퍼 열기 ──────────────────────────────────────────────────────────
    def _open_gripper(self):
        msg = Float64()
        msg.data = GRIPPER_OPEN_VAL
        self._gripper_pub.publish(msg)
        with open(GRIPPER_CMD_FILE, "w") as f:
            f.write(f"{GRIPPER_OPEN_VAL:.6f}")
        self.get_logger().info(f"[GRIPPER] 열기 → {GRIPPER_OPEN_VAL:.3f} rad")

    # ── /harvest_done 발행 ────────────────────────────────────────────────────
    def _publish_harvest_done(self):
        msg = Bool()
        msg.data = True
        self._harvest_done_pub.publish(msg)
        self.get_logger().info("[DONE] /harvest_done True 발행 → straight.py")

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

    # ── 카메라→arm 하드코딩 변환 ─────────────────────────────────────────────
    def _transform_to_arm_base(self, x, y, z) -> np.ndarray:
        cam_point = np.array([x, y, z], dtype=np.float32)
        arm_point = CAM_TO_ARM_R @ cam_point + CAM_TO_ARM_T
        x_ok = -0.4 <= arm_point[0] <= 0.4
        y_ok = -0.9 <= arm_point[1] <= -0.6
        z_ok = 0.05 <= arm_point[2] <= 0.30
        self.get_logger().info(
            f"카메라→arm: ({x:.3f},{y:.3f},{z:.3f}) → "
            f"({arm_point[0]:.3f},{arm_point[1]:.3f},{arm_point[2]:.3f})  "
            f"[x={'OK' if x_ok else 'OUT'} y={'OK' if y_ok else 'OUT'} z={'OK' if z_ok else 'OUT'}]"
        )
        return arm_point

    # ── 관절 파싱 ─────────────────────────────────────────────────────────────
    def _get_joints(self):
        if self._joint_state is None:
            return DEFAULT_JOINT_POS.copy(), np.zeros(8, dtype=np.float32)
        n2i = {n: i for i, n in enumerate(self._joint_state.name)}
        pos = np.zeros(8, dtype=np.float32)
        vel = np.zeros(8, dtype=np.float32)
        for k, name in enumerate([f"joint_{i}" for i in range(1, 7)] +
                                   ["finger_joint", "left_outer_knuckle_joint"]):
            idx = n2i.get(name)
            if idx is not None:
                pos[k] = self._joint_state.position[idx]
                vel[k] = self._joint_state.velocity[idx]
        return pos, vel

    def _pub_cmd(self, q6):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = [f"joint_{i}" for i in range(1, 7)]
        msg.position = [float(v) for v in q6]
        self._joint_cmd_pub.publish(msg)

    def destroy_node(self):
        self._csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HarvestStrawberryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

