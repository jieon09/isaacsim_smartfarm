"""
robot2_go_to_pot_by_number.py  (Robot2GoToPotByNumber)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Robot2(Spot)의 Nav2 자율주행 노드.

straight.py(PatrolNode)가 /call_quadruped 에 발행한 JSON 메시지를 수신해
pot_stop 번호에 대응하는 map 좌표로 Nav2 NavigateToPose 액션을 실행.

입력 토픽:
  /call_quadruped  (std_msgs/String)  ← straight.py
  JSON 형식: {"strawberry": {"x","y","z"}, "pot_stop": int}

출력 토픽:
  /robot2/pot_goal_status  (std_msgs/String)  → 상태 로그 (IDLE/SEND GOAL/ARRIVED/FAILED)

출력 액션:
  /navigate_to_pose  (nav2_msgs/action/NavigateToPose)  → Nav2 BT Navigator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from action_msgs.msg import GoalStatus
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose


# ── 토픽 / 액션 이름 ──────────────────────────────

# Robot1이 Robot2를 호출할 때 사용하는 토픽 (JSON String)
CALL_QUADRUPED_TOPIC = "/call_quadruped"

# Robot2 내비게이션 상태를 외부에 알리는 토픽
STATUS_TOPIC = "/robot2/pot_goal_status"

# Nav2 BT Navigator 액션 서버 이름
NAV2_ACTION_NAME = "/navigate_to_pose"


# ── 화분 번호별 Nav2 목표 좌표 (map 프레임) ─────────
# RViz에서 측정한 각 화분 앞 정차 위치.
# yaw_deg=180 → 화분을 바라보는 방향
POT_GOALS = {
    1: {"x": 2.75, "y":  7.09, "yaw_deg": 180.0},
    2: {"x": 2.75, "y":  3.64, "yaw_deg": 180.0},
    3: {"x": 2.75, "y":  0.69, "yaw_deg": 180.0},
    4: {"x": 2.75, "y": -2.35, "yaw_deg": 180.0},
    5: {"x": 2.75, "y": -5.35, "yaw_deg": 180.0},
}


class Robot2GoToPotByNumber(Node):
    def __init__(self):
        super().__init__("robot2_go_to_pot_by_number")

        # Nav2 NavigateToPose 액션 클라이언트
        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            NAV2_ACTION_NAME,
        )

        # 상태 퍼블리셔 (로그 목적, 외부 모니터링용)
        self.status_pub = self.create_publisher(String, STATUS_TOPIC, 10)

        # Robot1의 호출 메시지 구독 (std_msgs/String, JSON 인코딩)
        self.create_subscription(
            String,
            CALL_QUADRUPED_TOPIC,
            self.call_quadruped_callback,
            10,
        )

        self.busy = False        # 현재 내비게이션 진행 중 여부
        self.active_pot = None   # 현재 이동 목표 화분 번호

        self.publish_status("IDLE: waiting for /call_quadruped JSON message")
        self.get_logger().info("Robot2 pot navigation node started")
        self.get_logger().info(f"Listening topic: {CALL_QUADRUPED_TOPIC}")
        self.get_logger().info("Message type: std_msgs/msg/String")
        self.get_logger().info(f"Nav2 action: {NAV2_ACTION_NAME}")
        self.get_logger().info(f"Available pot_stop: {list(POT_GOALS.keys())}")

    def publish_status(self, text: str):
        """상태 문자열을 STATUS_TOPIC에 발행하고 로그에도 출력."""
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)

    def call_quadruped_callback(self, msg: String):
        """
        /call_quadruped 콜백.
        JSON을 파싱해 pot_stop 번호를 추출한 뒤 handle_pot_stop 호출.

        예상 JSON:
        {
          "strawberry": {"x": -1.86, "y": -0.49, "z": 2.34},
          "pot_stop": 3
        }
        """
        raw_text = msg.data.strip()

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            self.publish_status(f"ERROR: invalid JSON from /call_quadruped: {e}")
            self.get_logger().error(f"raw data: {raw_text}")
            return

        if "pot_stop" not in data:
            self.publish_status("ERROR: JSON has no 'pot_stop' field")
            self.get_logger().error(f"received json: {data}")
            return

        try:
            pot_stop = int(data["pot_stop"])
        except Exception:
            self.publish_status(f"ERROR: pot_stop is not int: {data.get('pot_stop')}")
            return

        # strawberry 좌표는 현재 로그 출력만 하고 별도 활용 없음
        # (향후 Spot 팔 제어 등에 활용 가능)
        strawberry = data.get("strawberry", None)
        self.get_logger().info(f"Received /call_quadruped JSON: {data}")

        if strawberry is not None:
            self.get_logger().info(
                "Strawberry position from robot1: "
                f"x={strawberry.get('x')}, "
                f"y={strawberry.get('y')}, "
                f"z={strawberry.get('z')}"
            )

        self.handle_pot_stop(pot_stop)

    def handle_pot_stop(self, pot_stop: int):
        """
        pot_stop 유효성 확인 및 중복 요청 필터링.
        - 알 수 없는 번호 → 오류 반환
        - 이미 해당 화분으로 이동 중 → 중복 무시
        - 다른 화분으로 이동 중 → BUSY 상태 알림 후 무시
        """
        if pot_stop not in POT_GOALS:
            self.publish_status(f"ERROR: unknown pot_stop {pot_stop}")
            return

        if self.busy:
            if pot_stop == self.active_pot:
                self.get_logger().info(
                    f"Ignoring duplicate pot_stop while moving: {pot_stop}"
                )
                return

            self.publish_status(
                f"BUSY: currently moving to pot_stop {self.active_pot}. "
                f"Ignoring new pot_stop {pot_stop}"
            )
            return

        self.send_goal_to_pot(pot_stop)

    def send_goal_to_pot(self, pot_stop: int):
        """
        POT_GOALS에서 목표 좌표를 읽어 NavigateToPose 목표 메시지 생성 후 액션 전송.
        yaw_deg → 쿼터니언 변환 (z, w 성분만 사용하는 2D 회전).
        """
        goal = POT_GOALS[pot_stop]

        x = float(goal["x"])
        y = float(goal["y"])
        yaw_deg = float(goal["yaw_deg"])
        yaw = math.radians(yaw_deg)

        # 2D yaw → 쿼터니언 (x=0, y=0, z=sin(yaw/2), w=cos(yaw/2))
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        # Nav2 액션 서버 준비 여부 확인 (최대 2초 대기)
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.publish_status("ERROR: Nav2 /navigate_to_pose action server not ready")
            self.busy = False
            self.active_pot = None
            return

        # NavigateToPose 목표 메시지 구성
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.busy = True
        self.active_pot = pot_stop

        self.publish_status(
            f"SEND GOAL: pot_stop={pot_stop}, x={x:.3f}, y={y:.3f}, yaw={yaw_deg:.1f}"
        )

        # 비동기로 목표 전송 → goal_response_callback에서 수락 여부 확인
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Nav2가 목표를 수락/거절했을 때 호출. 수락이면 결과 콜백 등록."""
        try:
            goal_handle = future.result()
        except Exception as e:
            self.publish_status(f"GOAL RESPONSE ERROR: {e}")
            self.busy = False
            self.active_pot = None
            return

        if not goal_handle.accepted:
            self.publish_status(f"GOAL REJECTED: pot_stop={self.active_pot}")
            self.busy = False
            self.active_pot = None
            return

        self.publish_status(f"GOAL ACCEPTED: pot_stop={self.active_pot}")

        # 내비게이션 완료까지 비동기 대기
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        """내비게이션 성공/실패 결과 수신 후 busy 플래그 해제."""
        try:
            result_response = future.result()
            status = result_response.status
        except Exception as e:
            self.publish_status(f"GOAL RESULT ERROR: {e}")
            self.busy = False
            self.active_pot = None
            return

        pot = self.active_pot

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.publish_status(f"ARRIVED: pot_stop={pot}")
        else:
            self.publish_status(f"FAILED: pot_stop={pot}, status={status}")

        # 다음 호출을 받을 수 있도록 busy 해제
        self.busy = False
        self.active_pot = None


def main():
    rclpy.init()
    node = Robot2GoToPotByNumber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
