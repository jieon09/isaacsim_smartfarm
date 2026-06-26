"""
robot2_state_udp_bridge.py  (Robot2StateUdpBridge)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Isaac Sim UDP 상태 → ROS2 Odometry / tf 변환 브릿지 (Robot2 Spot).

main_sim.py의 UdpStateSender가 50Hz로 보내는 Spot 위치·yaw를
ROS2 Nav2가 사용할 수 있는 Odometry, PoseStamped, tf 로 변환해 발행.

입력:
  UDP 127.0.0.1:5006  JSON {"x", "y", "z", "yaw"}  ← main_sim.py

출력 토픽:
  /robot2/odom  (nav_msgs/Odometry)         → Nav2 amcl / costmap
  /robot2/pose  (geometry_msgs/PoseStamped) → 디버그·시각화
  /tf           (robot2/odom → robot2/base_link)  → Nav2 tf 트리
  /tf_static    (map → robot2/odom, 항등 변환)    → Nav2 tf 트리
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import math
import socket

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster


def angle_diff(a, b):
    """두 각도의 차이를 [-π, π] 범위로 정규화. 각속도 계산 시 wrap-around 방지."""
    return math.atan2(math.sin(a - b), math.cos(a - b))

def quat_from_yaw(yaw):
    """yaw 각도(rad)만 가진 2D 쿼터니언 (x=0, y=0, z=sin(yaw/2), w=cos(yaw/2)) 반환."""
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return 0.0, 0.0, qz, qw


class Robot2StateUdpBridge(Node):
    def __init__(self):
        super().__init__("robot2_state_udp_bridge")

        # 이전 상태 저장용 (속도 수치 미분 계산에 사용)
        self.last_state = None  # (x, y, yaw, timestamp_sec)

        # UDP 수신 소켓 (Isaac Sim의 UdpStateSender가 포트 5006으로 송신)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 5006))
        self.sock.setblocking(False)  # 타이머 콜백을 블로킹하지 않음

        # ROS2 퍼블리셔
        self.odom_pub = self.create_publisher(Odometry, "/robot2/odom", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/robot2/pose", 10)

        # 동적 tf 브로드캐스터 (robot2/odom → robot2/base_link, 매 수신마다 갱신)
        self.tf_broadcaster = TransformBroadcaster(self)
        # 정적 tf 브로드캐스터 (map → robot2/odom, 항등 변환 1회만 발행)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # map → robot2/odom 항등 정적 tf 발행 (Nav2는 map 프레임 기준으로 계획)
        self._publish_static_map_to_odom()

        # 50Hz 타이머로 UDP 소켓 폴링
        self.timer = self.create_timer(0.02, self.spin_udp)

        self.get_logger().info("UDP 127.0.0.1:5006 → /robot2/odom + /tf")

    def _publish_static_map_to_odom(self):
        """
        map → robot2/odom 항등 변환을 정적 tf로 1회 발행.
        Isaac Sim 좌표계 = ROS2 map 좌표계로 가정하므로 변환 없음.
        """
        now = self.get_clock().now().to_msg()

        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = "map"
        t.child_frame_id = "robot2/odom"

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0  # 항등 회전

        self.static_tf_broadcaster.sendTransform(t)

    def spin_udp(self):
        """50Hz 타이머 콜백: UDP 소켓 버퍼를 비우며 최신 상태 패킷을 처리."""
        while True:
            try:
                data, _ = self.sock.recvfrom(2048)
            except BlockingIOError:
                break  # 수신 데이터 없음 → 정상 종료

            msg = json.loads(data.decode("utf-8"))

            x   = float(msg["x"])
            y   = float(msg["y"])
            z   = float(msg["z"])
            yaw = float(msg["yaw"])

            self.publish_state(x, y, z, yaw)

    def publish_state(self, x, y, z, yaw):
        """
        수신된 위치·yaw로부터 속도를 수치 미분 후
        Odometry, PoseStamped, tf를 발행.

        속도 계산:
          - world 프레임 선속도 → robot body 프레임으로 회전 변환
          - 각속도: angle_diff(현재yaw, 이전yaw) / dt
        """
        now_clock = self.get_clock().now()
        now = now_clock.to_msg()
        now_sec = now_clock.nanoseconds * 1e-9

        qx, qy, qz, qw = quat_from_yaw(yaw)

        # 첫 수신이면 속도 0으로 초기화
        if self.last_state is None:
            vx_body = 0.0
            vy_body = 0.0
            wz = 0.0
        else:
            last_x, last_y, last_yaw, last_t = self.last_state
            dt = max(now_sec - last_t, 1e-6)  # 0 나누기 방지

            # world 프레임 속도 (수치 미분)
            vx_world = (x - last_x) / dt
            vy_world = (y - last_y) / dt
            wz = angle_diff(yaw, last_yaw) / dt

            # world → body 프레임 회전 변환 (yaw 회전 역적용)
            cy = math.cos(yaw)
            sy = math.sin(yaw)
            vx_body =  cy * vx_world + sy * vy_world
            vy_body = -sy * vx_world + cy * vy_world

        self.last_state = (x, y, yaw, now_sec)

        # ── Odometry 메시지 구성 ──────────────────
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "robot2/odom"
        odom.child_frame_id = "robot2/base_link"

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x  = vx_body
        odom.twist.twist.linear.y  = vy_body
        odom.twist.twist.angular.z = wz

        self.odom_pub.publish(odom)

        # ── PoseStamped (map 프레임, 디버그용) ───────
        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = "map"
        pose.pose = odom.pose.pose
        self.pose_pub.publish(pose)

        # ── 동적 tf: robot2/odom → robot2/base_link ──
        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = "robot2/odom"
        tf.child_frame_id = "robot2/base_link"
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.translation.z = z
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(tf)


def main():
    rclpy.init()
    node = Robot2StateUdpBridge()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
