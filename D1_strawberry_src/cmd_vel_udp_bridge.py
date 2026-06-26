"""
cmd_vel_udp_bridge.py  (CmdVelUdpBridge)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROS2 cmd_vel 토픽 → UDP JSON 변환 브릿지 (Robot2 Spot 방향).

Nav2가 /robot2/cmd_vel 등에 발행하는 Twist 메시지를
Isaac Sim이 수신할 수 있는 UDP 패킷(포트 5005)으로 전달.

입력 토픽:
  /robot2/cmd_vel      (geometry_msgs/Twist)  ← Nav2 or 외부 제어기
  /cmd_vel_nav         (geometry_msgs/Twist)  ← (예비)
  /cmd_vel_smoothed    (geometry_msgs/Twist)  ← (예비)

출력:
  UDP 127.0.0.1:5005   JSON {"vx", "vy", "wz"}  → main_sim.py (UdpCmdVelReceiver)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import socket

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdVelUdpBridge(Node):
    def __init__(self):
        super().__init__("cmd_vel_udp_bridge")

        # UDP 송신 소켓 (Isaac Sim의 UdpCmdVelReceiver가 포트 5005로 수신)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target = ("127.0.0.1", 5005)

        # 여러 cmd_vel 토픽을 동일한 콜백으로 처리
        # /robot2/cmd_vel : Nav2가 Robot2(Spot)에 보내는 주 토픽
        # /cmd_vel_nav, /cmd_vel_smoothed : 네비게이션 스택의 보조 토픽 (필요시 주석 해제)
        for topic in [
            "/robot2/cmd_vel",
            # "/cmd_vel",
            "/cmd_vel_nav",
            "/cmd_vel_smoothed",
        ]:
            self.create_subscription(
                Twist,
                topic,
                self.cb_cmd_vel,
                10,
        )
        self.get_logger().info(f"Subscribing {topic} -> UDP 127.0.0.1:5005")

    def cb_cmd_vel(self, msg: Twist):
        """
        Twist 메시지를 JSON으로 직렬화해 UDP 전송.
        Isaac Sim의 UdpCmdVelReceiver가 수신 후 LINEAR_X_SCALE / ANGULAR_Z_SCALE 적용.
        """
        packet = {
            "vx": float(msg.linear.x),
            "vy": float(msg.linear.y),
            "wz": float(msg.angular.z),
        }

        data = json.dumps(packet).encode("utf-8")
        self.sock.sendto(data, self.target)

        self.get_logger().info(
            f"UDP send vx={packet['vx']:.2f}, vy={packet['vy']:.2f}, wz={packet['wz']:.2f}"
        )


def main():
    rclpy.init()
    node = CmdVelUdpBridge()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
