#!/usr/bin/env python3
"""
straight.py  (PatrolNode)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Robot1(Nova Carter) 순찰 제어 노드.

시나리오:
  1. /tmp/growth_done.txt 파일이 생길 때까지 대기 (성장 완료 신호)
  2. SEGMENTS(5)개 구간을 순서대로 순회
     - 각 구간 앞에서 PAUSE_SEC(3초) 동안 /harvest_request 구독
     - ripe 딸기 감지 → /call_quadruped 에 JSON 발행 → Robot2 호출
     - 감지 없으면 다음 구간으로 직진
  3. 전체 순회 완료 후 종료

입력 토픽:
  /harvest_request  (std_msgs/String)  ← yolo_detector_node_4.py
  /harvest_done     (std_msgs/Bool)    ← Robot1 팔 제어 노드 (팔이 바구니에 딸기를 넣은 후 True 발행)

출력 토픽:
  /cmd_vel          (geometry_msgs/Twist)  → Nova Carter 구동
  /call_quadruped   (std_msgs/String)      → robot2_go_to_pot_by_number.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
import time
import json
import os

DRIVE_FRAMES = 400   # 직진 유지 프레임 수 (370 × 0.05s = 18.5초 직진)
PAUSE_SEC    = 3.0   # 각 정차 위치에서 딸기 감지를 기다리는 시간 (초)
SEGMENTS     = 5     # 총 정차 구간 수 (화분 개수와 일치)


class PatrolNode(Node):
    def __init__(self):
        super().__init__("patrol_node")

        # Robot1(Nova Carter) 구동 명령 퍼블리셔
        self._pub = self.create_publisher(Twist, "/robot1/cmd_vel", 10)

        # Robot2(Spot) 호출 메시지 퍼블리셔 (JSON: strawberry 좌표 + pot_stop 번호)
        self._quad_pub = self.create_publisher(String, "/call_quadruped", 10)

        self._ripe_detected   = False  # 현재 구간에서 ripe 감지 여부
        self._ripe_data       = None   # 감지된 딸기의 3D 좌표 dict
        self._sub             = None   # /harvest_request 구독자 (구간별 생성/소멸)
        self._harvest_done    = False  # Robot1 수확 완료 여부

        # /harvest_done 상시 구독 (Robot1가 수확 완료 시 발행)
        self.create_subscription(Bool, "/harvest_done", self._cb_harvest_done, 10)

    # ── 성장 완료 대기 ─────────────────────────
    def wait_for_growth(self):
        """
        Isaac Sim의 타임랩스가 끝나면 /tmp/growth_done.txt 파일을 생성.
        해당 파일이 존재할 때까지 ROS2 스핀을 유지하며 대기.
        실행 전 기존 파일은 삭제해 이전 실행과 혼동을 방지.
        """
        DONE_FILE = "/tmp/growth_done.txt"
        if os.path.exists(DONE_FILE):
            os.remove(DONE_FILE)
        self.get_logger().info("🌱 성장 완료 대기 중... (/tmp/growth_done.txt)")
        while not os.path.exists(DONE_FILE):
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("✅ 성장 완료! 정찰 시작!")

    # ── 직진 ───────────────────────────────────
    def drive(self):
        """
        DRIVE_FRAMES 동안 선속도 0.3m/s로 직진.
        각 프레임마다 0.05초 대기하므로 총 이동 시간 ≈ 18.5초.
        """
        cmd = Twist()
        cmd.linear.x  = 0.3
        cmd.angular.z = 0.0
        for _ in range(DRIVE_FRAMES):
            self._pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.05)

    # ── 정지 ───────────────────────────────────
    def stop(self):
        """20프레임 동안 Twist() 정지 명령을 발행해 Robot1을 확실히 정지."""
        for _ in range(20):
            self._pub.publish(Twist())
            rclpy.spin_once(self, timeout_sec=0.05)

    # ── 수확 완료 콜백 ─────────────────────────
    def _cb_harvest_done(self, msg: Bool):
        """
        /harvest_done 콜백.
        Robot1 팔 제어 노드가 딸기를 바구니에 넣은 후 True 발행.
        True 수신 시 wait_harvest_done 루프 탈출 → 다음 화분으로 이동.
        False가 오면 무시하고 계속 대기.
        """
        if msg.data:
            self._harvest_done = True
            self.get_logger().info("  ✅ 수확 완료 신호 수신 → 다음 화분으로 이동")

    # ── Robot2 수확 완료 대기 ───────────────────
    def wait_harvest_done(self):
        """
        /harvest_done 수신까지 무한 대기.
        신호가 오지 않으면 Robot1은 계속 정지 상태 유지.
        """
        self._harvest_done = False
        self.get_logger().info("  ⏳ 수확 완료 대기 중... (/harvest_done)")
        while not self._harvest_done:
            self._pub.publish(Twist())  # 대기 중 계속 정지 명령 발행
            rclpy.spin_once(self, timeout_sec=0.05)

    # ── ripe 감지 콜백 ─────────────────────────
    def _cb_ripe(self, msg):
        """
        /harvest_request 콜백. 한 구간에서 최초 감지 1회만 처리.
        msg.data: JSON 문자열 {"x": float, "y": float, "z": float}
        """
        if not self._ripe_detected:
            self._ripe_detected = True
            self._ripe_data = json.loads(msg.data)
            self.get_logger().info(
                f"  🍓 Ripe 감지! X={self._ripe_data['x']}m "
                f"Y={self._ripe_data['y']}m Z={self._ripe_data['z']}m"
            )

    # ── 정차 후 감지 ────────────────────────────
    def check_ripe(self):
        """
        PAUSE_SEC 동안 /harvest_request 토픽을 구독해 ripe 딸기를 감지.
        반환: (감지여부: bool, 딸기 좌표 dict 또는 None)
        구독은 이 함수 내에서만 유효하며 종료 후 destroy.
        """
        self._ripe_detected = False
        self._ripe_data     = None
        # 구간마다 새로 구독 생성 (이전 구간의 신호가 섞이지 않게)
        self._sub = self.create_subscription(
            String, "/harvest_request", self._cb_ripe, 10
        )
        self.get_logger().info(f"  감지 중... ({PAUSE_SEC}초)")
        deadline = time.time() + PAUSE_SEC
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        self.destroy_subscription(self._sub)
        self._sub = None
        return self._ripe_detected, self._ripe_data

    # ── Robot2(Spot) 호출 ───────────────────────
    def call_quadruped(self, ripe_data, stop_idx):
        """
        /call_quadruped 토픽에 JSON 메시지 발행.
        robot2_go_to_pot_by_number.py가 수신해 Nav2로 해당 화분으로 이동.

        ripe_data:  {"x": float, "y": float, "z": float}  (카메라 기준 상대 좌표)
        stop_idx:   화분 번호 (1~5)
        """
        msg = String()
        msg.data = json.dumps({
            "strawberry": ripe_data,   # 딸기 상대 좌표 (카메라 기준)
            "pot_stop": stop_idx,      # 몇 번째 정차 위치 (1~5)
        })
        self._quad_pub.publish(msg)
        self.get_logger().info(
            f"  🐾 사족보행 호출! pot_stop={stop_idx} → /call_quadruped"
        )

    # ── 메인 순찰 루프 ──────────────────────────
    def run(self):
        """
        성장 완료 대기 → SEGMENTS 구간 순차 순회.
        각 구간: 감지(check_ripe) → (ripe면 Robot2 호출) → 직진(마지막 제외)
        """
        self.wait_for_growth()
        self.get_logger().info("정찰 시작!")

        for i in range(SEGMENTS):
            # 현재 구간에서 딸기 감지 시도
            self.get_logger().info(f"\n[{i+1}/{SEGMENTS}] 감지...")
            ripe, data = self.check_ripe()

            if ripe:
                # 익은 딸기 발견 → Robot2 호출 후 수확 완료 신호 대기
                self.call_quadruped(data, i + 1)
                self.wait_harvest_done()
            else:
                self.get_logger().info("  미숙")

            # 마지막 구간이 아니면 다음 화분 위치로 직진
            if i < SEGMENTS - 1:
                self.get_logger().info(f"  직진...")
                self.drive()
                self.stop()

        self.get_logger().info("\n✅ 정찰 완료!")


def main():
    rclpy.init()
    node = PatrolNode()
    try:
        node.run()
    except KeyboardInterrupt:
        # Ctrl+C 시 즉시 정지 명령 발행
        node._pub.publish(Twist())
        node.get_logger().info("중단!")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
