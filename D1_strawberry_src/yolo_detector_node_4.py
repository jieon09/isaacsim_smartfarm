"""
yolo_detector_node_4.py  (YoloDetectorNode)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Nova Carter 카메라 영상에서 YOLO로 익은(ripe) 딸기를 감지하고
3D 좌표를 계산해 /harvest_request 토픽으로 발행.

입력 토픽:
  /left_hawk/left/camera_info   (sensor_msgs/CameraInfo)  ← Isaac Sim
  /left_hawk/left/depth/image_raw (sensor_msgs/Image)     ← Isaac Sim
  /left_hawk/left/image_raw     (sensor_msgs/Image)       ← Isaac Sim

출력 토픽:
  /harvest_request  (std_msgs/String)  → straight.py (PatrolNode)
  (화면)            OpenCV 윈도우에 바운딩박스 + 3D 좌표 표시
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import message_filters
from ultralytics import YOLO
import os
import numpy as np
import json


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.get_logger().info("YOLO Detector Node 시작...")

        # RGB 이미지를 scale_factor 배 확대 후 YOLO 추론
        # 작은 딸기 바운딩박스 검출 정확도 향상용
        self.scale_factor = 1.5

        # 이 픽셀 면적(확대 이미지 기준) 미만인 박스는 위치 발행 안 함
        self.MIN_BOX_AREA = 500
        self.bridge = CvBridge()

        # 실행 파일 기준 상대 경로에서 YOLO 가중치 로드
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(base_dir, "resource", "best_10n_isaac.pt")
        try:
            self.model = YOLO(weights_path)
            self.get_logger().info(f"YOLO 모델 로드 성공: {weights_path}")
        except Exception as e:
            self.get_logger().error(f"YOLO 모델 로드 실패: {e}")

        # 감지 결과(3D 좌표) 퍼블리셔
        self._ripe_pub = self.create_publisher(String, "/harvest_request", 10)


        # CameraInfo, Depth, RGB 세 토픽을 타임스탬프 기준으로 동기화
        info_sub  = message_filters.Subscriber(self, CameraInfo, '/robot1/left_hawk/left/camera_info')
        depth_sub = message_filters.Subscriber(self, Image,      '/robot1/left_hawk/left/depth/image_raw')
        rgb_sub   = message_filters.Subscriber(self, Image,      '/robot1/left_hawk/left/image_raw')

        # 세 토픽의 타임스탬프 차이가 slop(0.1초) 이내인 메시지들을 묶어 콜백 호출
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [info_sub, depth_sub, rgb_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

    def preprocess_for_glare(self, bgr_image, gamma=0.6, saturation_scale=1.5):
        """
        Isaac Sim 조명 과노출(글레어) 보정을 위한 전처리.
        1) 감마 보정으로 밝은 영역 어둡게 (gamma < 1.0 → 어둡게)
        2) HSV 채도 증가로 색상 대비 향상 → 빨간 딸기 검출률 상승
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i/255.0)**inv_gamma)*255 for i in np.arange(256)]).astype("uint8")
        gamma_corrected = cv2.LUT(bgr_image, table)
        hsv = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(cv2.multiply(s, saturation_scale), 0, 255).astype(hsv.dtype)
        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    def sync_callback(self, info_msg, depth_msg, rgb_msg):
        """
        동기화된 CameraInfo + Depth + RGB 수신 콜백.

        처리 순서:
          1) RGB 이미지 전처리 (글레어 보정) + scale_factor 배 확대
          2) YOLO 추론 → 클래스 'ripe' 바운딩박스 탐색
          3) 바운딩박스 중심의 깊이값(z) 읽기
          4) 핀홀 역투영 공식으로 3D 좌표 (x, y, z) 계산
          5) /harvest_request 토픽 발행
          6) OpenCV 창에 결과 표시
        """
        try:
            # ROS2 Image 메시지 → OpenCV BGR 이미지 변환
            raw_cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image  = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            orig_h, orig_w = raw_cv_image.shape[:2]

            # 전처리 후 1.5배 확대 (작은 열매 검출 향상)
            preprocessed  = self.preprocess_for_glare(raw_cv_image)
            new_w = int(orig_w * self.scale_factor)
            new_h = int(orig_h * self.scale_factor)
            scaled = cv2.resize(preprocessed, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # CameraInfo에서 핀홀 파라미터 추출 (K 행렬)
            # K = [fx,  0, cx,
            #       0, fy, cy,
            #       0,  0,  1]
            fx = info_msg.k[0]; cx = info_msg.k[2]
            fy = info_msg.k[4]; cy = info_msg.k[5]

            # YOLO 추론 (확대 이미지 기준)
            results        = self.model(scaled, verbose=False, conf=0.5)
            annotated_frame = results[0].plot()
            depth_h, depth_w = depth_image.shape[:2]

            for box in results[0].boxes:
                class_id   = int(box.cls[0].item())
                class_name = self.model.names[class_id]
                conf_val   = float(box.conf[0].item())
                self.get_logger().info(f"감지: {class_name} ({conf_val:.2f})")

                if class_name == 'ripe':
                    # 확대 이미지 기준 바운딩박스 중심 픽셀
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # 박스 면적이 MIN_BOX_AREA 미만이면 너무 멀거나 작은 딸기 → 발행 스킵
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area < self.MIN_BOX_AREA:
                        self.get_logger().debug(f"박스 너무 작음 ({box_area:.0f}px²) → 스킵")
                        continue

                    u_scaled = int((x1 + x2) / 2)
                    v_scaled = int((y1 + y2) / 2)

                    # 깊이 이미지는 원본 해상도 → 스케일 역변환
                    u_orig = int(np.clip(u_scaled / self.scale_factor, 0, depth_w-1))
                    v_orig = int(np.clip(v_scaled / self.scale_factor, 0, depth_h-1))

                    z_raw = depth_image[v_orig, u_orig]
                    if z_raw == 0 or z_raw != z_raw:  # 0 또는 NaN이면 스킵
                        continue

                    # float 타입이면 미터, uint16이면 mm → m 변환
                    z = float(z_raw) if depth_image.dtype.kind == 'f' else float(z_raw)/1000.0

                    # 핀홀 역투영: 픽셀 좌표 + 깊이 → 카메라 좌표계 3D 좌표
                    # x = (u - cx) * z / fx
                    # y = (v - cy) * z / fy
                    x = (u_orig - cx) * z / fx
                    y = (v_orig - cy) * z / fy

                    self.get_logger().info(
                        f"Ripe 탐지됨! 상대 좌표: X={x:.3f}m, Y={y:.3f}m, Z={z:.3f}m"
                    )

                    # /harvest_request 토픽 발행 (straight.py가 수신)
                    msg = String()
                    msg.data = json.dumps({
                        "x": round(x, 3),
                        "y": round(y, 3),
                        "z": round(z, 3)
                    })
                    self._ripe_pub.publish(msg)

                    # OpenCV 창에 3D 좌표 텍스트 오버레이
                    text = f"X:{x:.2f}m Y:{y:.2f}m Z:{z:.2f}m"
                    cv2.putText(annotated_frame, text, (u_scaled, v_scaled-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            # YOLO 결과 시각화 창 표시 (q 키로 종료)
            cv2.imshow("YOLO Final Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"콜백 오류: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("종료")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
