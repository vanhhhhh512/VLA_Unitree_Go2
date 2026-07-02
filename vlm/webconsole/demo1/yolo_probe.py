"""Soi YOLO thấy gì trên 1 frame camera hiện tại (để biết vật được gán nhãn gì).

Chạy:
  source /opt/ros/jazzy/setup.bash && source ~/ros2_vlm/install/setup.bash
  cd ~/ros2_vlm/src
  python3.12 -m webconsole.demo1.yolo_probe          # conf 0.10, imgsz 1280
  CONF=0.05 IMGSZ=1280 python3.12 -m webconsole.demo1.yolo_probe
"""
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from .yolo_detector import YoloDetector


def main():
    conf = float(os.getenv("CONF", "0.10"))
    os.environ.setdefault("VLA_YOLO_IMGSZ", os.getenv("IMGSZ", "1280"))
    rclpy.init()
    node = Node("yolo_probe")
    bridge = CvBridge()
    box = {"frame": None}
    node.create_subscription(
        Image, "/camera/image_raw",
        lambda m: box.__setitem__("frame", bridge.imgmsg_to_cv2(m, "bgr8")),
        qos_profile_sensor_data,
    )
    print("Đợi frame camera…")
    t0 = time.time()
    while box["frame"] is None and time.time() - t0 < 10:
        rclpy.spin_once(node, timeout_sec=0.1)
    if box["frame"] is None:
        print("KHÔNG nhận được frame /camera/image_raw (driver chạy chưa? topic đúng?)")
        return

    det = YoloDetector(conf=conf)
    dets = det.detect(box["frame"], wanted=None)        # KHÔNG lọc lớp -> thấy hết
    h, w = box["frame"].shape[:2]
    print(f"\nFrame {w}x{h}, conf>={conf}, imgsz={det.imgsz} — YOLO thấy {len(dets)} vật:")
    for x1, y1, x2, y2, label, c in dets:
        cx = (x1 + x2) / 2
        side = "giữa" if abs(cx - w / 2) < w * 0.1 else ("phải" if cx > w / 2 else "trái")
        print(f"  - {label:15s} conf={c:.2f}  box=({x1},{y1},{x2},{y2})  ~{side}")
    if not any("microwave" in d[4] for d in dets):
        print("\n>>> KHÔNG có 'microwave'. Vật bạn muốn có thể đang mang nhãn khác ở trên.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
