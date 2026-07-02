"""RosFrameSource: subscribe /camera/image_raw -> latest BGR frame."""


def _waiting_frame(text="Dang cho camera /camera/image_raw..."):
    import numpy as np
    import cv2
    frame = np.full((480, 640, 3), 245, dtype=np.uint8)
    cv2.putText(frame, text, (30, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)
    return frame


class RosFrameSource:
    def __init__(self, node, image_topic="/camera/image_raw"):
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        # Camera của Go2 publish bằng QoS BEST_EFFORT (sensor data); subscriber
        # phải khớp, nếu dùng RELIABLE mặc định sẽ không nhận được frame nào.
        from rclpy.qos import qos_profile_sensor_data
        self.node = node
        self.bridge = CvBridge()
        self._latest = _waiting_frame()  # placeholder để web không quay vô tận
        self._got = False
        self.sub = node.create_subscription(
            Image, image_topic, self._on_image, qos_profile_sensor_data
        )

    def _on_image(self, msg):
        self._latest = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._got = True

    @property
    def is_connected(self):
        return self._got

    def get_latest_frame(self):
        return self._latest
