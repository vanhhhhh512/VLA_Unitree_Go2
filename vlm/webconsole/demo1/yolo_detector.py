"""YOLO detector (ultralytics) — detect + bbox, theo cách bên AnhDV/yolo_search.py.

YOLO chạy thẳng trên frame gốc nên toạ độ box đã ở frame-space (không cần scale).
Import ultralytics nằm trong __init__ (lazy) để module import được khi chưa có ultralytics.
"""
import os

_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "models", "yolo11n.pt")


class YoloDetector:
    def __init__(self, weights=None, conf=0.35):
        from ultralytics import YOLO
        self.conf = conf
        self.model = YOLO(weights or _DEFAULT_WEIGHTS)
        self.names = list(self.model.names.values())

    def detect(self, frame_bgr, wanted=None):
        """Trả [(x1, y1, x2, y2, label, conf)] (pixel, frame-space), sort theo conf.
        wanted: set tên lớp muốn giữ; None -> giữ tất cả."""
        res = self.model(frame_bgr, conf=self.conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            label = self.model.names[int(b.cls)]
            if wanted and label not in wanted:
                continue
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            out.append((x1, y1, x2, y2, label, float(b.conf)))
        out.sort(key=lambda t: -t[5])
        return out
