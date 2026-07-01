"""YOLO detector (ultralytics) — detect + bbox, theo cách bên AnhDV/yolo_search.py.

YOLO chạy thẳng trên frame gốc nên toạ độ box đã ở frame-space (không cần scale).
Import ultralytics nằm trong __init__ (lazy) để module import được khi chưa có ultralytics.
"""
import os
import threading

_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), "models", "yolo11n.pt")


class YoloDetector:
    def __init__(self, weights=None, conf=None):
        from ultralytics import YOLO
        # conf GỐC của YOLO. PHẢI ≤ ngưỡng lọc annotator (VLA_YOLO_MIN_CONF), nếu cao hơn thì
        # YOLO đã cắt vật conf thấp TRƯỚC -> hạ min_conf vô tác dụng. Mặc định bám min_conf.
        if conf is None:
            conf = float(os.getenv("VLA_YOLO_CONF",
                                   os.getenv("VLA_YOLO_MIN_CONF", "0.25")))
        self.conf = conf
        # Model: yolo11n (nano) nhận YẾU. Đổi to hơn để nhận tốt hơn nhiều (RTX thừa sức):
        #   VLA_YOLO_WEIGHTS=yolo11s.pt | yolo11m.pt | yolo11l.pt | yolo11x.pt (tự tải về).
        weights = weights or os.getenv("VLA_YOLO_WEIGHTS", "") or _DEFAULT_WEIGHTS
        self.model = YOLO(weights)
        self.names = list(self.model.names.values())
        # imgsz LỚN = nhận vật nhỏ/ở XA tốt hơn (giữ độ phân giải). 960 mặc định; nâng
        # 1280 nếu cần xa hơn (chậm hơn chút), hạ 640 nếu YOLO lag.
        self.imgsz = int(os.getenv("VLA_YOLO_IMGSZ", "960"))
        # device='cuda' nếu có GPU (ultralytics tự chọn, ép cho chắc).
        self.device = os.getenv("VLA_YOLO_DEVICE", "")
        # Serialize: annotator (live feed) + navloop (gợi ý vật cản) cùng gọi 1 model.
        self._lock = threading.Lock()

    def detect(self, frame_bgr, wanted=None):
        """Trả [(x1, y1, x2, y2, label, conf)] (pixel, frame-space), sort theo conf.
        wanted: set tên lớp muốn giữ; None -> giữ tất cả."""
        kw = {"conf": self.conf, "verbose": False, "imgsz": self.imgsz}
        if self.device:
            kw["device"] = self.device
        with self._lock:
            res = self.model(frame_bgr, **kw)[0]
        out = []
        for b in res.boxes:
            label = self.model.names[int(b.cls)]
            if wanted and label not in wanted:
                continue
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            out.append((x1, y1, x2, y2, label, float(b.conf)))
        out.sort(key=lambda t: -t[5])
        return out
