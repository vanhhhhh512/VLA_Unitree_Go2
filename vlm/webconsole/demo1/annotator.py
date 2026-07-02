"""YOLO khoanh vùng (hộp chữ nhật) vật thể MỤC TIÊU theo prompt, lên camera trực tiếp.

Có LỌC NHIỄU để box ổn định (đỡ nhấp nháy -> điểm dừng chính xác):
  - lọc theo độ tin cậy (min_conf),
  - chỉ giữ 1 box tốt nhất (conf cao nhất) cho mục tiêu,
  - làm mượt toạ độ bằng trung bình động EMA qua các frame,
  - mất dấu vài frame mới xoá box.

parse text -> lớp COCO và EMA là hàm/logic thuần (unit-test được).
"""
import os
import re
import threading
import time

import cv2

# Vài từ tiếng Việt -> lớp COCO (để gõ tiếng Việt vẫn khoanh đúng vật).
_VN2COCO = {
    "bình": "bottle", "chai": "bottle", "ghế": "chair", "người": "person",
    "bàn": "dining table", "cốc": "cup", "ly": "cup", "tách": "cup",
    "ba lô": "backpack", "balo": "backpack", "cặp": "backpack",
    "điện thoại": "cell phone", "laptop": "laptop", "máy tính": "laptop",
    "màn hình": "tv", "tivi": "tv", "chuột": "mouse", "bàn phím": "keyboard",
    "sách": "book", "đồng hồ": "clock", "chậu cây": "potted plant",
    "cây": "potted plant", "chó": "dog", "mèo": "cat", "xe": "car",
}


def _has_word(word, text):
    """Khớp 'word' theo ranh giới từ (tránh 'chai' khớp nhầm trong 'chair')."""
    return re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text) is not None


def targets_from_text(text, coco_names):
    """Tìm các lớp COCO được nhắc trong câu lệnh (tên Anh + 1 số từ Việt). -> set."""
    t = (text or "").lower()
    coco = {str(n).lower() for n in coco_names}
    found = {n for n in coco if n and _has_word(n, t)}
    for vn, en in _VN2COCO.items():
        if en in coco and _has_word(vn, t):
            found.add(en)
    return found


def ema_box(prev, cur, alpha):
    """Trung bình động: prev*(1-alpha) + cur*alpha. prev/cur = (x1,y1,x2,y2)."""
    if prev is None:
        return tuple(float(v) for v in cur)
    return tuple(alpha * c + (1.0 - alpha) * p for p, c in zip(prev, cur))


def _area(d):
    return max(0, d[2] - d[0]) * max(0, d[3] - d[1])


def pick_best(dets, min_conf, by="conf"):
    """Lọc theo conf rồi chọn 1 box. by='area' -> TO NHẤT (gần nhất, ưu tiên khi trùng
    tên/nhiều vật); by='conf' -> tin cậy cao nhất. -> (x1,y1,x2,y2,label,conf) | None."""
    good = [d for d in dets if len(d) > 5 and d[5] >= min_conf]
    if not good:
        return None
    if by == "area":
        return max(good, key=_area)
    return max(good, key=lambda d: d[5])


def draw_dets(frame_bgr, dets):
    """Vẽ hộp chữ nhật xanh + nhãn cho list (x1, y1, x2, y2, label, conf)."""
    out = frame_bgr.copy()
    for d in dets:
        if len(d) < 5:
            continue
        x1, y1, x2, y2, label = int(d[0]), int(d[1]), int(d[2]), int(d[3]), d[4]
        conf = float(d[5]) if len(d) > 5 else 0.0
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{label} {conf:.2f}", (x1, max(12, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


class LiveAnnotator:
    """YOLO nền (throttle) + lọc nhiễu -> 1 box mục tiêu ĐÃ LÀM MƯỢT (EMA).

    render() vẽ box mượt lên frame; target_box()/frame_height() cho override dùng chung
    một nguồn box ổn định (không tự gọi YOLO riêng -> tránh lệch/nhiễu)."""

    def __init__(self, frame_source, detector, fps=8.0,
                 min_conf=None, smooth_alpha=None, max_miss=None):
        self.frame_source = frame_source
        self.detector = detector
        self.interval = 1.0 / max(0.5, fps)
        self.targets = set()
        self.boxes = []                 # để render (đã mượt)
        self.smoothed = None            # (x1,y1,x2,y2) float, đã EMA (để vẽ)
        self.raw_box = None             # (x1,y1,x2,y2) box THÔ mới nhất (cho điều khiển)
        self.label = None
        self.conf = 0.0
        self.frame_h = None
        self.frame_w = None
        self._miss = 0
        names = getattr(detector, "names", []) or []
        self._coco = list(names.values()) if isinstance(names, dict) else list(names)
        # Tham số lọc nhiễu (env chỉnh được).
        self.min_conf = float(os.getenv("VLA_YOLO_MIN_CONF", "0.3")) if min_conf is None else min_conf
        self.alpha = float(os.getenv("VLA_BOX_SMOOTH", "0.4")) if smooth_alpha is None else smooth_alpha
        self.max_miss = int(os.getenv("VLA_BOX_MAX_MISS", "3")) if max_miss is None else max_miss
        # Khi nhiều vật trùng tên: 'area' = chọn TO NHẤT (gần nhất) | 'conf' = tin cậy nhất.
        self.pick_by = os.getenv("VLA_PICK", "area")
        self._stop = False

    def set_target_from_text(self, text):
        """Đặt lớp mục tiêu từ câu lệnh GUI. Reset bộ lọc. Trả set lớp tìm được."""
        self.targets = targets_from_text(text, self._coco)
        self.smoothed = None
        self.raw_box = None
        self._miss = 0
        self.boxes = []
        return self.targets

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._stop = True

    def _update_once(self, frame):
        """Một bước: detect mục tiêu -> lọc conf -> chọn box tốt nhất -> EMA. (test được)"""
        if frame is None:
            return
        self.frame_h = frame.shape[0]
        self.frame_w = frame.shape[1]
        try:
            dets = self.detector.detect(frame, wanted=self.targets)
        except Exception:
            dets = []
        best = pick_best(dets, self.min_conf, by=self.pick_by)
        if best is None:
            self._miss += 1
            if self._miss > self.max_miss:
                self.smoothed = None
                self.raw_box = None
                self.boxes = []
            return
        self._miss = 0
        self.label, self.conf = best[4], float(best[5])
        self.raw_box = tuple(int(v) for v in best[:4])     # box thô cho điều khiển (đỡ trễ EMA)
        self.smoothed = ema_box(self.smoothed, best[:4], self.alpha)
        x1, y1, x2, y2 = (int(v) for v in self.smoothed)
        self.boxes = [(x1, y1, x2, y2, self.label, self.conf)]

    def _loop(self):
        while not self._stop:
            if self.targets and self.detector is not None:
                self._update_once(self.frame_source.get_latest_frame())
            time.sleep(self.interval)

    def render(self, frame):
        """Vẽ box mượt (đã tính ở thread nền) lên frame; không có thì trả nguyên frame."""
        if self.targets and self.boxes:
            return draw_dets(frame, self.boxes)
        return frame

    # -- cho YOLO override (navloop) dùng chung box đã mượt ----------------- #
    def target_box(self):
        """Box mục tiêu đã làm mượt (x1,y1,x2,y2) hoặc None nếu đang mất dấu."""
        return self.smoothed

    def frame_height(self):
        return self.frame_h

    def frame_width(self):
        return self.frame_w

    def raw_target(self):
        """Box THÔ mới nhất (cho điều khiển) — cùng nguồn YOLO với hiển thị, đỡ chạy trùng."""
        return self.raw_box
