"""Perception: YOLO detect + box (chính xác) + Qwen trả lời tự nhiên.

Theo cách AnhDV: YOLO (ultralytics) lo phát hiện + khoanh vùng vật thể (80 lớp COCO,
toạ độ frame-space), Qwen2.5-VL lo trả lời câu hỏi (được mớm kết quả YOLO để chính xác
hơn). Nếu không có YOLO -> fallback dùng Qwen grounding như trước.
"""
import re
import base64
from dataclasses import dataclass

import cv2

from ..vlm_engine import encode_frame_jpeg

_VERDICT = re.compile(r"\b(YES|NO|ON|OFF|UNKNOWN)\b")
# Qwen2.5-VL grounding (chỉ dùng khi KHÔNG có YOLO): {"bbox_2d":[x1,y1,x2,y2],"label":..}
_BBOX = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    r'\s*,\s*"label"\s*:\s*"([^"]*)"'
)
_JSON_NOISE = re.compile(r"```.*?```|\[\s*\{.*?\}\s*\]", re.DOTALL)


@dataclass
class Result:
    state: str
    annotated_jpeg_b64: str
    answer: str


def parse_verdict(text):
    m = _VERDICT.search(text or "")
    if m:
        return m.group(1)
    t = (text or "").strip().lower()
    if t.startswith("yes"):
        return "YES"
    if t.startswith("no"):
        return "NO"
    return "UNKNOWN"


def parse_detections(text):
    """[(x1, y1, x2, y2, label)] từ JSON grounding của Qwen (fallback)."""
    out = []
    for m in _BBOX.finditer(text or ""):
        x1, y1, x2, y2 = (int(m.group(i)) for i in range(1, 5))
        out.append((x1, y1, x2, y2, m.group(5)))
    return out


def clean_answer(text):
    return _JSON_NOISE.sub("", text or "").strip()


def _draw(frame_bgr, dets):
    out = frame_bgr.copy()
    for (x1, y1, x2, y2, label) in dets:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, label, (x1, max(14, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    return out


def answer_prompt(question, detected_note=""):
    return (
        "Look at the image and answer like a helpful assistant." + detected_note +
        " First describe in one sentence what you see that is relevant, then add a "
        f"final sentence that starts with YES or NO to answer: \"{question}\". "
        "Do not output any coordinates or JSON."
    )


def detect_prompt(question):
    return (
        f"Detect the objects relevant to this question: \"{question}\".\n"
        "Output ONLY a JSON list, nothing else:\n"
        '[{"bbox_2d": [x1, y1, x2, y2], "label": "object name"}]'
    )


class Perception:
    def __init__(self, engine, detector=None):
        self.engine = engine
        self.detector = detector
        self._dets = []   # YOLO dets (frame-space) từ lần observe gần nhất

    def _wanted(self, text):
        """Tập lớp COCO được nhắc trong câu hỏi; None -> giữ tất cả."""
        if not self.detector:
            return None
        t = (text or "").lower()
        w = {n for n in self.detector.names if n in t}
        return w or None

    def observe(self, frame_bgr, target_object, question):
        """Lần gọi Qwen: stream câu trả lời tự nhiên (mớm kết quả YOLO nếu có)."""
        self._dets = []
        note = ""
        if self.detector is not None:
            self._dets = self.detector.detect(
                frame_bgr, self._wanted(question or target_object))
            labels = sorted({d[4] for d in self._dets})
            note = (f" A YOLO object detector found these objects in the image: "
                    f"{', '.join(labels)}." if labels else
                    " A YOLO object detector did not find the relevant objects.")
        return self.engine.stream_infer(frame_bgr, answer_prompt(question, note))

    def _detect_qwen(self, frame_bgr, question):
        """Fallback: lấy box bằng Qwen grounding (cần scale theo ảnh model)."""
        text = "".join(self.engine.stream_infer(frame_bgr, detect_prompt(question)))
        dets = parse_detections(text)
        msize = getattr(self.engine, "last_image_size", None)
        return dets, msize

    def _render(self, frame_bgr, dets, msize):
        if not dets:
            return None
        h, w = frame_bgr.shape[:2]
        mw, mh = msize if msize else (w, h)
        scaled = []
        for d in dets:
            x1, y1, x2, y2, label = d[0], d[1], d[2], d[3], d[4]
            scaled.append((
                max(0, min(int(x1 * w / mw), w)),
                max(0, min(int(y1 * h / mh), h)),
                max(0, min(int(x2 * w / mw), w)),
                max(0, min(int(y2 * h / mh), h)),
                label,
            ))
        return base64.b64encode(encode_frame_jpeg(_draw(frame_bgr, scaled))).decode("ascii")

    def finalize(self, answer_text, frame_bgr, target_object, question=None):
        answer = clean_answer(answer_text)
        state = parse_verdict(answer)
        if self.detector is not None:
            # YOLO: toạ độ đã ở frame-space (msize=None), nhãn kèm conf
            dets = [(x1, y1, x2, y2, f"{label} {conf:.2f}")
                    for (x1, y1, x2, y2, label, conf) in self._dets]
            annotated = self._render(frame_bgr, dets, None)
        elif self.engine is not None and question:
            dets, msize = self._detect_qwen(frame_bgr, question)
            annotated = self._render(frame_bgr, dets, msize)
        else:
            annotated = self._render(frame_bgr, parse_detections(answer_text), None)
        return Result(state=state, annotated_jpeg_b64=annotated, answer=answer)
