"""Perception: VLM trả lời tự nhiên (mô tả + kết luận) + vẽ bbox (Qwen2.5-VL).

Dùng 2 lần suy luận tách biệt để câu trả lời được tự nhiên (không bị "grounding mode"
làm cụt): 1) trả lời câu hỏi, 2) detect bbox.
"""
import re
import base64
from dataclasses import dataclass

import cv2

from ..vlm_engine import encode_frame_jpeg

_VERDICT = re.compile(r"\b(YES|NO|ON|OFF|UNKNOWN)\b")
# Qwen2.5-VL grounding: {"bbox_2d": [x1, y1, x2, y2], "label": "..."}  (pixel tuyệt đối)
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
    """Suy verdict (YES/NO/ON/OFF/UNKNOWN) từ câu trả lời tự nhiên — best-effort."""
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
    """[(x1, y1, x2, y2, label)] trong KHÔNG GIAN pixel của model."""
    out = []
    for m in _BBOX.finditer(text or ""):
        x1, y1, x2, y2 = (int(m.group(i)) for i in range(1, 5))
        out.append((x1, y1, x2, y2, m.group(5)))
    return out


def clean_answer(text):
    """Bỏ phần JSON, giữ câu trả lời ngôn ngữ tự nhiên."""
    return _JSON_NOISE.sub("", text or "").strip()


def _draw(frame_bgr, dets):
    out = frame_bgr.copy()
    for (x1, y1, x2, y2, label) in dets:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, label, (x1, max(14, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    return out


def answer_prompt(question):
    return (
        "Look at the image and answer like a helpful assistant. First describe in "
        "one sentence what you see that is relevant, then add a final sentence that "
        f"starts with YES or NO to directly answer: \"{question}\". "
        "Do not output any coordinates or JSON."
    )


def detect_prompt(question):
    return (
        f"Detect the objects relevant to this question: \"{question}\".\n"
        "Output ONLY a JSON list, nothing else:\n"
        '[{"bbox_2d": [x1, y1, x2, y2], "label": "object name"}]'
    )


class Perception:
    def __init__(self, engine):
        self.engine = engine

    def observe(self, frame_bgr, target_object, question):
        """Lần gọi 1: stream câu trả lời tự nhiên."""
        return self.engine.stream_infer(frame_bgr, answer_prompt(question))

    def _detect(self, frame_bgr, question):
        """Lần gọi 2: lấy bbox (model coord space)."""
        text = "".join(self.engine.stream_infer(frame_bgr, detect_prompt(question)))
        return parse_detections(text), getattr(self.engine, "last_image_size", None)

    def _render(self, frame_bgr, dets, msize):
        if not dets:
            return None
        h, w = frame_bgr.shape[:2]
        mw, mh = msize if msize else (w, h)
        scaled = []
        for (x1, y1, x2, y2, label) in dets:
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
        if self.engine is not None and question:
            dets, msize = self._detect(frame_bgr, question)
        else:  # fallback (test / không có engine): lấy box ngay trong answer_text
            dets, msize = parse_detections(answer_text), None
        annotated = self._render(frame_bgr, dets, msize)
        return Result(state=state, annotated_jpeg_b64=annotated, answer=answer)
