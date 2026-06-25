import base64
import numpy as np
from vlm.webconsole.demo1.perception import (
    parse_verdict, parse_detections, clean_answer, Perception, Result,
)


class FakeEngine:
    """stream_infer trả answer (hoặc detect JSON tùy prompt); có last_image_size."""
    def __init__(self, answer="", det="", size=(640, 480)):
        self.answer = answer
        self.det = det
        self.last_image_size = size

    def stream_infer(self, frame, prompt):
        yield self.det if "bbox_2d" in prompt else self.answer


class FakeDetector:
    names = ["bottle", "microwave", "cup", "person"]

    def __init__(self, dets):
        self._dets = dets

    def detect(self, frame, wanted=None):
        if wanted:
            return [d for d in self._dets if d[4] in wanted]
        return list(self._dets)


def test_parse_verdict():
    assert parse_verdict("ON, the display is lit.") == "ON"
    assert parse_verdict("Yes, the bottle is on the microwave.") == "YES"
    assert parse_verdict("No, there is no bottle.") == "NO"
    assert parse_verdict("I cannot tell.") == "UNKNOWN"


def test_parse_detections_qwen_format():
    text = '[{"bbox_2d": [730, 476, 1170, 728], "label": "microwave"}]'
    assert parse_detections(text) == [(730, 476, 1170, 728, "microwave")]


def test_clean_answer_strips_json():
    text = 'Yes, on the microwave.\n[{"bbox_2d": [1, 2, 3, 4], "label": "bottle"}]'
    assert clean_answer(text) == "Yes, on the microwave."


def test_wanted_classes_from_question():
    p = Perception(engine=None, detector=FakeDetector([]))
    assert p._wanted("is the bottle on the microwave?") == {"bottle", "microwave"}
    assert p._wanted("something unrelated") is None


def test_observe_feeds_yolo_labels_into_prompt():
    eng = FakeEngine(answer="YES.")
    det = FakeDetector([(10, 10, 50, 80, "bottle", 0.9),
                        (60, 60, 200, 200, "microwave", 0.8)])
    p = Perception(eng, detector=det)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    list(p.observe(frame, "microwave", "is the bottle on the microwave?"))
    # đã lưu dets để finalize vẽ
    assert len(p._dets) == 2


def test_finalize_with_yolo_draws_framespace_box():
    eng = FakeEngine(answer="YES, the bottle is on the microwave.")
    det = FakeDetector([(100, 100, 300, 300, "bottle", 0.91)])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    p = Perception(eng, detector=det)
    list(p.observe(frame, "microwave", "is the bottle on the microwave?"))
    res = p.finalize(eng.answer, frame, "microwave",
                     question="is the bottle on the microwave?")
    assert isinstance(res, Result)
    assert res.state == "YES"
    assert res.annotated_jpeg_b64 is not None
    assert base64.b64decode(res.annotated_jpeg_b64)[:2] == b"\xff\xd8"


def test_finalize_qwen_fallback_no_detector():
    # không có detector -> dùng Qwen grounding (call 2), scale theo last_image_size
    eng = FakeEngine(answer="NO.",
                     det='[{"bbox_2d": [640, 480, 1280, 960], "label": "x"}]',
                     size=(1280, 960))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = Perception(eng, detector=None).finalize("NO.", frame, "x", question="q?")
    assert res.annotated_jpeg_b64 is not None
    assert res.state == "NO"


def test_finalize_no_detection_no_image():
    eng = FakeEngine(answer="NO, nothing here.")
    det = FakeDetector([])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    p = Perception(eng, detector=det)
    list(p.observe(frame, "microwave", "is the bottle there?"))
    res = p.finalize("NO, nothing here.", frame, "microwave",
                     question="is the bottle there?")
    assert res.annotated_jpeg_b64 is None
    assert res.state == "NO"
