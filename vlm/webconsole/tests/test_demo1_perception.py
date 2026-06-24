import base64
import numpy as np
from vlm.webconsole.demo1.perception import (
    parse_verdict, parse_detections, clean_answer, Perception, Result,
)


class FakeEngine:
    """stream_infer trả answer hoặc detect JSON tùy prompt; có last_image_size."""
    def __init__(self, answer="", det="", size=(640, 480)):
        self.answer = answer
        self.det = det
        self.last_image_size = size

    def stream_infer(self, frame, prompt):
        yield self.det if "bbox_2d" in prompt else self.answer


def test_parse_verdict():
    assert parse_verdict("ON, the microwave display is lit.") == "ON"
    assert parse_verdict("OFF. No lights are visible.") == "OFF"
    assert parse_verdict("Yes, the bottle is on the microwave.") == "YES"
    assert parse_verdict("No, there is no bottle.") == "NO"
    assert parse_verdict("I cannot tell.") == "UNKNOWN"


def test_parse_detections_qwen_format():
    text = ('[{"bbox_2d": [730, 476, 1170, 728], "label": "microwave"}, '
            '{"bbox_2d": [905, 392, 964, 510], "label": "bottle"}]')
    dets = parse_detections(text)
    assert dets[0] == (730, 476, 1170, 728, "microwave")
    assert dets[1] == (905, 392, 964, 510, "bottle")


def test_clean_answer_strips_json():
    text = ('Yes, the bottle is on the microwave.\n'
            '[{"bbox_2d": [1, 2, 3, 4], "label": "bottle"}]')
    assert clean_answer(text) == "Yes, the bottle is on the microwave."


def test_finalize_two_calls_answer_and_box():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    eng = FakeEngine(
        answer="I see a bottle on top of the microwave. YES, it is on the microwave.",
        det='[{"bbox_2d": [100, 100, 300, 300], "label": "bottle"}]',
        size=(640, 480),
    )
    res = Perception(eng).finalize(eng.answer, frame, "microwave",
                                   question="is the bottle on the microwave?")
    assert isinstance(res, Result)
    assert res.state == "YES"
    assert "bottle" in res.answer.lower()
    assert res.annotated_jpeg_b64 is not None
    assert base64.b64decode(res.annotated_jpeg_b64)[:2] == b"\xff\xd8"


def test_finalize_scales_with_model_size():
    # last_image_size (1280,960) -> box scale về frame 640x480
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    eng = FakeEngine(answer="NO.",
                     det='[{"bbox_2d": [640, 480, 1280, 960], "label": "x"}]',
                     size=(1280, 960))
    res = Perception(eng).finalize("NO.", frame, "x", question="q?")
    assert res.annotated_jpeg_b64 is not None


def test_finalize_no_box_no_image():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    eng = FakeEngine(answer="NO, nothing here.", det="[]", size=(640, 480))
    res = Perception(eng).finalize("NO, nothing here.", frame, "microwave",
                                   question="q?")
    assert res.annotated_jpeg_b64 is None
    assert res.state == "NO"


def test_finalize_fallback_without_engine():
    # engine=None -> lấy box ngay trong answer_text (coords ở frame space)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    text = ('ON.\n[{"bbox_2d": [100, 100, 300, 300], "label": "microwave"}]')
    res = Perception(engine=None).finalize(text, frame, "microwave")
    assert res.state == "ON"
    assert res.annotated_jpeg_b64 is not None
