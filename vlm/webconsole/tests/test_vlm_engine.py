import numpy as np
from vlm.webconsole.vlm_engine import (
    parse_boxes,
    draw_boxes,
    encode_frame_jpeg,
    build_messages,
)


def test_parse_boxes_single():
    # ymin=100, xmin=200, ymax=300, xmax=400 trên thang 1000
    text = "The phone is at [100, 200, 300, 400]."
    boxes = parse_boxes(text, width=1000, height=1000)
    assert boxes == [(200, 100, 400, 300)]  # (x1, y1, x2, y2)


def test_parse_boxes_scales_to_image():
    text = "[0, 0, 500, 1000]"
    boxes = parse_boxes(text, width=640, height=480)
    # x1=0, y1=0, x2 = 1000/1000*640 = 640, y2 = 500/1000*480 = 240
    assert boxes == [(0, 0, 640, 240)]


def test_parse_boxes_none():
    assert parse_boxes("no objects detected", 640, 480) == []


def test_parse_boxes_clamps_overflow():
    text = "[0, 0, 1200, 1200]"  # vượt 1000
    boxes = parse_boxes(text, width=100, height=100)
    assert boxes == [(0, 0, 100, 100)]


def test_draw_boxes_returns_same_shape_and_copy():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = draw_boxes(frame, [(10, 10, 100, 100)], "phone")
    assert out.shape == frame.shape
    assert out is not frame  # không sửa frame gốc
    assert out.sum() > 0      # đã vẽ gì đó


def test_encode_frame_jpeg_magic_bytes():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    data = encode_frame_jpeg(frame)
    assert isinstance(data, bytes)
    assert data[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_build_messages_shape():
    msgs = build_messages("PIL_PLACEHOLDER", "find the phone")
    assert msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["image"] == "PIL_PLACEHOLDER"
    assert content[1] == {"type": "text", "text": "find the phone"}


import pytest
from vlm.webconsole.vlm_engine import VLMEngine


def test_engine_starts_unloaded():
    eng = VLMEngine()
    assert eng.loaded is False


def test_stream_infer_requires_load():
    eng = VLMEngine()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        list(eng.stream_infer(frame, "describe"))
