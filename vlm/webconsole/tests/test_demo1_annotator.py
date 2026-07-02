import numpy as np
from vlm.webconsole.demo1.annotator import (
    targets_from_text, draw_dets, LiveAnnotator, ema_box, pick_best,
)

COCO = ["person", "bottle", "chair", "dining table", "cup", "tv", "backpack"]


def test_target_english_bottle():
    assert targets_from_text("move to the big blue bottle and stop", COCO) == {"bottle"}


def test_target_vietnamese_chair():
    assert targets_from_text("đi tới cái ghế rồi dừng", COCO) == {"chair"}


def test_target_vietnamese_binh_nuoc_to_bottle():
    assert targets_from_text("đi tới bình nước", COCO) == {"bottle"}


def test_target_multiple():
    got = targets_from_text("find the person sitting on the chair", COCO)
    assert got == {"person", "chair"}


def test_target_none_when_no_match():
    assert targets_from_text("đi loanh quanh thôi", COCO) == set()


def test_target_ignores_unknown_class():
    # 'rocket' không thuộc COCO -> không trả gì.
    assert targets_from_text("go to the rocket", COCO) == set()


def test_draw_dets_keeps_shape():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    out = draw_dets(frame, [(5, 5, 30, 30, "bottle", 0.91)])
    assert out.shape == frame.shape


def test_draw_dets_empty_ok():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    assert draw_dets(frame, []).shape == frame.shape


class FakeDetector:
    names = {0: "person", 1: "bottle", 2: "chair"}

    def detect(self, frame, wanted=None):
        return [(1, 1, 10, 10, "bottle", 0.9)] if (wanted and "bottle" in wanted) else []


class FakeFrames:
    def get_latest_frame(self):
        return np.zeros((48, 64, 3), dtype=np.uint8)


# --- lọc nhiễu + EMA -------------------------------------------------------- #
def test_ema_box_first_is_current():
    assert ema_box(None, (0, 0, 10, 10), 0.4) == (0, 0, 10, 10)


def test_ema_box_blends():
    assert ema_box((0, 0, 0, 0), (10, 10, 10, 10), 0.5) == (5, 5, 5, 5)


def test_pick_best_filters_conf_and_picks_max():
    dets = [(0, 0, 5, 5, "bottle", 0.3), (1, 1, 6, 6, "bottle", 0.9),
            (2, 2, 7, 7, "bottle", 0.6)]
    assert pick_best(dets, 0.45)[5] == 0.9
    assert pick_best(dets, 0.95) is None       # tất cả dưới ngưỡng -> None


class JumpyDetector:
    """Trả box nhảy lung tung quanh y2≈300 để test EMA làm mượt."""
    names = ["bottle"]

    def __init__(self, y2_seq):
        self.y2 = list(y2_seq)
        self.n = 0

    def detect(self, frame, wanted=None):
        y2 = self.y2[min(self.n, len(self.y2) - 1)]
        self.n += 1
        return [(100, 100, 200, y2, "bottle", 0.9)]


class Frames720:
    def get_latest_frame(self):
        return np.zeros((720, 1280, 3), dtype=np.uint8)


def test_update_smooths_jumpy_y2():
    ann = LiveAnnotator(Frames720(), JumpyDetector([300, 500, 300, 500]),
                        smooth_alpha=0.4, min_conf=0.1)
    ann.set_target_from_text("go to the bottle")
    f = np.zeros((720, 1280, 3), dtype=np.uint8)
    ys = []
    for _ in range(4):
        ann._update_once(f)
        ys.append(ann.target_box()[3])
    # box thô nhảy 300<->500 nhưng giá trị mượt phải nằm GIỮA, biên độ nhỏ hơn nhiều.
    assert all(300 <= y <= 500 for y in ys)
    assert max(ys[1:]) - min(ys[1:]) < 200       # dao động bị nén lại
    assert ann.frame_height() == 720


def test_update_clears_after_misses():
    class NoneDet:
        names = ["bottle"]
        def detect(self, frame, wanted=None):
            return []
    ann = LiveAnnotator(Frames720(), NoneDet(), max_miss=2)
    ann.set_target_from_text("go to the bottle")
    ann.smoothed = (1, 1, 2, 2)
    f = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(3):
        ann._update_once(f)
    assert ann.target_box() is None              # mất dấu > max_miss -> xoá box


def test_annotator_render_draws_only_when_target_set():
    ann = LiveAnnotator(FakeFrames(), FakeDetector())
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    # chưa set target -> trả nguyên frame (không box)
    assert np.array_equal(ann.render(frame), frame)
    # set target = bottle, nạp box thủ công (bỏ qua thread nền) -> có vẽ
    ann.set_target_from_text("đi tới bình nước")
    assert ann.targets == {"bottle"}
    ann.boxes = ann.detector.detect(frame, wanted=ann.targets)
    assert not np.array_equal(ann.render(frame), frame)
