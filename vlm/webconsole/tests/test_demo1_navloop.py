import json
import math
import threading

import numpy as np
import pytest

from vlm.webconsole.demo1.navloop import (
    parse_vla_json, build_nav_prompt, VlaError, NavLoopAgent,
    MAX_STEP_METERS, parse_navila_action,
)


# --- parse_navila_action (NaVILA ngôn ngữ tự nhiên -> schema JSON) ---------- #
def test_navila_move_forward_cm():
    s = parse_navila_action("The next action is move forward 25 cm.")
    step = parse_vla_json(s)
    assert step.action == "move_forward"
    assert math.isclose(step.to_motion_cmd().value, 0.25)


def test_navila_turn_left_degrees():
    step = parse_vla_json(parse_navila_action("turn left 15 degrees"))
    assert step.action == "turn_left"
    assert math.isclose(step.to_motion_cmd().value, math.radians(15))


def test_navila_turn_right_degrees():
    step = parse_vla_json(parse_navila_action("The next action is turn right 30 degrees."))
    assert step.action == "turn_right" and step.to_motion_cmd().value < 0


def test_navila_stop_is_finished():
    step = parse_vla_json(parse_navila_action("stop, the task is completed."))
    assert step.finished and step.to_motion_cmd() is None


def test_navila_unparsed_falls_back_to_stop():
    # NaVILA ra câu vô nghĩa (không có số) -> mặc định DỪNG an toàn (không xoay mù).
    step = parse_vla_json(parse_navila_action("hmm not sure what to do"))
    assert step.finished


# --- parse_vla_json -------------------------------------------------------- #
def test_parse_move_forward_cm():
    s = parse_vla_json('{"action":"move_forward","value":75,"unit":"cm",'
                       '"reasoning":"đi tới","is_finished":false}')
    assert s.action == "move_forward" and not s.finished
    mc = s.to_motion_cmd()
    assert mc.kind == "move" and math.isclose(mc.value, 0.75)


def test_parse_move_backward_negative():
    s = parse_vla_json('{"action":"move_backward","value":0.3,"unit":"m"}')
    assert math.isclose(s.to_motion_cmd().value, -0.3)


def test_parse_turn_left_degrees_positive():
    s = parse_vla_json('{"action":"turn_left","value":30,"unit":"degrees"}')
    assert math.isclose(s.to_motion_cmd().value, math.radians(30))


def test_parse_turn_right_negative():
    s = parse_vla_json('{"action":"turn_right","value":45,"unit":"deg"}')
    assert math.isclose(s.to_motion_cmd().value, -math.radians(45))


def test_parse_strips_markdown_fence():
    s = parse_vla_json('```json\n{"action":"stop","is_finished":true,'
                       '"reasoning":"tới rồi"}\n```')
    assert s.finished and s.to_motion_cmd() is None


def test_parse_is_finished_without_action_is_stop():
    s = parse_vla_json('{"is_finished":true,"reasoning":"done"}')
    assert s.finished


def test_parse_obstacles_list():
    s = parse_vla_json('{"action":"stop","obstacles_detected":["chair","table"]}')
    assert s.obstacles == ["chair", "table"]


def test_parse_clamps_huge_distance():
    s = parse_vla_json('{"action":"move_forward","value":999,"unit":"m"}')
    assert abs(s.to_motion_cmd().value) <= MAX_STEP_METERS


def test_parse_no_json_raises():
    with pytest.raises(VlaError):
        parse_vla_json("xin chào, không có json ở đây")


def test_parse_invalid_action_raises():
    with pytest.raises(VlaError):
        parse_vla_json('{"action":"teleport","value":5,"unit":"m"}')


def test_parse_bad_value_raises():
    with pytest.raises(VlaError):
        parse_vla_json('{"action":"move_forward","value":"abc","unit":"m"}')


# --- build_nav_prompt ------------------------------------------------------ #
def test_prompt_has_schema_and_goal():
    p = build_nav_prompt("đi tới ghế", history=["move forward 0.50 m"],
                         obstacle_hint="chair")
    assert "đi tới ghế" in p
    assert "move_forward" in p and "is_finished" in p
    assert "move forward 0.50 m" in p and "chair" in p


# --- NavLoopAgent loop ----------------------------------------------------- #
class FakeBrain:
    """Trả lần lượt các câu JSON đã định sẵn."""
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def decide(self, frame, prompt):
        out = self.outputs[self.calls]
        self.calls += 1
        return out


class FakeMotion:
    def __init__(self, error_on=None):
        self.error_on = error_on        # message lỗi để mô phỏng (vd vật cản)
        self.ran = []

    def run(self, cmd, cancel=None):
        self.ran.append(cmd)
        yield {"type": "step", "id": "motion", "status": "running", "title": "go"}
        if self.error_on:
            yield {"type": "step", "id": "motion", "status": "error"}
            yield {"type": "error", "message": self.error_on}
            return
        yield {"type": "step", "id": "motion", "status": "done"}
        yield {"type": "answer", "text": "Done.", "state": "UNKNOWN"}


class FakeFrames:
    def get_latest_frame(self):
        return object()             # frame giả; FakeBrain không dùng tới


def _answers(events):
    return [e for e in events if e.get("type") == "answer"]


def test_loop_moves_then_finishes(tmp_path):
    brain = FakeBrain([
        '{"action":"move_forward","value":50,"unit":"cm","reasoning":"tiến"}',
        '{"action":"turn_left","value":30,"unit":"degrees","reasoning":"xoay"}',
        '{"action":"stop","is_finished":true,"reasoning":"đã tới ghế"}',
    ])
    motion = FakeMotion()
    agent = NavLoopAgent(brain, FakeFrames(), motion, log_dir=str(tmp_path), settle_s=0)
    events = list(agent.run("đi tới ghế"))
    ans = _answers(events)
    assert ans and ans[-1]["state"] == "YES"
    assert ans[-1]["text"] == "đã tới ghế"
    assert len(motion.ran) == 2                       # 2 lệnh chuyển động trước stop
    # motion's own "answer" must be swallowed -> chỉ 1 answer cuối cùng.
    assert len(ans) == 1


def test_loop_writes_jsonl_log(tmp_path):
    brain = FakeBrain([
        '{"action":"move_forward","value":50,"unit":"cm","reasoning":"tiến"}',
        '{"action":"stop","is_finished":true,"reasoning":"xong"}',
    ])
    agent = NavLoopAgent(brain, FakeFrames(), FakeMotion(), log_dir=str(tmp_path), settle_s=0)
    list(agent.run("đi"))
    logs = list(tmp_path.glob("nav_*.jsonl"))
    assert len(logs) == 1
    lines = [json.loads(l) for l in logs[0].read_text(encoding="utf-8").splitlines()]
    assert lines[0]["event"] == "start"
    assert any(r.get("action") == "move_forward" for r in lines[1:])


def test_loop_obstacle_reobserves_not_terminate(tmp_path):
    # Bước 1 đi -> vật cản (không kết thúc), bước 2 stop.
    brain = FakeBrain([
        '{"action":"move_forward","value":100,"unit":"cm","reasoning":"thử đi"}',
        '{"action":"stop","is_finished":true,"reasoning":"né xong, dừng"}',
    ])
    motion = FakeMotion(error_on="⛔ Vật cản phía trước — đã dừng an toàn.")
    agent = NavLoopAgent(brain, FakeFrames(), motion, log_dir=str(tmp_path), settle_s=0)
    events = list(agent.run("đi tới"))
    ans = _answers(events)
    assert ans[-1]["state"] == "YES"                  # vẫn chạy tới bước stop
    assert brain.calls == 2                            # đã quan sát lại sau vật cản


def test_loop_terminates_on_motion_timeout(tmp_path):
    brain = FakeBrain([
        '{"action":"move_forward","value":50,"unit":"cm","reasoning":"đi"}',
        '{"action":"stop","is_finished":true,"reasoning":"không nên tới đây"}',
    ])
    motion = FakeMotion(error_on="Quá thời gian chuyển động.")
    agent = NavLoopAgent(brain, FakeFrames(), motion, log_dir=str(tmp_path), settle_s=0)
    events = list(agent.run("đi"))
    assert any(e.get("type") == "error" for e in events)
    assert brain.calls == 1                            # dừng ngay, không quan sát lại


def test_loop_respects_cancel(tmp_path):
    brain = FakeBrain(['{"action":"move_forward","value":50,"unit":"cm"}'] * 5)
    cancel = threading.Event()
    cancel.set()
    agent = NavLoopAgent(brain, FakeFrames(), FakeMotion(), log_dir=str(tmp_path), settle_s=0)
    events = list(agent.run("đi", cancel=cancel))
    assert any("dừng theo yêu cầu" in e.get("message", "") for e in events)
    assert brain.calls == 0


def test_loop_max_steps_cap(tmp_path):
    # Luôn move_forward, không bao giờ stop -> phải dừng ở max_steps.
    brain = FakeBrain(['{"action":"move_forward","value":50,"unit":"cm"}'] * 10)
    agent = NavLoopAgent(brain, FakeFrames(), FakeMotion(), max_steps=3,
                         log_dir=str(tmp_path), settle_s=0)
    events = list(agent.run("đi"))
    assert brain.calls == 3
    assert _answers(events)[-1]["state"] == "UNKNOWN"


class _AnnHint:
    raw_box = (1, 1, 2, 2)
    label = "bottle"


def test_obstacle_hint_from_annotator(tmp_path):
    # Hint lấy từ annotator (1 nguồn YOLO), KHÔNG detect riêng.
    agent = NavLoopAgent(FakeBrain([]), FakeFrames(), FakeMotion(),
                         annotator=_AnnHint(), log_dir=str(tmp_path), settle_s=0)
    assert agent._obstacle_hint(None) == "bottle"
    # không có annotator -> None
    agent2 = NavLoopAgent(FakeBrain([]), FakeFrames(), FakeMotion(),
                          log_dir=str(tmp_path), settle_s=0)
    assert agent2._obstacle_hint(None) is None


def test_loop_bad_json_terminates(tmp_path):
    brain = FakeBrain(["không phải json"])
    agent = NavLoopAgent(brain, FakeFrames(), FakeMotion(), log_dir=str(tmp_path), settle_s=0)
    events = list(agent.run("đi"))
    assert any("JSON VLM hỏng" in e.get("message", "") for e in events)


# --- YOLO override điểm dừng ----------------------------------------------- #
class FakeYolo:
    """detect(wanted) trả box 'bottle' với y2 lấy lần lượt từ y2_list (mô phỏng
    robot tiến lại gần -> đáy hộp tụt dần về mép dưới ảnh). Call obstacle-hint
    (wanted=None) trả box cố định, KHÔNG tiêu thụ y2_list."""
    names = ["bottle", "chair"]

    def __init__(self, y2_list):
        self.y2 = list(y2_list)
        self.n = 0

    def detect(self, frame, wanted=None):
        if not wanted:
            return [(100, 100, 200, 250, "bottle", 0.9)]
        y2 = self.y2[min(self.n, len(self.y2) - 1)]
        self.n += 1
        return [(100, 100, 200, y2, "bottle", 0.9)]


class ShapedFrames:
    def get_latest_frame(self):
        return np.zeros((720, 1280, 3), dtype=np.uint8)   # khớp camera thật 1280x720


def test_no_override_when_target_at_bottom(tmp_path):
    brain = FakeBrain(['{"action":"stop","is_finished":true,"reasoning":"tới"}'])
    motion = FakeMotion()
    yolo = FakeYolo([460])                        # gap 20 <= 50 -> chấp nhận dừng
    agent = NavLoopAgent(brain, ShapedFrames(), motion, detector=yolo,
                         log_dir=str(tmp_path), settle_s=0, stop_bottom_px=50)
    events = list(agent.run("go to the bottle"))
    assert not any("YOLO override" in e.get("text", "") for e in events if e.get("type") == "token")
    assert len(motion.ran) == 0
    assert _answers(events)[-1]["state"] == "YES"


# --- servo hình học (ưu tiên: stop > center > forward) --------------------- #
def _servo_agent(tmp_path):
    return NavLoopAgent(FakeBrain([]), ShapedFrames(), FakeMotion(),
                        log_dir=str(tmp_path), settle_s=0, stop_bottom_px=20)


class SeqAnn:
    """annotator giả: trả box lần lượt theo list (mô phỏng robot tiến lại gần)."""
    label = "bottle"

    def __init__(self, boxes):
        self.boxes = boxes
        self.i = 0

    def target_box(self):
        b = self.boxes[min(self.i, len(self.boxes) - 1)]
        self.i += 1
        return b

    def frame_height(self):
        return 720

    def frame_width(self):
        return 1280


class ServoMotion:
    """Ghi lại các (vx, wz) được bơm + số lần stop (cho visual servo liên tục)."""
    def __init__(self):
        self.pubs = []
        self.stops = 0

    def _publish(self, vx, wz):
        self.pubs.append((round(vx, 3), round(wz, 3)))

    def _stop(self):
        self.stops += 1

    def front_distance(self):
        return None


def test_visual_servo_turn_then_forward_then_stop(tmp_path):
    # run() tiêu thụ box[0]; servo: lệch -> XOAY (vx=0); giữa -> TIẾN (wz=0); đáy chạm -> DỪNG.
    ann = SeqAnn([(1120, 100, 1220, 300),    # run() đọc trước
                  (1120, 100, 1220, 300),    # servo: lệch ~37° (>30) -> CHỈ xoay (vx=0)
                  (600, 100, 680, 300),      # servo: giữa (lệch 0) -> tiến thẳng (wz=0)
                  (600, 100, 680, 710)])     # servo: đáy 10px <= 20 -> dừng
    motion = ServoMotion()
    ag = NavLoopAgent(FakeBrain([]), ShapedFrames(), motion, annotator=ann,
                      log_dir=str(tmp_path), settle_s=0, stop_bottom_px=20, control="servo")
    ag.servo_hz = 1000.0; ag.servo_pulse_s = 0; ag.servo_refresh_s = 0; ag.stop_settle_s = 0; ag.final_push_m = 0                       # chạy nhanh cho test
    events = list(ag.run("go to the bottle"))
    assert _answers(events)[-1]["state"] == "YES"
    assert any(vx == 0.0 and wz != 0.0 for vx, wz in motion.pubs)   # P1 xoay căn giữa
    assert any(vx > 0.0 and wz == 0.0 for vx, wz in motion.pubs)    # P2 tiến thẳng
    assert motion.stops >= 1


def test_visual_servo_stops_immediately_at_bottom(tmp_path):
    # Đáy đã sát mép ngay từ đầu -> servo DỪNG, không bơm vx tiến.
    ann = SeqAnn([(600, 100, 680, 712), (600, 100, 680, 712)])
    motion = ServoMotion()
    ag = NavLoopAgent(FakeBrain([]), ShapedFrames(), motion, annotator=ann,
                      log_dir=str(tmp_path), settle_s=0, stop_bottom_px=20, control="servo")
    ag.servo_hz = 1000.0; ag.servo_pulse_s = 0; ag.servo_refresh_s = 0; ag.stop_settle_s = 0; ag.final_push_m = 0
    events = list(ag.run("go to the bottle"))
    assert _answers(events)[-1]["state"] == "YES"
    assert not any(vx > 0.0 for vx, wz in motion.pubs)     # không tiến (đã sát đáy)
    assert motion.stops >= 1


def test_state_text_injects_geometry(tmp_path):
    ag = _servo_agent(tmp_path)
    s = ag._state_text((1000, 100, 1100, 300), 1280, 720, 1.5)
    assert "lệch" in s and "phải" in s
    assert "đáy cách mép dưới 420px" in s and "1.50m" in s


def test_vlm_turn_capped_to_max(tmp_path):
    # VLM xuất turn 90° -> code chặn xuống ≤ max_turn_deg (10°).
    brain = FakeBrain(['{"action":"turn_left","value":90,"unit":"degrees"}'])
    motion = FakeMotion()
    ag = NavLoopAgent(brain, FakeFrames(), motion, max_steps=1,
                      log_dir=str(tmp_path), settle_s=0, control="vlm")
    list(ag.run("scan"))
    assert len(motion.ran) == 1
    assert abs(motion.ran[0].value) <= math.radians(10) + 1e-6


def test_search_accumulator_stops_after_full_scan(tmp_path):
    # Quét tìm xoay mãi không thấy -> đủ ngưỡng góc thì DỪNG (chống xoay vô tận).
    brain = FakeBrain(['{"action":"turn_left","value":10,"unit":"degrees"}'] * 10)
    motion = FakeMotion()
    ag = NavLoopAgent(brain, FakeFrames(), motion, log_dir=str(tmp_path),
                      settle_s=0, control="vlm")
    ag.search_max_deg = 25                      # 10+10+10=30 >= 25 -> dừng ở bước 3
    events = list(ag.run("go to the bottle"))
    assert any("Không tìm thấy mục tiêu" in e.get("message", "") for e in events)
    assert len(motion.ran) <= 3


def test_vlm_trusts_stop_pure_vlm(tmp_path):
    # THUẦN VLM: VLM báo stop -> tin ngay (không còn code override).
    brain = FakeBrain(['{"action":"stop","is_finished":true,"reasoning":"tới"}'])
    agent = NavLoopAgent(brain, FakeFrames(), FakeMotion(), log_dir=str(tmp_path),
                         settle_s=0, control="vlm")
    events = list(agent.run("go to the bottle"))
    assert _answers(events)[-1]["state"] == "YES"
