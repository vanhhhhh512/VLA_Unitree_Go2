import math
from vlm.webconsole.demo1.motion import parse_motion, MotionCmd


def test_move_forward_cm():
    c = parse_motion("move forward 75 cm")
    assert c.kind == "move" and math.isclose(c.value, 0.75)


def test_move_forward_m():
    c = parse_motion("go forward 0.5 m")
    assert c.kind == "move" and math.isclose(c.value, 0.5)


def test_move_backward_negative():
    c = parse_motion("move backward 30 cm")
    assert c.kind == "move" and math.isclose(c.value, -0.30)


def test_turn_left_deg_positive():
    c = parse_motion("turn left 90 deg")
    assert c.kind == "turn" and math.isclose(c.value, math.radians(90))


def test_turn_right_deg_negative():
    c = parse_motion("turn right 45 degrees")
    assert c.kind == "turn" and math.isclose(c.value, -math.radians(45))


def test_turn_radians():
    c = parse_motion("rotate left 1.57 rad")
    assert c.kind == "turn" and math.isclose(c.value, 1.57)


def test_non_motion_returns_none():
    assert parse_motion("is the bottle on the microwave?") is None
    assert parse_motion("go to the kitchen") is None
    assert parse_motion("describe the scene") is None
