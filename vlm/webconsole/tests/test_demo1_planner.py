import pytest
from vlm.webconsole.demo1.rooms import Room
from vlm.webconsole.demo1.planner import parse_plan, PlanError, Plan

ROOMS = {
    "kitchen": Room("kitchen", 1, 2, 0, ["microwave", "fridge"]),
    "living_room": Room("living_room", 0, 0, 0, ["sofa", "tv"]),
}


def test_parse_plan_valid():
    text = '''Đây là kế hoạch:
    {"room": "kitchen", "target_object": "microwave",
     "observation_question": "lò đang bật hay tắt?",
     "reasoning": "microwave ở kitchen"}'''
    plan = parse_plan(text, ROOMS)
    assert isinstance(plan, Plan)
    assert plan.room == "kitchen"
    assert plan.target_object == "microwave"
    assert "bật" in plan.observation_question


def test_parse_plan_unknown_room():
    text = '{"room": "garage", "target_object": "car", "observation_question": "?", "reasoning": "x"}'
    with pytest.raises(PlanError):
        parse_plan(text, ROOMS)


def test_parse_plan_no_json():
    with pytest.raises(PlanError):
        parse_plan("không có json ở đây", ROOMS)
