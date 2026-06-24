import numpy as np
from vlm.webconsole.demo1.rooms import Room
from vlm.webconsole.demo1.planner import Plan, PlanError
from vlm.webconsole.demo1.perception import Result
from vlm.webconsole.demo1.agent import Agent

ROOMS = {"kitchen": Room("kitchen", 1.0, 2.0, 0.0, ["microwave"])}


class FakePlanner:
    def plan(self, command, rooms):
        return Plan("kitchen", "microwave", "lò bật hay tắt?", "microwave ở kitchen")


class FakeNavigator:
    def go_to(self, room):
        yield {"kind": "feedback", "distance_remaining": 1.0}
        yield {"kind": "done", "success": True}


class FakeFrame:
    def get_latest_frame(self):
        return np.zeros((48, 64, 3), dtype=np.uint8)


class FakePerception:
    def observe(self, frame, target, question):
        yield "microwave [100,100,300,300]. "
        yield "It is ON."

    def finalize(self, text, frame, target, question=None):
        return Result("ON", "ZmFrZQ==", text)


def _collect(agent, command):
    return list(agent.run(command))


def test_agent_event_sequence():
    agent = Agent(FakePlanner(), FakeNavigator(), FakeFrame(), FakePerception(), ROOMS)
    events = _collect(agent, "đồ ăn nóng chưa?")
    steps = [(e.get("id"), e.get("status")) for e in events if e["type"] == "step"]
    assert ("plan", "running") in steps
    assert ("plan", "done") in steps
    assert ("nav", "running") in steps
    assert ("nav", "done") in steps
    assert ("perceive", "running") in steps
    assert ("perceive", "done") in steps
    answers = [e for e in events if e["type"] == "answer"]
    assert len(answers) == 1
    assert answers[0]["state"] == "ON"
    assert any(e["type"] == "image" for e in events)


def test_agent_nav_error_stops():
    class BadNav:
        def go_to(self, room):
            yield {"kind": "error", "message": "robot kẹt"}

    agent = Agent(FakePlanner(), BadNav(), FakeFrame(), FakePerception(), ROOMS)
    events = _collect(agent, "x")
    assert any(e["type"] == "error" for e in events)
    assert not any(e["type"] == "answer" for e in events)


def test_agent_plan_error_stops():
    class BadPlanner:
        def plan(self, command, rooms):
            raise PlanError("không xác định được phòng")

    agent = Agent(BadPlanner(), FakeNavigator(), FakeFrame(), FakePerception(), ROOMS)
    events = _collect(agent, "x")
    assert any(e["type"] == "error" for e in events)
    assert not any(e["type"] == "answer" for e in events)


def test_agent_no_frame_stops():
    class NoFrame:
        def get_latest_frame(self):
            return None

    agent = Agent(FakePlanner(), FakeNavigator(), NoFrame(), FakePerception(), ROOMS)
    events = _collect(agent, "x")
    assert any(e["type"] == "error" for e in events)
    assert not any(e["type"] == "answer" for e in events)
