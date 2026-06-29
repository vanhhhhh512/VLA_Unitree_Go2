import numpy as np
from fastapi.testclient import TestClient
from vlm.webconsole.demo1.rooms import Room
from vlm.webconsole.demo1.planner import Plan
from vlm.webconsole.demo1.perception import Result
from vlm.webconsole.demo1.agent import Agent
from vlm.webconsole.demo1.agent_server import create_agent_app

ROOMS = {"kitchen": Room("kitchen", 1.0, 2.0, 0.0, ["microwave"])}


class FakePlanner:
    def plan(self, c, r):
        return Plan("kitchen", "microwave", "bật hay tắt?", "vì microwave ở kitchen")


class FakeNav:
    def go_to(self, room):
        yield {"kind": "done", "success": True}


class FakeFrame:
    is_mock = True
    is_connected = False

    def get_latest_frame(self):
        return np.zeros((48, 64, 3), dtype=np.uint8)


class FakePerception:
    def observe(self, f, t, q):
        yield "microwave [10,10,30,30]. It is ON."

    def finalize(self, text, f, t, question=None):
        return Result("ON", None, text)


def make_client():
    agent = Agent(FakePlanner(), FakeNav(), FakeFrame(), FakePerception(), ROOMS)
    return TestClient(create_agent_app(agent, FakeFrame()))


def test_root_html():
    r = make_client().get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()


def test_status():
    assert make_client().get("/status").json() == {"connected": False, "mock": True}


def test_ws_runs_pipeline_to_answer():
    with make_client().websocket_connect("/ws") as ws:
        ws.send_json({"command": "đồ ăn nóng chưa?"})
        events = []
        while True:
            m = ws.receive_json()
            events.append(m)
            if m["type"] in ("answer", "error"):
                break
    assert events[-1]["type"] == "answer"
    assert events[-1]["state"] == "ON"
    assert any(e["type"] == "step" and e.get("id") == "nav" for e in events)


class FakeMotion:
    def run(self, cmd, cancel=None):
        yield {"type": "step", "id": "motion", "status": "running", "title": "Moving"}
        yield {"type": "nav", "distance_remaining": 0.3}
        yield {"type": "step", "id": "motion", "status": "done"}
        yield {"type": "answer", "text": "Done.", "state": "UNKNOWN"}


def test_motion_command_routes_to_motion():
    agent = Agent(FakePlanner(), FakeNav(), FakeFrame(), FakePerception(), ROOMS)
    client = TestClient(create_agent_app(agent, FakeFrame(), motion=FakeMotion()))
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"command": "move forward 75 cm"})
        events = []
        while True:
            m = ws.receive_json()
            events.append(m)
            if m["type"] in ("answer", "error"):
                break
    # đi qua motion (không qua nav/plan của agentic)
    assert any(e.get("id") == "motion" for e in events if e["type"] == "step")
    assert not any(e.get("id") == "plan" for e in events if e["type"] == "step")
    assert events[-1]["type"] == "answer"
