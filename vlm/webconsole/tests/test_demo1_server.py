import numpy as np
from fastapi.testclient import TestClient
from vlm.webconsole.demo1.agent_server import create_agent_app


class FakeFrame:
    is_mock = True
    is_connected = False

    def get_latest_frame(self):
        return np.zeros((48, 64, 3), dtype=np.uint8)


class FakeMotion:
    def run(self, cmd, cancel=None):
        yield {"type": "step", "id": "motion", "status": "running", "title": "Moving"}
        yield {"type": "nav", "distance_remaining": 0.3}
        yield {"type": "step", "id": "motion", "status": "done"}
        yield {"type": "answer", "text": "Done.", "state": "UNKNOWN"}

    def estop(self):
        pass


class FakeAction:
    def run(self, act, cancel=None):
        yield {"type": "step", "id": "action", "status": "running", "title": act["vi"]}
        yield {"type": "step", "id": "action", "status": "done"}
        yield {"type": "answer", "text": "ok", "state": "UNKNOWN"}


class FakeNavLoop:
    def run(self, command, cancel=None):
        yield {"type": "step", "id": "vla", "status": "running", "title": "Bước 1"}
        yield {"type": "answer", "text": "đã tới", "state": "YES"}


def _drain(ws):
    events = []
    while True:
        m = ws.receive_json()
        events.append(m)
        if m["type"] in ("answer", "error"):
            return events


def test_root_html():
    r = TestClient(create_agent_app(FakeFrame())).get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()


def test_status():
    assert TestClient(create_agent_app(FakeFrame())).get("/status").json() == {
        "connected": False, "mock": True}


def test_actions_endpoint():
    data = TestClient(create_agent_app(FakeFrame())).get("/actions").json()
    assert isinstance(data, list) and any(a["api_id"] == 1016 for a in data)


def test_motion_command_routes_to_motion():
    client = TestClient(create_agent_app(FakeFrame(), motion=FakeMotion(),
                                         navloop=FakeNavLoop()))
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"command": "move forward 75 cm"})
        events = _drain(ws)
    assert any(e.get("id") == "motion" for e in events if e["type"] == "step")
    assert not any(e.get("id") == "vla" for e in events if e["type"] == "step")
    assert events[-1]["type"] == "answer"


def test_action_command_routes_to_action():
    client = TestClient(create_agent_app(FakeFrame(), action=FakeAction(),
                                         navloop=FakeNavLoop()))
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"command": "đứng dậy"})
        events = _drain(ws)
    assert any(e.get("id") == "action" for e in events if e["type"] == "step")
    assert not any(e.get("id") == "vla" for e in events if e["type"] == "step")


def test_natural_command_routes_to_navloop():
    """Lệnh ngôn ngữ tự nhiên (không phải move/turn/action) -> VLA loop (VLM)."""
    client = TestClient(create_agent_app(FakeFrame(), motion=FakeMotion(),
                                         action=FakeAction(), navloop=FakeNavLoop()))
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"command": "đi tới cái bình nước rồi rẽ phải, gặp ghế thì dừng"})
        events = _drain(ws)
    assert any(e.get("id") == "vla" for e in events if e["type"] == "step")
    assert events[-1]["type"] == "answer" and events[-1]["state"] == "YES"


def test_natural_command_without_navloop_errors():
    client = TestClient(create_agent_app(FakeFrame(), motion=FakeMotion()))
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"command": "đi tới cái ghế"})
        m = ws.receive_json()
    assert m["type"] == "error" and "VLM nav" in m["message"]


class FakeAnn:
    label = "bottle"
    def target_box(self): return (1000, 100, 1100, 300)
    def frame_height(self): return 720
    def frame_width(self): return 1280


def test_debug_endpoint_metrics():
    client = TestClient(create_agent_app(FakeFrame(), motion=FakeMotion(),
                                         annotator=FakeAnn(), navloop=FakeNavLoop()))
    d = client.get("/debug").json()
    assert d["target_detected"] is True
    assert d["label"] == "bottle"
    assert d["yolo_gap_px"] == 420                 # 720 - 300
    assert d["center_offset_deg"] is not None and d["center_offset_deg"] > 0


def test_debug_endpoint_no_target():
    d = TestClient(create_agent_app(FakeFrame())).get("/debug").json()
    assert d["target_detected"] is False
    assert d["yolo_gap_px"] is None


def test_estop_returns_halt_message():
    client = TestClient(create_agent_app(FakeFrame(), motion=FakeMotion()))
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"action": "estop"})
        m = ws.receive_json()
    assert m["type"] == "error" and "EMERGENCY" in m["message"]
