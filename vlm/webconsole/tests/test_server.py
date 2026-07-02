import base64
import numpy as np
from fastapi.testclient import TestClient
from vlm.webconsole.server import create_app


class FakeSource:
    is_mock = True
    is_connected = False

    def get_latest_frame(self):
        return np.zeros((48, 64, 3), dtype=np.uint8)


class FakeEngine:
    loaded = True

    def stream_infer(self, frame, prompt):
        yield "I see "
        yield "a phone [100, 100, 500, 500]."


def make_client():
    app = create_app(FakeSource(), FakeEngine())
    return TestClient(app)


def test_root_serves_html():
    r = make_client().get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()


def test_status_endpoint():
    r = make_client().get("/status")
    assert r.json() == {"connected": False, "mock": True}


def test_ws_streams_tokens_and_image():
    client = make_client()
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"prompt": "find the phone"})
        msgs = []
        while True:
            m = ws.receive_json()
            msgs.append(m)
            if m["type"] in ("done", "error"):
                break
    types = [m["type"] for m in msgs]
    assert "token" in types
    assert types[-1] == "done"
    tokens = "".join(m["text"] for m in msgs if m["type"] == "token")
    assert "phone" in tokens
    images = [m for m in msgs if m["type"] == "image"]
    assert len(images) == 1
    raw = base64.b64decode(images[0]["data"])
    assert raw[:2] == b"\xff\xd8"


def test_ws_no_box_no_image():
    class NoBoxEngine(FakeEngine):
        def stream_infer(self, frame, prompt):
            yield "nothing here"

    app = create_app(FakeSource(), NoBoxEngine())
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"prompt": "anything"})
        msgs = []
        while True:
            m = ws.receive_json()
            msgs.append(m)
            if m["type"] in ("done", "error"):
                break
    assert not any(m["type"] == "image" for m in msgs)
    assert msgs[-1]["type"] == "done"
