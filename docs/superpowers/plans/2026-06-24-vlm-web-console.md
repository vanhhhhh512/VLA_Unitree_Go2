# VLM Web Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Web GUI localhost cho phép nhập lệnh tự do, lấy ảnh live từ Unitree Go2 qua WebRTC, chạy Qwen2.5-VL, stream reasoning token-by-token và hiển thị ảnh đã vẽ bounding box, giao diện glassmorphism.

**Architecture:** Một tiến trình FastAPI/asyncio. Background task giữ kết nối WebRTC tới Go2 và lưu frame mới nhất. `/video_feed` phát MJPEG live; `/ws` nhận lệnh → snapshot frame → `VLMEngine.stream_infer` yield token → vẽ bbox → trả ảnh base64. Logic WebRTC và VLM tách thành module dùng lại được; server nhận chúng qua dependency injection để test bằng fake.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, websockets, OpenCV, numpy, Pillow, PyTorch + transformers (Qwen2.5-VL-3B-Instruct), aiortc (qua `Go2Connection` có sẵn), pytest.

## Global Constraints

- Python 3.12 (môi trường hiện tại).
- numpy==1.26.4 (đã ghim trong `requirements.txt`) — không nâng version.
- Tái dùng `Go2Connection` tại `go2_robot_sdk/go2_robot_sdk/infrastructure/webrtc/go2_connection.py`; constructor: `Go2Connection(robot_ip, robot_num, token="", on_validated=None, on_message=None, on_open=None, on_video_frame=None, decode_lidar=True)`; methods `await connect()`, `await disconnect()`, `await disableTrafficSaving(bool)`, `data_channel.send(str)`.
- Model id cố định: `Qwen/Qwen2.5-VL-3B-Instruct`.
- Frame nội bộ là numpy **BGR** (giống output `frame.to_ndarray(format="bgr24")` và `cv2`).
- Bbox từ model theo thứ tự `[ymin, xmin, ymax, xmax]`, thang đo 0–1000, scale = giá trị / 1000 * kích thước ảnh.
- Tất cả code mới nằm trong package `vlm/webconsole/`. Tests trong `vlm/webconsole/tests/`.
- Server mặc định cổng `8000`, host `127.0.0.1`.
- Đọc `ROBOT_IP` từ env; thiếu → mock mode.

---

## File Structure

- Create: `vlm/webconsole/__init__.py` — đánh dấu package.
- Create: `vlm/webconsole/vlm_engine.py` — helper thuần (`parse_boxes`, `draw_boxes`, `encode_frame_jpeg`, `build_messages`) + class `VLMEngine`.
- Create: `vlm/webconsole/frame_source.py` — class `FrameSource` (WebRTC + mock mode).
- Create: `vlm/webconsole/server.py` — `create_app(source, engine)` + `main()`.
- Create: `vlm/webconsole/webui/index.html` — UI glassmorphism (CSS/JS inline).
- Create: `vlm/webconsole/tests/__init__.py`
- Create: `vlm/webconsole/tests/test_vlm_engine.py`
- Create: `vlm/webconsole/tests/test_frame_source.py`
- Create: `vlm/webconsole/tests/test_server.py`
- Modify: `requirements.txt` — thêm `fastapi`, `uvicorn[standard]`, `transformers`, `qwen-vl-utils`, `pillow`.
- Create: `vlm/webconsole/README.md` — cách chạy.

---

### Task 1: Scaffold package + dependencies

**Files:**
- Create: `vlm/webconsole/__init__.py`
- Create: `vlm/webconsole/tests/__init__.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: package `vlm.webconsole` import được.

- [ ] **Step 1: Tạo file package rỗng**

Tạo `vlm/webconsole/__init__.py` với nội dung:

```python
"""VLM Web Console cho Unitree Go2."""
```

Tạo `vlm/webconsole/tests/__init__.py` rỗng (không nội dung).

- [ ] **Step 2: Thêm dependencies**

Mở `requirements.txt` và thêm các dòng sau vào cuối file (sau `numpy==1.26.4`):

```
fastapi
uvicorn[standard]
transformers
qwen-vl-utils
pillow
```

- [ ] **Step 3: Cài dependencies**

Run: `pip install fastapi "uvicorn[standard]" pillow`
Expected: cài thành công (transformers/qwen-vl-utils thường đã có sẵn cùng torch; nếu thiếu thì `pip install transformers qwen-vl-utils`).

- [ ] **Step 4: Verify import**

Run: `python -c "import vlm.webconsole; import fastapi, uvicorn; print('ok')"`
(Chạy từ `/home/dsc-labs/ros2_ws/src`.)
Expected: in ra `ok`.

- [ ] **Step 5: Commit**

```bash
git add vlm/webconsole/__init__.py vlm/webconsole/tests/__init__.py requirements.txt
git commit -m "feat(vlm-webconsole): scaffold package and add web deps"
```

---

### Task 2: Pure helpers — parse_boxes, draw_boxes, encode_frame_jpeg, build_messages

**Files:**
- Create: `vlm/webconsole/vlm_engine.py`
- Test: `vlm/webconsole/tests/test_vlm_engine.py`

**Interfaces:**
- Consumes: numpy, cv2, PIL.
- Produces:
  - `parse_boxes(text: str, width: int, height: int) -> list[tuple[int, int, int, int]]` — list `(x1, y1, x2, y2)` đã scale & clamp về trong ảnh.
  - `draw_boxes(frame_bgr: np.ndarray, boxes: list[tuple[int,int,int,int]], label: str) -> np.ndarray` — trả ảnh BGR mới có box + label.
  - `encode_frame_jpeg(frame_bgr: np.ndarray, quality: int = 80) -> bytes` — JPEG bytes.
  - `build_messages(pil_image, prompt: str) -> list[dict]` — messages cho Qwen chat template.

- [ ] **Step 1: Viết test thất bại**

Tạo `vlm/webconsole/tests/test_vlm_engine.py`:

```python
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
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `cd /home/dsc-labs/ros2_ws/src && python -m pytest vlm/webconsole/tests/test_vlm_engine.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` (chưa có `vlm_engine`).

- [ ] **Step 3: Viết implementation tối thiểu**

Tạo `vlm/webconsole/vlm_engine.py`:

```python
"""VLM helpers + Qwen2.5-VL engine."""
import re
import cv2
import numpy as np

BOX_PATTERN = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]")


def parse_boxes(text, width, height):
    """Parse box '[ymin,xmin,ymax,xmax]' (0-1000) -> [(x1,y1,x2,y2)] theo pixel."""
    boxes = []
    for ymin, xmin, ymax, xmax in BOX_PATTERN.findall(text):
        x1 = int(int(xmin) * width / 1000.0)
        y1 = int(int(ymin) * height / 1000.0)
        x2 = int(int(xmax) * width / 1000.0)
        y2 = int(int(ymax) * height / 1000.0)
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
        boxes.append((x1, y1, x2, y2))
    return boxes


def draw_boxes(frame_bgr, boxes, label):
    """Vẽ box + label lên bản copy của frame, trả frame mới."""
    out = frame_bgr.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out, label, (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
        )
    return out


def encode_frame_jpeg(frame_bgr, quality=80):
    """Encode BGR frame thành JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def build_messages(pil_image, prompt):
    """Tạo messages cho Qwen chat template."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `cd /home/dsc-labs/ros2_ws/src && python -m pytest vlm/webconsole/tests/test_vlm_engine.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add vlm/webconsole/vlm_engine.py vlm/webconsole/tests/test_vlm_engine.py
git commit -m "feat(vlm-webconsole): pure helpers for boxes, jpeg, messages"
```

---

### Task 3: VLMEngine class (Qwen streaming wrapper)

**Files:**
- Modify: `vlm/webconsole/vlm_engine.py`
- Test: `vlm/webconsole/tests/test_vlm_engine.py` (thêm test)

**Interfaces:**
- Consumes: `build_messages`, transformers, torch, PIL, qwen_vl_utils.
- Produces:
  - `class VLMEngine`:
    - `__init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", device=None)` — chưa load model.
    - `loaded -> bool` (property).
    - `load() -> None` — load model + processor (chọn cuda/cpu).
    - `stream_infer(self, frame_bgr, prompt) -> Iterator[str]` — yield từng đoạn text khi model sinh; raise `RuntimeError` nếu chưa `load()`.

  Server sẽ tích lũy text từ generator để parse box sau khi stream xong.

- [ ] **Step 1: Viết test thất bại**

Thêm vào cuối `vlm/webconsole/tests/test_vlm_engine.py`:

```python
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
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `cd /home/dsc-labs/ros2_ws/src && python -m pytest vlm/webconsole/tests/test_vlm_engine.py -k engine -v`
Expected: FAIL — `ImportError: cannot import name 'VLMEngine'`.

- [ ] **Step 3: Viết implementation**

Thêm vào cuối `vlm/webconsole/vlm_engine.py`:

```python
import threading
from PIL import Image as PILImage


class VLMEngine:
    """Wrapper Qwen2.5-VL với streaming token."""

    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", device=None):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None

    @property
    def loaded(self):
        return self.model is not None and self.processor is not None

    def load(self):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("[VLMEngine] WARNING: chạy trên CPU, inference sẽ rất chậm.")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        print(f"[VLMEngine] Model loaded on {self.device}.")

    def stream_infer(self, frame_bgr, prompt):
        if not self.loaded:
            raise RuntimeError("VLMEngine chưa load(). Gọi load() trước.")

        from transformers import TextIteratorStreamer
        from qwen_vl_utils import process_vision_info

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)
        messages = build_messages(pil_image, prompt)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs = dict(**inputs, max_new_tokens=256, streamer=streamer)
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        thread.join()
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `cd /home/dsc-labs/ros2_ws/src && python -m pytest vlm/webconsole/tests/test_vlm_engine.py -v`
Expected: PASS (9 passed). (Test không load model thật nên không cần GPU.)

- [ ] **Step 5: Commit**

```bash
git add vlm/webconsole/vlm_engine.py vlm/webconsole/tests/test_vlm_engine.py
git commit -m "feat(vlm-webconsole): VLMEngine streaming wrapper for Qwen2.5-VL"
```

---

### Task 4: FrameSource (WebRTC + mock mode)

**Files:**
- Create: `vlm/webconsole/frame_source.py`
- Test: `vlm/webconsole/tests/test_frame_source.py`

**Interfaces:**
- Consumes: `Go2Connection`, numpy, cv2.
- Produces:
  - `class FrameSource`:
    - `__init__(self, robot_ip: str | None)`.
    - `is_mock -> bool` (property) — True khi `robot_ip` falsy.
    - `is_connected -> bool` (property) — True chỉ khi WebRTC validated (mock = False).
    - `get_latest_frame(self) -> np.ndarray | None` — frame BGR mới nhất (mock: ảnh test).
    - `async start(self) -> None` — mock: tạo ảnh test ngay; real: kết nối Go2 nền.
    - `async stop(self) -> None`.

- [ ] **Step 1: Viết test thất bại**

Tạo `vlm/webconsole/tests/test_frame_source.py`:

```python
import asyncio
import numpy as np
from vlm.webconsole.frame_source import FrameSource


def test_mock_mode_when_no_ip():
    src = FrameSource(robot_ip=None)
    assert src.is_mock is True
    assert src.is_connected is False


def test_mock_start_provides_frame():
    src = FrameSource(robot_ip="")
    asyncio.run(src.start())
    frame = src.get_latest_frame()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3 and frame.shape[2] == 3


def test_real_mode_no_frame_before_start():
    src = FrameSource(robot_ip="192.168.1.10")
    assert src.is_mock is False
    assert src.get_latest_frame() is None
    assert src.is_connected is False
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `cd /home/dsc-labs/ros2_ws/src && python -m pytest vlm/webconsole/tests/test_frame_source.py -v`
Expected: FAIL — `ModuleNotFoundError: vlm.webconsole.frame_source`.

- [ ] **Step 3: Viết implementation**

Tạo `vlm/webconsole/frame_source.py`:

```python
"""Nguồn frame: WebRTC tới Go2 hoặc mock mode."""
import sys
import asyncio
import json
import numpy as np
import cv2

sys.path.insert(0, "/home/dsc-labs/ros2_ws/src/go2_robot_sdk")


def _make_mock_frame():
    frame = np.full((480, 640, 3), 30, dtype=np.uint8)
    cv2.putText(frame, "MOCK MODE - no ROBOT_IP", (60, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 255), 2)
    return frame


class FrameSource:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        self._latest = None
        self._connected = False
        self._conn = None

    @property
    def is_mock(self):
        return not self.robot_ip

    @property
    def is_connected(self):
        return self._connected

    def get_latest_frame(self):
        return self._latest

    async def start(self):
        if self.is_mock:
            self._latest = _make_mock_frame()
            return
        await self._connect_webrtc()

    async def _connect_webrtc(self):
        from go2_robot_sdk.infrastructure.webrtc.go2_connection import Go2Connection
        from go2_robot_sdk.domain.constants import RTC_TOPIC

        async def on_video_frame(track, robot_id):
            while True:
                try:
                    frame = await track.recv()
                    self._latest = frame.to_ndarray(format="bgr24")
                except Exception as e:
                    print(f"[FrameSource] video stream closed: {e}")
                    break

        def on_validated(robot_num):
            self._connected = True
            asyncio.create_task(self._conn.disableTrafficSaving(True))
            try:
                for topic in RTC_TOPIC.values():
                    self._conn.data_channel.send(
                        json.dumps({"type": "subscribe", "topic": topic})
                    )
            except Exception as e:
                print(f"[FrameSource] subscribe failed: {e}")

        self._conn = Go2Connection(
            robot_ip=self.robot_ip, robot_num=0, token="",
            on_validated=on_validated, on_video_frame=on_video_frame,
            decode_lidar=False,
        )
        await self._conn.connect()

    async def stop(self):
        self._connected = False
        if self._conn is not None:
            await self._conn.disconnect()
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `cd /home/dsc-labs/ros2_ws/src && python -m pytest vlm/webconsole/tests/test_frame_source.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add vlm/webconsole/frame_source.py vlm/webconsole/tests/test_frame_source.py
git commit -m "feat(vlm-webconsole): FrameSource with WebRTC and mock mode"
```

---

### Task 5: FastAPI server (routes + WebSocket)

**Files:**
- Create: `vlm/webconsole/server.py`
- Test: `vlm/webconsole/tests/test_server.py`

**Interfaces:**
- Consumes: `FrameSource`, `VLMEngine`, helpers (`parse_boxes`, `draw_boxes`, `encode_frame_jpeg`), FastAPI.
- Produces:
  - `create_app(source, engine) -> FastAPI` — DI để test bằng fake.
    - `GET /` → HTML từ `webui/index.html`.
    - `GET /status` → `{"connected": bool, "mock": bool}`.
    - `GET /video_feed` → MJPEG (`multipart/x-mixed-replace; boundary=frame`).
    - `WS /ws` → nhận `{"prompt": str}`; nếu đang xử lý → gửi `{"type":"busy"}`; ngược lại snapshot frame, stream `{"type":"token","text":...}`, rồi `{"type":"image","data": "<base64>"}` (nếu có box) và `{"type":"done"}`; lỗi → `{"type":"error","message":...}`.
  - `main()` — dựng `FrameSource(os.getenv("ROBOT_IP"))` + `VLMEngine()`, `engine.load()`, chạy uvicorn ở `127.0.0.1:8000`, gọi `source.start()` ở startup event.

- [ ] **Step 1: Viết test thất bại**

Tạo `vlm/webconsole/tests/test_server.py`:

```python
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
    # base64 hợp lệ, decode ra JPEG
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
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `cd /home/dsc-labs/ros2_ws/src && python -m pytest vlm/webconsole/tests/test_server.py -v`
Expected: FAIL — `ModuleNotFoundError: vlm.webconsole.server`.

- [ ] **Step 3: Viết implementation**

Tạo `vlm/webconsole/server.py`:

```python
"""FastAPI server cho VLM Web Console."""
import os
import base64
import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

from vlm.webconsole.vlm_engine import parse_boxes, draw_boxes, encode_frame_jpeg

WEBUI = Path(__file__).parent / "webui" / "index.html"


def create_app(source, engine):
    app = FastAPI()
    app.state.busy = False

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return WEBUI.read_text(encoding="utf-8")

    @app.get("/status")
    async def status():
        return JSONResponse({
            "connected": bool(source.is_connected),
            "mock": bool(source.is_mock),
        })

    @app.get("/video_feed")
    async def video_feed():
        async def gen():
            while True:
                frame = source.get_latest_frame()
                if frame is not None:
                    jpg = encode_frame_jpeg(frame)
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + jpg + b"\r\n")
                await asyncio.sleep(0.05)  # ~20 fps
        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                prompt = (data or {}).get("prompt", "").strip()
                if not prompt:
                    continue
                if app.state.busy:
                    await websocket.send_json({"type": "busy"})
                    continue
                app.state.busy = True
                try:
                    await _handle_prompt(websocket, source, engine, prompt)
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
                finally:
                    app.state.busy = False
        except WebSocketDisconnect:
            return

    return app


async def _handle_prompt(websocket, source, engine, prompt):
    frame = source.get_latest_frame()
    if frame is None:
        await websocket.send_json(
            {"type": "error", "message": "Chưa có ảnh từ robot."}
        )
        return

    snapshot = frame.copy()
    loop = asyncio.get_event_loop()
    full = []

    # Bơm generator (blocking) qua executor, đẩy token ra WS.
    def produce(q):
        try:
            for chunk in engine.stream_infer(snapshot, prompt):
                asyncio.run_coroutine_threadsafe(q.put(("token", chunk)), loop)
            asyncio.run_coroutine_threadsafe(q.put(("end", None)), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(q.put(("err", str(e))), loop)

    q: asyncio.Queue = asyncio.Queue()
    loop.run_in_executor(None, produce, q)

    while True:
        kind, payload = await q.get()
        if kind == "token":
            full.append(payload)
            await websocket.send_json({"type": "token", "text": payload})
        elif kind == "err":
            await websocket.send_json({"type": "error", "message": payload})
            return
        else:  # end
            break

    text = "".join(full)
    h, w = snapshot.shape[:2]
    boxes = parse_boxes(text, w, h)
    if boxes:
        drawn = draw_boxes(snapshot, boxes, prompt)
        b64 = base64.b64encode(encode_frame_jpeg(drawn)).decode("ascii")
        await websocket.send_json({"type": "image", "data": b64})
    await websocket.send_json({"type": "done"})


def main():
    import uvicorn
    from vlm.webconsole.frame_source import FrameSource
    from vlm.webconsole.vlm_engine import VLMEngine

    source = FrameSource(os.getenv("ROBOT_IP"))
    engine = VLMEngine()
    print("[server] Loading VLM model...")
    engine.load()

    app = create_app(source, engine)

    @app.on_event("startup")
    async def _startup():
        await source.start()

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Tạo placeholder webui để test `/` chạy**

Tạo tạm `vlm/webconsole/webui/index.html` (sẽ thay ở Task 6):

```html
<!doctype html>
<html><head><meta charset="utf-8"><title>VLM Console</title></head>
<body>placeholder</body></html>
```

- [ ] **Step 5: Chạy test để xác nhận pass**

Run: `cd /home/dsc-labs/ros2_ws/src && python -m pytest vlm/webconsole/tests/test_server.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add vlm/webconsole/server.py vlm/webconsole/tests/test_server.py vlm/webconsole/webui/index.html
git commit -m "feat(vlm-webconsole): FastAPI server with MJPEG feed and WS streaming"
```

---

### Task 6: Glassmorphism UI

**Files:**
- Modify: `vlm/webconsole/webui/index.html` (thay placeholder bằng UI thật)

**Interfaces:**
- Consumes: `GET /video_feed`, `GET /status`, `WS /ws` với message types `token` / `image` / `busy` / `error` / `done`.
- Produces: trang web hoàn chỉnh.

- [ ] **Step 1: Viết UI**

Thay toàn bộ nội dung `vlm/webconsole/webui/index.html`:

```html
<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VLM Console · Unitree Go2</title>
<style>
  :root { --glass: rgba(255,255,255,0.08); --line: rgba(255,255,255,0.15); }
  * { box-sizing: border-box; }
  body {
    margin: 0; height: 100vh; font-family: "Segoe UI", system-ui, sans-serif;
    color: #eef; overflow: hidden;
    background: radial-gradient(1200px 800px at 10% 10%, #1b2a5e, transparent),
                radial-gradient(1000px 700px at 90% 90%, #3a1b5e, transparent),
                linear-gradient(135deg, #0a0e27, #0e1430);
  }
  .app { display: grid; grid-template-columns: 1.2fr 1fr; gap: 18px;
         height: 100vh; padding: 18px; }
  .panel {
    background: var(--glass); border: 1px solid var(--line);
    border-radius: 22px; backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.4); overflow: hidden;
    display: flex; flex-direction: column;
  }
  .panel-head { padding: 16px 20px; display: flex; align-items: center;
                justify-content: space-between; border-bottom: 1px solid var(--line); }
  .panel-head h1 { font-size: 16px; margin: 0; letter-spacing: .5px; font-weight: 600; }
  .badge { font-size: 12px; padding: 5px 12px; border-radius: 999px;
           display: flex; align-items: center; gap: 7px; }
  .dot { width: 9px; height: 9px; border-radius: 50%; }
  .badge.on  { background: rgba(0,255,150,.12); color: #7fffc4; }
  .badge.on .dot  { background: #2bff9c; box-shadow: 0 0 10px #2bff9c; }
  .badge.off { background: rgba(255,80,80,.12); color: #ff9d9d; }
  .badge.off .dot { background: #ff5b5b; box-shadow: 0 0 10px #ff5b5b; }
  .video-wrap { flex: 1; display: flex; align-items: center; justify-content: center;
                padding: 16px; }
  .video-wrap img { max-width: 100%; max-height: 100%; border-radius: 14px;
                    border: 1px solid var(--line); }
  .chat { flex: 1; overflow-y: auto; padding: 18px; display: flex;
          flex-direction: column; gap: 14px; }
  .msg { max-width: 88%; padding: 12px 16px; border-radius: 16px; line-height: 1.5;
         font-size: 14px; white-space: pre-wrap; animation: rise .25s ease; }
  @keyframes rise { from { opacity: 0; transform: translateY(8px); } }
  .msg.user { align-self: flex-end; background: linear-gradient(135deg,#5b8cff,#8a5bff);
              color: #fff; border-bottom-right-radius: 4px; }
  .msg.bot  { align-self: flex-start; background: rgba(255,255,255,0.06);
              border: 1px solid var(--line); border-bottom-left-radius: 4px; }
  .msg img { display: block; margin-top: 10px; max-width: 100%; border-radius: 12px; }
  .cursor { display: inline-block; width: 8px; height: 15px; background: #8a5bff;
            margin-left: 2px; animation: blink 1s steps(2) infinite; vertical-align: -2px; }
  @keyframes blink { 50% { opacity: 0; } }
  .composer { padding: 16px; border-top: 1px solid var(--line); display: flex; gap: 10px; }
  .composer input {
    flex: 1; background: rgba(0,0,0,0.25); border: 1px solid var(--line);
    border-radius: 14px; padding: 13px 16px; color: #eef; font-size: 14px; outline: none;
  }
  .composer input:focus { border-color: #8a5bff; box-shadow: 0 0 0 3px rgba(138,91,255,.2); }
  .composer button {
    background: linear-gradient(135deg,#5b8cff,#8a5bff); border: 0; color: #fff;
    border-radius: 14px; padding: 0 22px; font-size: 14px; font-weight: 600;
    cursor: pointer; transition: transform .1s, filter .2s;
  }
  .composer button:hover { filter: brightness(1.1); }
  .composer button:active { transform: scale(.96); }
  .composer button:disabled { opacity: .5; cursor: not-allowed; }
  .empty { margin: auto; color: rgba(255,255,255,.4); font-size: 14px; text-align: center; }
</style>
</head>
<body>
<div class="app">
  <section class="panel">
    <div class="panel-head">
      <h1>🐾 GO2 · LIVE CAMERA</h1>
      <span id="badge" class="badge off"><span class="dot"></span><span id="badge-txt">…</span></span>
    </div>
    <div class="video-wrap"><img id="feed" src="/video_feed" alt="live feed"></div>
  </section>

  <section class="panel">
    <div class="panel-head"><h1>💬 VLM REASONING</h1></div>
    <div id="chat" class="chat">
      <div class="empty">Nhập lệnh để hỏi VLM về cảnh robot đang nhìn thấy.</div>
    </div>
    <form id="composer" class="composer">
      <input id="prompt" placeholder="vd: tìm điện thoại, mô tả cảnh, có người không?" autocomplete="off">
      <button id="send" type="submit">Gửi</button>
    </form>
  </section>
</div>

<script>
const chat = document.getElementById('chat');
const form = document.getElementById('composer');
const input = document.getElementById('prompt');
const sendBtn = document.getElementById('send');
const badge = document.getElementById('badge');
const badgeTxt = document.getElementById('badge-txt');

async function pollStatus() {
  try {
    const s = await (await fetch('/status')).json();
    const on = s.connected;
    badge.className = 'badge ' + (on ? 'on' : 'off');
    badgeTxt.textContent = on ? 'CONNECTED' : (s.mock ? 'MOCK MODE' : 'DISCONNECTED');
  } catch (e) { badge.className = 'badge off'; badgeTxt.textContent = 'OFFLINE'; }
}
pollStatus(); setInterval(pollStatus, 3000);

function clearEmpty() { const e = chat.querySelector('.empty'); if (e) e.remove(); }
function addMsg(cls) {
  clearEmpty();
  const d = document.createElement('div'); d.className = 'msg ' + cls;
  chat.appendChild(d); chat.scrollTop = chat.scrollHeight; return d;
}

let ws, busy = false;
function connectWS() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onclose = () => setTimeout(connectWS, 1500);
}
connectWS();

form.addEventListener('submit', (e) => {
  e.preventDefault();
  const prompt = input.value.trim();
  if (!prompt || busy || ws.readyState !== 1) return;
  addMsg('user').textContent = prompt;
  input.value = '';
  busy = true; sendBtn.disabled = true;

  const bot = addMsg('bot');
  const span = document.createElement('span'); bot.appendChild(span);
  const cursor = document.createElement('span'); cursor.className = 'cursor';
  bot.appendChild(cursor);

  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === 'token') {
      span.textContent += m.text; chat.scrollTop = chat.scrollHeight;
    } else if (m.type === 'image') {
      const img = new Image(); img.src = 'data:image/jpeg;base64,' + m.data;
      bot.appendChild(img); chat.scrollTop = chat.scrollHeight;
    } else if (m.type === 'busy') {
      span.textContent = '⏳ Đang xử lý lệnh trước, thử lại sau.';
      finish(bot, cursor);
    } else if (m.type === 'error') {
      span.textContent += '\n⚠️ ' + m.message; finish(bot, cursor);
    } else if (m.type === 'done') {
      finish(bot, cursor);
    }
  };
  ws.send(JSON.stringify({ prompt }));
});

function finish(bot, cursor) {
  cursor.remove(); busy = false; sendBtn.disabled = false; input.focus();
}
</script>
</body>
</html>
```

- [ ] **Step 2: Verify server tests vẫn pass (HTML hợp lệ)**

Run: `cd /home/dsc-labs/ros2_ws/src && python -m pytest vlm/webconsole/tests/ -v`
Expected: PASS toàn bộ (test_root_serves_html vẫn thấy `<html`).

- [ ] **Step 3: Verify thủ công bằng mock mode**

Run (terminal riêng, KHÔNG set ROBOT_IP):
`cd /home/dsc-labs/ros2_ws/src && python -m vlm.webconsole.server`
Mở `http://localhost:8000`. Kỳ vọng: thấy panel video (ảnh "MOCK MODE"), badge "MOCK MODE" màu đỏ. Gõ một lệnh → có bong bóng user + bong bóng bot. (Reasoning thật cần model load; nếu chưa muốn tải model, có thể bỏ qua kiểm tra phần token ở bước này.)
Dừng bằng Ctrl+C.

- [ ] **Step 4: Commit**

```bash
git add vlm/webconsole/webui/index.html
git commit -m "feat(vlm-webconsole): glassmorphism web UI"
```

---

### Task 7: README + full test run

**Files:**
- Create: `vlm/webconsole/README.md`

**Interfaces:**
- Consumes: tất cả module trên.
- Produces: tài liệu chạy.

- [ ] **Step 1: Viết README**

Tạo `vlm/webconsole/README.md`:

```markdown
# VLM Web Console — Unitree Go2

Web GUI localhost: nhập lệnh tự do → lấy ảnh live từ Go2 (WebRTC) → Qwen2.5-VL
stream reasoning + vẽ bounding box.

## Cài đặt

```bash
pip install -r ../../requirements.txt
```

## Chạy

```bash
export ROBOT_IP=192.168.123.161   # bỏ qua để chạy MOCK MODE (không cần robot)
python -m vlm.webconsole.server
```

Mở http://localhost:8000

- Panel trái: video live + trạng thái kết nối.
- Panel phải: gõ lệnh (vd "tìm điện thoại", "mô tả cảnh"), xem reasoning stream
  và ảnh bounding box.

## Test

```bash
cd /home/dsc-labs/ros2_ws/src
python -m pytest vlm/webconsole/tests/ -v
```
```

- [ ] **Step 2: Chạy toàn bộ test**

Run: `cd /home/dsc-labs/ros2_ws/src && python -m pytest vlm/webconsole/tests/ -v`
Expected: PASS toàn bộ (16 tests).

- [ ] **Step 3: Commit**

```bash
git add vlm/webconsole/README.md
git commit -m "docs(vlm-webconsole): add usage README"
```

---

## Notes về verify với robot thật

- Set `ROBOT_IP` đúng và bật robot, chạy server. Lần đầu sẽ tải model Qwen2.5-VL-3B
  (~vài GB) — cần GPU để chạy mượt.
- Kiểm tra: video live hiển thị, badge xanh "CONNECTED", gõ "tìm <vật>" → reasoning
  stream dần và ảnh có bbox xuất hiện trong chat.
- Nếu inference chậm/treo trên CPU: đó là kỳ vọng (xem cảnh báo log). Cần GPU CUDA.
