# StrikeRobot Demo1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Demo1 agentic: gõ câu lệnh tự nhiên → Qwen planner chọn phòng → Nav2 điều hướng (né vật cản) → VLM quan sát microwave BẬT/TẮT → trả kết luận; UI timeline trắng có logo StrikeRobot.

**Architecture:** Một tiến trình FastAPI + rclpy node (spin trong thread). Orchestrator `Agent` chạy chuỗi `plan → nav → perceive → answer` và phát step-events qua WebSocket. Các phần phụ thuộc ROS (`Nav2Navigator`, `RosFrameSource`) import lazy và được tiêm vào server qua dependency injection nên test chạy bằng conda với Fake, không cần ROS.

**Tech Stack:** Python, FastAPI, uvicorn, websockets, OpenCV, numpy, Pillow, PyYAML, Qwen2.5-VL (transformers), rclpy + nav2_simple_commander + cv_bridge (chỉ khi chạy thật).

## Global Constraints

- Tái dùng `vlm.webconsole.vlm_engine`: `VLMEngine.stream_infer(frame_bgr, prompt) -> Iterator[str]`, `parse_boxes(text, w, h) -> list[(x1,y1,x2,y2)]`, `draw_boxes(frame_bgr, boxes, label) -> np.ndarray`, `encode_frame_jpeg(frame_bgr, quality=80) -> bytes`.
- Code mới trong `vlm/webconsole/demo1/`. Tests trong `vlm/webconsole/tests/` (đặt tên `test_demo1_*.py`).
- Frame nội bộ là numpy **BGR**.
- Bbox model theo thứ tự `[ymin, xmin, ymax, xmax]`, thang 0–1000.
- Map frame `map`; vùng map hợp lệ x∈[-2.118, 6.732], y∈[-5.803, 3.847].
- Server demo1 mặc định host `0.0.0.0`, port **8001** (web console cũ giữ 8000).
- Cờ `DEMO_MOCK=1` → chạy không cần ROS (FakeNavigator + ảnh tĩnh).
- Module ROS (`rclpy`, `nav2_simple_commander`, `cv_bridge`) **chỉ import bên trong** `Nav2Navigator`/`RosFrameSource`, không import ở top-level (để test bằng conda).
- Chạy thật bằng python3.12 + `source /opt/ros/jazzy/setup.bash`.
- Lệnh test: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vlm/webconsole -v` (đã có `vlm/webconsole/pytest.ini` set `--import-mode=importlib`).
- Import nội bộ trong package dùng **relative** (`from .rooms import Room`) để chạy được cả tên `vlm.webconsole` (test) lẫn `webconsole` (runtime ROS).

---

## File Structure

- Create: `vlm/webconsole/demo1/__init__.py`
- Create: `vlm/webconsole/demo1/rooms.py` — `Room`, `load_rooms`, `random_rooms`, `save_rooms`.
- Create: `vlm/webconsole/demo1/planner.py` — `Plan`, `PlanError`, `parse_plan`, `Planner`.
- Create: `vlm/webconsole/demo1/perception.py` — `Result`, `parse_state`, `Perception`.
- Create: `vlm/webconsole/demo1/navigator.py` — `NavEvent`, `Nav2Navigator` (lazy ROS).
- Create: `vlm/webconsole/demo1/ros_frame_source.py` — `RosFrameSource` (lazy ROS).
- Create: `vlm/webconsole/demo1/agent.py` — `Agent` orchestrator.
- Create: `vlm/webconsole/demo1/agent_server.py` — `create_agent_app`, `main`.
- Create: `vlm/webconsole/webui/demo1.html` — UI timeline trắng + logo.
- Create: `vlm/webconsole/config/rooms.yaml` — sinh ở Task 2 (random).
- Create: `run_demo1.sh` — launcher python3.12 + ROS.
- Create: tests `test_demo1_rooms.py`, `test_demo1_planner.py`, `test_demo1_perception.py`, `test_demo1_agent.py`, `test_demo1_server.py`.
- Modify: `requirements.txt` — thêm `pyyaml`.

---

### Task 1: Scaffold demo1 package + PyYAML

**Files:**
- Create: `vlm/webconsole/demo1/__init__.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: package `vlm.webconsole.demo1` import được.

- [ ] **Step 1: Tạo package**

Tạo `vlm/webconsole/demo1/__init__.py`:

```python
"""StrikeRobot demo1: agentic nav + VLM."""
```

- [ ] **Step 2: Thêm dependency**

Thêm vào cuối `requirements.txt`:

```
pyyaml
```

- [ ] **Step 3: Cài + verify**

Run: `cd /home/dsc-labs/ros2_ws/src && pip install pyyaml && python -c "import vlm.webconsole.demo1, yaml; print('ok')"`
Expected: in `ok`.

- [ ] **Step 4: Commit**

```bash
git add vlm/webconsole/demo1/__init__.py requirements.txt
git commit -m "feat(demo1): scaffold package + pyyaml"
```

---

### Task 2: rooms.py — Room, load/random/save

**Files:**
- Create: `vlm/webconsole/demo1/rooms.py`
- Test: `vlm/webconsole/tests/test_demo1_rooms.py`

**Interfaces:**
- Produces:
  - `@dataclass Room{name: str, x: float, y: float, yaw: float, landmarks: list[str]}`.
  - `MAP_BOUNDS = (-2.118, 6.732, -5.803, 3.847)`  # (xmin, xmax, ymin, ymax)
  - `random_rooms() -> dict[str, Room]` — kitchen/living_room/bedroom, toạ độ trong MAP_BOUNDS, landmarks cố định.
  - `save_rooms(rooms: dict[str, Room], path: str) -> None` — ghi YAML.
  - `load_rooms(path: str) -> dict[str, Room]` — đọc YAML.

- [ ] **Step 1: Viết test thất bại**

Tạo `vlm/webconsole/tests/test_demo1_rooms.py`:

```python
import os
from vlm.webconsole.demo1.rooms import (
    Room, MAP_BOUNDS, random_rooms, save_rooms, load_rooms,
)


def test_random_rooms_keys_and_bounds():
    rooms = random_rooms()
    assert set(rooms) == {"kitchen", "living_room", "bedroom"}
    xmin, xmax, ymin, ymax = MAP_BOUNDS
    for r in rooms.values():
        assert isinstance(r, Room)
        assert xmin <= r.x <= xmax
        assert ymin <= r.y <= ymax
    assert "microwave" in rooms["kitchen"].landmarks


def test_save_and_load_roundtrip(tmp_path):
    rooms = {"kitchen": Room("kitchen", 1.0, 2.0, 0.5, ["microwave", "fridge"])}
    p = os.path.join(tmp_path, "rooms.yaml")
    save_rooms(rooms, p)
    loaded = load_rooms(p)
    assert loaded["kitchen"].x == 1.0
    assert loaded["kitchen"].landmarks == ["microwave", "fridge"]
    assert loaded["kitchen"].name == "kitchen"
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `cd /home/dsc-labs/ros2_ws/src && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vlm/webconsole/tests/test_demo1_rooms.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Viết implementation**

Tạo `vlm/webconsole/demo1/rooms.py`:

```python
"""Định nghĩa phòng + toạ độ trên map."""
import random
from dataclasses import dataclass, asdict

import yaml

# (xmin, xmax, ymin, ymax) theo mét, suy từ cty.yaml
MAP_BOUNDS = (-2.118, 6.732, -5.803, 3.847)

_LANDMARKS = {
    "kitchen": ["microwave", "fridge", "stove", "sink"],
    "living_room": ["sofa", "tv", "coffee table"],
    "bedroom": ["bed", "wardrobe", "lamp"],
}


@dataclass
class Room:
    name: str
    x: float
    y: float
    yaw: float
    landmarks: list


def random_rooms():
    xmin, xmax, ymin, ymax = MAP_BOUNDS
    rooms = {}
    for name, lm in _LANDMARKS.items():
        rooms[name] = Room(
            name=name,
            x=round(random.uniform(xmin + 0.5, xmax - 0.5), 2),
            y=round(random.uniform(ymin + 0.5, ymax - 0.5), 2),
            yaw=round(random.uniform(-3.14, 3.14), 2),
            landmarks=list(lm),
        )
    return rooms


def save_rooms(rooms, path):
    data = {}
    for name, r in rooms.items():
        d = asdict(r)
        d.pop("name")
        data[name] = d
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_rooms(path):
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    rooms = {}
    for name, d in data.items():
        rooms[name] = Room(
            name=name,
            x=float(d["x"]), y=float(d["y"]), yaw=float(d.get("yaw", 0.0)),
            landmarks=list(d.get("landmarks", [])),
        )
    return rooms
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `cd /home/dsc-labs/ros2_ws/src && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vlm/webconsole/tests/test_demo1_rooms.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Sinh file config rooms.yaml mặc định**

Run:
```bash
cd /home/dsc-labs/ros2_ws/src && mkdir -p vlm/webconsole/config && python -c "
from vlm.webconsole.demo1.rooms import random_rooms, save_rooms
save_rooms(random_rooms(), 'vlm/webconsole/config/rooms.yaml')
print(open('vlm/webconsole/config/rooms.yaml').read())
"
```
Expected: in ra nội dung YAML có kitchen/living_room/bedroom.

- [ ] **Step 6: Commit**

```bash
git add vlm/webconsole/demo1/rooms.py vlm/webconsole/tests/test_demo1_rooms.py vlm/webconsole/config/rooms.yaml
git commit -m "feat(demo1): rooms config (Room, load/random/save) + default rooms.yaml"
```

---

### Task 3: planner.py — Plan, parse_plan, Planner

**Files:**
- Create: `vlm/webconsole/demo1/planner.py`
- Test: `vlm/webconsole/tests/test_demo1_planner.py`

**Interfaces:**
- Consumes: `Room` (Task 2); một engine có `stream_infer(frame_bgr, prompt) -> Iterator[str]` (Qwen text — truyền frame nhỏ rỗng).
- Produces:
  - `@dataclass Plan{room: str, target_object: str, observation_question: str, reasoning: str}`.
  - `class PlanError(Exception)`.
  - `parse_plan(text: str, rooms: dict) -> Plan` — tách JSON `{room,target_object,observation_question,reasoning}`; raise `PlanError` nếu thiếu khoá hoặc `room` không thuộc rooms.
  - `class Planner` với `plan(command: str, rooms: dict) -> Plan`.

- [ ] **Step 1: Viết test thất bại**

Tạo `vlm/webconsole/tests/test_demo1_planner.py`:

```python
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
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `cd /home/dsc-labs/ros2_ws/src && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vlm/webconsole/tests/test_demo1_planner.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Viết implementation**

Tạo `vlm/webconsole/demo1/planner.py`:

```python
"""Planner: câu lệnh -> phòng đích + vật cần quan sát (Qwen text)."""
import re
import json
from dataclasses import dataclass

import numpy as np

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_REQUIRED = ("room", "target_object", "observation_question", "reasoning")


class PlanError(Exception):
    pass


@dataclass
class Plan:
    room: str
    target_object: str
    observation_question: str
    reasoning: str


def parse_plan(text, rooms):
    m = _JSON_RE.search(text or "")
    if not m:
        raise PlanError("Không tìm thấy JSON trong output planner.")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise PlanError(f"JSON planner lỗi: {e}")
    for k in _REQUIRED:
        if k not in data:
            raise PlanError(f"Thiếu khoá '{k}' trong plan.")
    if data["room"] not in rooms:
        raise PlanError(
            f"Phòng '{data['room']}' không có. Phòng hợp lệ: {list(rooms)}"
        )
    return Plan(
        room=data["room"],
        target_object=str(data["target_object"]),
        observation_question=str(data["observation_question"]),
        reasoning=str(data["reasoning"]),
    )


def build_prompt(command, rooms):
    lines = [f"- {n}: {', '.join(r.landmarks)}" for n, r in rooms.items()]
    rooms_block = "\n".join(lines)
    return (
        "Bạn là bộ lập kế hoạch cho robot. Dưới đây là các phòng và vật mốc:\n"
        f"{rooms_block}\n\n"
        f"Lệnh người dùng: \"{command}\"\n\n"
        "Hãy chọn MỘT phòng robot cần tới để trả lời lệnh, vật cần quan sát, và câu "
        "hỏi quan sát. CHỈ trả về JSON đúng định dạng:\n"
        '{"room": "<tên phòng>", "target_object": "<vật>", '
        '"observation_question": "<câu hỏi>", "reasoning": "<giải thích ngắn>"}'
    )


class Planner:
    def __init__(self, engine):
        self.engine = engine

    def plan(self, command, rooms):
        prompt = build_prompt(command, rooms)
        blank = np.zeros((8, 8, 3), dtype=np.uint8)
        text = "".join(self.engine.stream_infer(blank, prompt))
        return parse_plan(text, rooms)
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `cd /home/dsc-labs/ros2_ws/src && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vlm/webconsole/tests/test_demo1_planner.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add vlm/webconsole/demo1/planner.py vlm/webconsole/tests/test_demo1_planner.py
git commit -m "feat(demo1): LLM planner (parse_plan + Planner)"
```

---

### Task 4: perception.py — parse_state, Perception

**Files:**
- Create: `vlm/webconsole/demo1/perception.py`
- Test: `vlm/webconsole/tests/test_demo1_perception.py`

**Interfaces:**
- Consumes: `vlm.webconsole.vlm_engine` (`parse_boxes`, `draw_boxes`, `encode_frame_jpeg`); engine `stream_infer`.
- Produces:
  - `@dataclass Result{state: str, annotated_jpeg_b64: str | None, summary: str}` (state ∈ {"ON","OFF","UNKNOWN"}).
  - `parse_state(text: str) -> str` — suy ON/OFF/UNKNOWN từ text.
  - `class Perception`:
    - `observe(frame_bgr, target_object, question) -> Iterator[str]` (stream token).
    - `finalize(full_text, frame_bgr, target_object) -> Result`.

- [ ] **Step 1: Viết test thất bại**

Tạo `vlm/webconsole/tests/test_demo1_perception.py`:

```python
import base64
import numpy as np
from vlm.webconsole.demo1.perception import parse_state, Perception, Result


def test_parse_state_on():
    assert parse_state("The microwave is ON, the display is lit.") == "ON"
    assert parse_state("Lò vi sóng đang bật.") == "ON"


def test_parse_state_off():
    assert parse_state("The microwave is OFF.") == "OFF"
    assert parse_state("Lò đang tắt, không có đèn.") == "OFF"


def test_parse_state_unknown():
    assert parse_state("I cannot tell clearly.") == "UNKNOWN"


def test_finalize_draws_box_when_present():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    text = "microwave [100, 100, 300, 300]. It is ON."
    p = Perception(engine=None)
    res = p.finalize(text, frame, "microwave")
    assert isinstance(res, Result)
    assert res.state == "ON"
    assert res.annotated_jpeg_b64 is not None
    raw = base64.b64decode(res.annotated_jpeg_b64)
    assert raw[:2] == b"\xff\xd8"


def test_finalize_no_box_no_image():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = Perception(engine=None).finalize("It is OFF, no box.", frame, "microwave")
    assert res.annotated_jpeg_b64 is None
    assert res.state == "OFF"
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `cd /home/dsc-labs/ros2_ws/src && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vlm/webconsole/tests/test_demo1_perception.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Viết implementation**

Tạo `vlm/webconsole/demo1/perception.py`:

```python
"""Perception: VLM detect + phán đoán BẬT/TẮT."""
import base64
from dataclasses import dataclass

from ..vlm_engine import parse_boxes, draw_boxes, encode_frame_jpeg

_ON = ("is on", "turned on", "running", "đang bật", "bật", "sáng đèn", "lit", "active")
_OFF = ("is off", "turned off", "not running", "đang tắt", "tắt", "no light", "off")


@dataclass
class Result:
    state: str
    annotated_jpeg_b64: str
    summary: str


def parse_state(text):
    t = (text or "").lower()
    on = any(k in t for k in _ON)
    off = any(k in t for k in _OFF)
    if on and not off:
        return "ON"
    if off and not on:
        return "OFF"
    return "UNKNOWN"


def build_prompt(target_object, question):
    return (
        f"Look at the image. Detect the {target_object} and give its bounding box "
        f"as [ymin,xmin,ymax,xmax]. Then answer: {question} "
        f"Clearly say whether the {target_object} is ON or OFF and explain why "
        "(lights, display, interior light)."
    )


class Perception:
    def __init__(self, engine):
        self.engine = engine

    def observe(self, frame_bgr, target_object, question):
        prompt = build_prompt(target_object, question)
        return self.engine.stream_infer(frame_bgr, prompt)

    def finalize(self, full_text, frame_bgr, target_object):
        h, w = frame_bgr.shape[:2]
        boxes = parse_boxes(full_text, w, h)
        b64 = None
        if boxes:
            drawn = draw_boxes(frame_bgr, boxes, target_object)
            b64 = base64.b64encode(encode_frame_jpeg(drawn)).decode("ascii")
        state = parse_state(full_text)
        return Result(state=state, annotated_jpeg_b64=b64, summary=full_text.strip())
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `cd /home/dsc-labs/ros2_ws/src && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vlm/webconsole/tests/test_demo1_perception.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add vlm/webconsole/demo1/perception.py vlm/webconsole/tests/test_demo1_perception.py
git commit -m "feat(demo1): perception (parse_state + Perception)"
```

---

### Task 5: agent.py — orchestrator + NavEvent contract

**Files:**
- Create: `vlm/webconsole/demo1/agent.py`
- Test: `vlm/webconsole/tests/test_demo1_agent.py`

**Interfaces:**
- Consumes:
  - `planner.plan(command, rooms) -> Plan`.
  - `navigator.go_to(room) -> Iterator[dict]` với dict dạng `{"kind":"feedback","distance_remaining":float}` / `{"kind":"done","success":bool}` / `{"kind":"error","message":str}`.
  - `frame_source.get_latest_frame() -> np.ndarray | None`.
  - `perception.observe(frame, target, question) -> Iterator[str]`; `perception.finalize(text, frame, target) -> Result`.
  - `rooms: dict[str, Room]`.
- Produces:
  - `class Agent` với `async run(command) -> async-generator[dict]` phát các event đúng schema ở spec §5.

- [ ] **Step 1: Viết test thất bại**

Tạo `vlm/webconsole/tests/test_demo1_agent.py`:

```python
import asyncio
import numpy as np
from vlm.webconsole.demo1.rooms import Room
from vlm.webconsole.demo1.planner import Plan
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

    def finalize(self, text, frame, target):
        return Result("ON", "ZmFrZQ==", text)


def _collect(agent, command):
    async def go():
        return [ev async for ev in agent.run(command)]
    return asyncio.run(go())


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
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `cd /home/dsc-labs/ros2_ws/src && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vlm/webconsole/tests/test_demo1_agent.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Viết implementation**

Tạo `vlm/webconsole/demo1/agent.py`:

```python
"""Orchestrator demo1: plan -> nav -> perceive -> answer."""
from .planner import PlanError


class Agent:
    def __init__(self, planner, navigator, frame_source, perception, rooms):
        self.planner = planner
        self.navigator = navigator
        self.frame_source = frame_source
        self.perception = perception
        self.rooms = rooms

    async def run(self, command):
        # 1) PLAN
        yield {"type": "step", "id": "plan", "status": "running",
               "title": "Planner đang suy luận..."}
        try:
            plan = self.planner.plan(command, self.rooms)
        except PlanError as e:
            yield {"type": "error", "message": f"Planner: {e}"}
            return
        yield {"type": "token", "step": "plan", "text": plan.reasoning}
        yield {"type": "step", "id": "plan", "status": "done",
               "data": {"room": plan.room, "target_object": plan.target_object}}

        room = self.rooms[plan.room]

        # 2) NAV
        yield {"type": "step", "id": "nav", "status": "running",
               "title": f"Đang đi tới {room.name}",
               "data": {"room": room.name, "x": room.x, "y": room.y}}
        success = False
        for ev in self.navigator.go_to(room):
            if ev["kind"] == "feedback":
                yield {"type": "nav", "distance_remaining": ev["distance_remaining"]}
            elif ev["kind"] == "error":
                yield {"type": "step", "id": "nav", "status": "error"}
                yield {"type": "error", "message": f"Navigation: {ev['message']}"}
                return
            elif ev["kind"] == "done":
                success = ev.get("success", False)
        if not success:
            yield {"type": "step", "id": "nav", "status": "error"}
            yield {"type": "error", "message": "Navigation thất bại."}
            return
        yield {"type": "step", "id": "nav", "status": "done"}

        # 3) PERCEIVE
        yield {"type": "step", "id": "perceive", "status": "running",
               "title": f"Đang quan sát {plan.target_object}"}
        frame = self.frame_source.get_latest_frame()
        if frame is None:
            yield {"type": "step", "id": "perceive", "status": "error"}
            yield {"type": "error", "message": "Không có ảnh camera."}
            return
        parts = []
        for chunk in self.perception.observe(
            frame, plan.target_object, plan.observation_question
        ):
            parts.append(chunk)
            yield {"type": "token", "step": "perceive", "text": chunk}
        result = self.perception.finalize("".join(parts), frame, plan.target_object)
        if result.annotated_jpeg_b64:
            yield {"type": "image", "data": result.annotated_jpeg_b64}
        yield {"type": "step", "id": "perceive", "status": "done"}

        # 4) ANSWER
        label = {"ON": "BẬT", "OFF": "TẮT", "UNKNOWN": "KHÔNG RÕ"}[result.state]
        answer = (
            f"{plan.target_object} đang {label}. {result.summary}"
            if result.state != "UNKNOWN"
            else f"Chưa chắc chắn trạng thái {plan.target_object}. {result.summary}"
        )
        yield {"type": "answer", "text": answer, "state": result.state}
```

- [ ] **Step 4: Chạy test để xác nhận pass**

Run: `cd /home/dsc-labs/ros2_ws/src && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vlm/webconsole/tests/test_demo1_agent.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add vlm/webconsole/demo1/agent.py vlm/webconsole/tests/test_demo1_agent.py
git commit -m "feat(demo1): Agent orchestrator (plan->nav->perceive->answer)"
```

---

### Task 6: navigator.py + ros_frame_source.py (lazy ROS)

**Files:**
- Create: `vlm/webconsole/demo1/navigator.py`
- Create: `vlm/webconsole/demo1/ros_frame_source.py`

**Interfaces:**
- Consumes: `Room`; rclpy/nav2_simple_commander/cv_bridge (import lazy bên trong).
- Produces:
  - `class Nav2Navigator` với `go_to(room) -> Iterator[dict]` (schema như Task 5).
  - `class RosFrameSource` với `get_latest_frame() -> np.ndarray | None`, `is_connected -> bool`, `spin_once(timeout_sec)`.

Lưu ý: không có unit test ROS (cần rclpy). Verify bằng import smoke ở conda là KHÔNG được (ROS thiếu), nên chỉ kiểm tra **cú pháp** bằng `py_compile` và verify thật ở Task 8 (chạy launcher). Module này chỉ được import khi `DEMO_MOCK` không bật.

- [ ] **Step 1: Viết navigator.py**

Tạo `vlm/webconsole/demo1/navigator.py`:

```python
"""Nav2Navigator: bọc BasicNavigator (nav2_simple_commander)."""
import math
import time


def _yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class Nav2Navigator:
    def __init__(self, timeout_sec=120.0):
        from nav2_simple_commander.robot_navigator import BasicNavigator
        self._BasicNavigator = BasicNavigator
        self.nav = BasicNavigator()
        self.timeout_sec = timeout_sec

    def _make_pose(self, room):
        from geometry_msgs.msg import PoseStamped
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = float(room.x)
        pose.pose.position.y = float(room.y)
        qx, qy, qz, qw = _yaw_to_quat(float(room.yaw))
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def go_to(self, room):
        from nav2_simple_commander.robot_navigator import TaskResult
        pose = self._make_pose(room)
        pose.header.stamp = self.nav.get_clock().now().to_msg()
        self.nav.goToPose(pose)
        start = time.time()
        while not self.nav.isTaskComplete():
            fb = self.nav.getFeedback()
            if fb is not None:
                yield {"kind": "feedback",
                       "distance_remaining": float(fb.distance_remaining)}
            if time.time() - start > self.timeout_sec:
                self.nav.cancelTask()
                yield {"kind": "error", "message": "Quá thời gian điều hướng."}
                return
            time.sleep(0.5)
        result = self.nav.getResult()
        yield {"kind": "done", "success": result == TaskResult.SUCCEEDED}
```

- [ ] **Step 2: Viết ros_frame_source.py**

Tạo `vlm/webconsole/demo1/ros_frame_source.py`:

```python
"""RosFrameSource: subscribe /camera/image_raw -> latest BGR frame."""


class RosFrameSource:
    def __init__(self, node, image_topic="/camera/image_raw"):
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        self.node = node
        self.bridge = CvBridge()
        self._latest = None
        self._got = False
        self.sub = node.create_subscription(
            Image, image_topic, self._on_image, 10
        )

    def _on_image(self, msg):
        self._latest = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._got = True

    @property
    def is_connected(self):
        return self._got

    def get_latest_frame(self):
        return self._latest

    def spin_once(self, timeout_sec=0.05):
        import rclpy
        rclpy.spin_once(self.node, timeout_sec=timeout_sec)
```

- [ ] **Step 3: Kiểm tra cú pháp (py_compile)**

Run: `cd /home/dsc-labs/ros2_ws/src && python -m py_compile vlm/webconsole/demo1/navigator.py vlm/webconsole/demo1/ros_frame_source.py && echo "compile OK"`
Expected: in `compile OK` (không import ROS vì các import nằm trong hàm/ctor).

- [ ] **Step 4: Đảm bảo test cũ vẫn xanh (không bị import ROS)**

Run: `cd /home/dsc-labs/ros2_ws/src && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vlm/webconsole -q`
Expected: PASS toàn bộ (các test demo1 trước + webconsole cũ).

- [ ] **Step 5: Commit**

```bash
git add vlm/webconsole/demo1/navigator.py vlm/webconsole/demo1/ros_frame_source.py
git commit -m "feat(demo1): Nav2Navigator + RosFrameSource (lazy ROS imports)"
```

---

### Task 7: agent_server.py (FastAPI + DI) + mock fakes

**Files:**
- Create: `vlm/webconsole/demo1/agent_server.py`
- Test: `vlm/webconsole/tests/test_demo1_server.py`

**Interfaces:**
- Consumes: `Agent`, một `frame_source` có `get_latest_frame`/`is_mock`/`is_connected`; `encode_frame_jpeg`.
- Produces:
  - `create_agent_app(agent, frame_source) -> FastAPI` — routes `/`, `/assets/{f}`, `/status`, `/video_feed`, `/ws`.
  - `MockNavigator`, `MockFrameSource` (cho `DEMO_MOCK=1`).
  - `main()`.

- [ ] **Step 1: Viết test thất bại**

Tạo `vlm/webconsole/tests/test_demo1_server.py`:

```python
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

    def finalize(self, text, f, t):
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
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `cd /home/dsc-labs/ros2_ws/src && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vlm/webconsole/tests/test_demo1_server.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Viết implementation**

Tạo `vlm/webconsole/demo1/agent_server.py`:

```python
"""FastAPI server cho demo1 (DI + mock mode)."""
import os
import asyncio
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse

from ..vlm_engine import encode_frame_jpeg

WEBUI = Path(__file__).parent.parent / "webui" / "demo1.html"
ASSETS = Path(__file__).parent.parent / "webui" / "assets"


class MockFrameSource:
    is_mock = True
    is_connected = False

    def __init__(self):
        f = np.full((480, 640, 3), 245, dtype=np.uint8)
        cv2.putText(f, "DEMO_MOCK - no robot", (60, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2)
        self._f = f

    def get_latest_frame(self):
        return self._f


class MockNavigator:
    def go_to(self, room):
        for d in (2.0, 1.0, 0.3):
            yield {"kind": "feedback", "distance_remaining": d}
        yield {"kind": "done", "success": True}


def create_agent_app(agent, frame_source):
    app = FastAPI()
    app.state.busy = False

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return WEBUI.read_text(encoding="utf-8")

    @app.get("/assets/{name}")
    async def asset(name: str):
        p = ASSETS / name
        if not p.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(str(p))

    @app.get("/status")
    async def status():
        return JSONResponse({
            "connected": bool(getattr(frame_source, "is_connected", False)),
            "mock": bool(getattr(frame_source, "is_mock", False)),
        })

    @app.get("/video_feed")
    async def video_feed():
        async def gen():
            while True:
                frame = frame_source.get_latest_frame()
                if frame is not None:
                    jpg = encode_frame_jpeg(frame)
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + jpg + b"\r\n")
                await asyncio.sleep(0.05)
        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                command = (data or {}).get("command", "").strip()
                if not command:
                    continue
                if app.state.busy:
                    await websocket.send_json(
                        {"type": "error", "message": "Đang xử lý lệnh khác."})
                    continue
                app.state.busy = True
                try:
                    async for ev in agent.run(command):
                        await websocket.send_json(ev)
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})
                finally:
                    app.state.busy = False
        except WebSocketDisconnect:
            return

    return app


def main():
    import uvicorn
    from .rooms import load_rooms, random_rooms, save_rooms
    from .planner import Planner
    from .perception import Perception
    from .agent import Agent
    from ..vlm_engine import VLMEngine

    cfg = Path(__file__).parent.parent / "config" / "rooms.yaml"
    if cfg.is_file():
        rooms = load_rooms(str(cfg))
    else:
        rooms = random_rooms()
        cfg.parent.mkdir(parents=True, exist_ok=True)
        save_rooms(rooms, str(cfg))

    mock = os.getenv("DEMO_MOCK") == "1"
    engine = VLMEngine()

    if mock:
        frame_source = MockFrameSource()
        navigator = MockNavigator()
        print("[demo1] DEMO_MOCK=1 -> không cần ROS/robot.")
        if os.getenv("VLM_SKIP_MODEL") != "1":
            engine.load()
    else:
        import rclpy
        from .navigator import Nav2Navigator
        from .ros_frame_source import RosFrameSource
        import threading
        rclpy.init()
        node = rclpy.create_node("strikerobot_demo1")
        frame_source = RosFrameSource(node)
        threading.Thread(
            target=lambda: rclpy.spin(node), daemon=True).start()
        navigator = Nav2Navigator()
        engine.load()

    planner = Planner(engine)
    perception = Perception(engine)
    agent = Agent(planner, navigator, frame_source, perception, rooms)

    app = create_agent_app(agent, frame_source)
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Tạo placeholder webui/demo1.html để test `/`**

Tạo `vlm/webconsole/webui/demo1.html` (thay ở Task 8):

```html
<!doctype html>
<html><head><meta charset="utf-8"><title>StrikeRobot Demo1</title></head>
<body>placeholder</body></html>
```

- [ ] **Step 5: Chạy test để xác nhận pass**

Run: `cd /home/dsc-labs/ros2_ws/src && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vlm/webconsole/tests/test_demo1_server.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add vlm/webconsole/demo1/agent_server.py vlm/webconsole/tests/test_demo1_server.py vlm/webconsole/webui/demo1.html
git commit -m "feat(demo1): FastAPI agent server (DI) + mock navigator/frame"
```

---

### Task 8: UI demo1.html (timeline trắng + logo) + launcher

**Files:**
- Modify: `vlm/webconsole/webui/demo1.html`
- Create: `run_demo1.sh`

**Interfaces:**
- Consumes: `/`, `/assets/logo.jpg`, `/status`, `/video_feed`, `/ws` (events spec §5).
- Produces: UI hoàn chỉnh + launcher.

- [ ] **Step 1: Viết UI**

Thay toàn bộ `vlm/webconsole/webui/demo1.html`:

```html
<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StrikeRobot · Demo1</title>
<style>
  :root { --ink:#111418; --line:#e6e8eb; --muted:#6b7280; --accent:#111418;
          --ok:#16a34a; --run:#2563eb; --err:#dc2626; --bg:#f6f7f9; }
  * { box-sizing: border-box; }
  body { margin:0; height:100vh; font-family:"Segoe UI",system-ui,sans-serif;
         color:var(--ink); background:var(--bg); overflow:hidden; }
  header { display:flex; align-items:center; gap:14px; padding:14px 22px;
           background:#fff; border-bottom:1px solid var(--line); }
  header img { width:40px; height:40px; border-radius:9px; object-fit:cover; }
  header .t { font-weight:700; letter-spacing:.5px; font-size:18px; }
  header .s { color:var(--muted); font-size:13px; margin-left:4px; }
  .badge { margin-left:auto; font-size:12px; padding:6px 12px; border-radius:999px;
           display:flex; align-items:center; gap:7px; border:1px solid var(--line); }
  .dot { width:9px; height:9px; border-radius:50%; background:#cbd5e1; }
  .badge.on .dot { background:var(--ok); } .badge.off .dot { background:var(--err); }
  .app { display:grid; grid-template-columns:1.1fr 1fr; gap:18px; padding:18px;
         height:calc(100vh - 69px); }
  .card { background:#fff; border:1px solid var(--line); border-radius:16px;
          box-shadow:0 1px 3px rgba(0,0,0,.04); display:flex; flex-direction:column;
          overflow:hidden; }
  .card h2 { font-size:13px; text-transform:uppercase; letter-spacing:.6px;
             color:var(--muted); margin:0; padding:14px 18px; border-bottom:1px solid var(--line); }
  .video { flex:1; display:flex; align-items:center; justify-content:center; padding:16px; }
  .video img { max-width:100%; max-height:100%; border-radius:10px; border:1px solid var(--line); }
  .timeline { flex:1; overflow-y:auto; padding:16px 18px; }
  .step { display:flex; gap:12px; padding:10px 0; border-bottom:1px dashed var(--line); }
  .step:last-child { border-bottom:0; }
  .ico { width:30px; height:30px; flex:none; border-radius:8px; display:flex;
         align-items:center; justify-content:center; font-size:15px; background:#f1f5f9; }
  .step.running .ico { background:#dbeafe; } .step.done .ico { background:#dcfce7; }
  .step.error .ico { background:#fee2e2; }
  .step .body { flex:1; }
  .step .title { font-weight:600; font-size:14px; }
  .step .detail { color:var(--muted); font-size:13px; margin-top:3px; white-space:pre-wrap; }
  .step .detail img { display:block; margin-top:8px; max-width:100%; border-radius:8px; }
  .answer { margin:14px 18px; padding:16px; border-radius:12px; border:1px solid var(--line);
            background:#fafafa; display:none; }
  .answer.show { display:block; }
  .answer .lab { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.5px; }
  .answer .txt { font-size:16px; font-weight:600; margin-top:6px; }
  .answer .state { display:inline-block; margin-top:8px; padding:4px 12px; border-radius:999px;
                   font-size:13px; font-weight:700; }
  .answer .state.ON { background:#dcfce7; color:#166534; }
  .answer .state.OFF { background:#e5e7eb; color:#374151; }
  .answer .state.UNKNOWN { background:#fef3c7; color:#92400e; }
  .composer { display:flex; gap:10px; padding:14px 18px; border-top:1px solid var(--line); }
  .composer input { flex:1; border:1px solid var(--line); border-radius:10px; padding:12px 14px;
                    font-size:14px; outline:none; }
  .composer input:focus { border-color:var(--accent); }
  .composer button { background:var(--accent); color:#fff; border:0; border-radius:10px;
                     padding:0 22px; font-weight:600; cursor:pointer; }
  .composer button:disabled { opacity:.5; cursor:not-allowed; }
  .empty { color:var(--muted); font-size:14px; }
</style>
</head>
<body>
<header>
  <img src="/assets/logo.jpg" alt="StrikeRobot" onerror="this.style.display='none'">
  <span class="t">STRIKE ROBOT</span><span class="s">Demo1 · Agentic Nav + VLM</span>
  <span id="badge" class="badge off"><span class="dot"></span><span id="btxt">…</span></span>
</header>

<div class="app">
  <section class="card">
    <h2>Live Camera</h2>
    <div class="video"><img id="feed" src="/video_feed" alt="camera"></div>
  </section>
  <section class="card">
    <h2>Reasoning</h2>
    <div id="timeline" class="timeline"><div class="empty">Nhập lệnh để bắt đầu (vd: "đồ ăn làm nóng xong chưa?").</div></div>
    <div id="answer" class="answer">
      <div class="lab">Kết luận</div>
      <div id="answer-txt" class="txt"></div>
      <span id="answer-state" class="state"></span>
    </div>
    <form id="composer" class="composer">
      <input id="cmd" placeholder='vd: "đồ ăn làm nóng xong chưa?"' autocomplete="off">
      <button id="send" type="submit">Gửi</button>
    </form>
  </section>
</div>

<script>
const timeline = document.getElementById('timeline');
const form = document.getElementById('composer');
const cmd = document.getElementById('cmd');
const sendBtn = document.getElementById('send');
const badge = document.getElementById('badge');
const btxt = document.getElementById('btxt');
const answerBox = document.getElementById('answer');
const answerTxt = document.getElementById('answer-txt');
const answerState = document.getElementById('answer-state');

const ICONS = { plan:'🧠', nav:'🧭', perceive:'👁️' };
const TITLES = { plan:'Planner suy luận', nav:'Điều hướng', perceive:'Quan sát' };
const steps = {};

async function pollStatus() {
  try { const s = await (await fetch('/status')).json();
    badge.className = 'badge ' + (s.connected ? 'on' : 'off');
    btxt.textContent = s.connected ? 'CONNECTED' : (s.mock ? 'MOCK' : 'DISCONNECTED');
  } catch(e){ badge.className='badge off'; btxt.textContent='OFFLINE'; }
}
pollStatus(); setInterval(pollStatus, 3000);

function clearEmpty(){ const e=timeline.querySelector('.empty'); if(e) e.remove(); }
function ensureStep(id){
  if(steps[id]) return steps[id];
  clearEmpty();
  const el=document.createElement('div'); el.className='step running';
  el.innerHTML=`<div class="ico">${ICONS[id]||'•'}</div><div class="body">
    <div class="title">${TITLES[id]||id}</div><div class="detail"></div></div>`;
  timeline.appendChild(el); timeline.scrollTop=timeline.scrollHeight;
  steps[id]={el, detail:el.querySelector('.detail')}; return steps[id];
}
function reset(){ timeline.innerHTML=''; for(const k in steps) delete steps[k];
  answerBox.classList.remove('show'); }

let ws, busy=false;
function connectWS(){ ws=new WebSocket(`ws://${location.host}/ws`);
  ws.onclose=()=>setTimeout(connectWS,1500); ws.onmessage=onMsg; }
connectWS();

function onMsg(ev){
  const m=JSON.parse(ev.data);
  if(m.type==='step'){
    const s=ensureStep(m.id);
    s.el.className='step '+m.status;
    if(m.title) s.el.querySelector('.title').textContent=m.title;
    if(m.data && m.data.x!==undefined)
      s.detail.textContent=`→ ${m.data.room} (x=${m.data.x}, y=${m.data.y})`;
  } else if(m.type==='token'){
    const s=ensureStep(m.step); s.detail.textContent+=m.text;
    timeline.scrollTop=timeline.scrollHeight;
  } else if(m.type==='nav'){
    const s=ensureStep('nav');
    s.detail.textContent=`Còn ${m.distance_remaining.toFixed(2)} m...`;
  } else if(m.type==='image'){
    const s=ensureStep('perceive'); const img=new Image();
    img.src='data:image/jpeg;base64,'+m.data; s.detail.appendChild(img);
  } else if(m.type==='answer'){
    answerTxt.textContent=m.text;
    answerState.textContent=m.state; answerState.className='state '+m.state;
    answerBox.classList.add('show'); finish();
  } else if(m.type==='error'){
    const s=ensureStep('plan'); finish();
    const e=document.createElement('div'); e.className='step error';
    e.innerHTML=`<div class="ico">⚠️</div><div class="body"><div class="title">Lỗi</div>
      <div class="detail">${m.message}</div></div>`;
    timeline.appendChild(e); timeline.scrollTop=timeline.scrollHeight;
  }
}

form.addEventListener('submit',(e)=>{
  e.preventDefault();
  const c=cmd.value.trim();
  if(!c||busy||ws.readyState!==1) return;
  reset(); busy=true; sendBtn.disabled=true; cmd.value='';
  ws.send(JSON.stringify({command:c}));
});
function finish(){ busy=false; sendBtn.disabled=false; cmd.focus(); }
</script>
</body>
</html>
```

- [ ] **Step 2: Verify server tests vẫn pass với HTML thật**

Run: `cd /home/dsc-labs/ros2_ws/src && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vlm/webconsole/tests/test_demo1_server.py -q`
Expected: PASS (3 passed) — `/` vẫn chứa `<html`.

- [ ] **Step 3: Viết launcher**

Tạo `run_demo1.sh`:

```bash
#!/usr/bin/env bash
# StrikeRobot demo1 (python3.12 + ROS). DEMO_MOCK=1 để chạy không cần robot.
set -e
WS=/home/dsc-labs/ros2_ws
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true
export PYTHONPATH="$WS/src/vlm:$PYTHONPATH"
exec python3.12 -m webconsole.demo1.agent_server
```

- [ ] **Step 4: Verify boot mock mode (không cần ROS, không tải model)**

Run:
```bash
cd /home/dsc-labs/ros2_ws/src && chmod +x run_demo1.sh
DEMO_MOCK=1 VLM_SKIP_MODEL=1 python -m vlm.webconsole.demo1.agent_server &
SV=$!; sleep 2
curl -s http://127.0.0.1:8001/status; echo
curl -s http://127.0.0.1:8001/ | grep -o "STRIKE ROBOT" | head -1
kill $SV 2>/dev/null
```
Expected: `{"connected":false,"mock":true}` và in `STRIKE ROBOT`.
(Lưu ý: với `VLM_SKIP_MODEL=1` planner/perception sẽ lỗi khi gửi lệnh vì engine chưa load — chỉ kiểm tra UI/route ở bước này.)

- [ ] **Step 5: Commit**

```bash
git add vlm/webconsole/webui/demo1.html run_demo1.sh
git commit -m "feat(demo1): white timeline UI with StrikeRobot logo + launcher"
```

---

### Task 9: README demo1 + full test run

**Files:**
- Create: `vlm/webconsole/DEMO1.md`

**Interfaces:**
- Consumes: tất cả ở trên.
- Produces: tài liệu chạy.

- [ ] **Step 1: Viết README**

Tạo `vlm/webconsole/DEMO1.md`:

```markdown
# StrikeRobot — Demo1 (Agentic Nav + VLM)

Gõ câu lệnh → Qwen planner chọn phòng → Nav2 điều hướng (né vật cản) → VLM quan sát
microwave BẬT/TẮT → kết luận. UI timeline trắng, logo StrikeRobot.

## Chạy thật (cần robot + Nav2)

```bash
# Terminal 1
ros2 launch go2_robot_sdk robot.launch.py
# Terminal 2
ros2 launch go2_robot_sdk navigation.launch.py
# Terminal 3
cd /home/dsc-labs/ros2_ws/src
./run_demo1.sh        # http://localhost:8001
```

Toạ độ phòng nằm trong `vlm/webconsole/config/rooms.yaml` — sửa cho khớp map thật.

## Xem UI không cần robot

```bash
cd /home/dsc-labs/ros2_ws/src
DEMO_MOCK=1 ./run_demo1.sh           # nav giả + ảnh tĩnh, có VLM thật
DEMO_MOCK=1 VLM_SKIP_MODEL=1 python -m vlm.webconsole.demo1.agent_server  # chỉ xem giao diện
```

## Test

```bash
cd /home/dsc-labs/ros2_ws/src
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vlm/webconsole -v
```
```

- [ ] **Step 2: Chạy toàn bộ test**

Run: `cd /home/dsc-labs/ros2_ws/src && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vlm/webconsole -q`
Expected: PASS toàn bộ (webconsole cũ + demo1: ~31 tests).

- [ ] **Step 3: Commit**

```bash
git add vlm/webconsole/DEMO1.md
git commit -m "docs(demo1): usage README"
```

---

## Notes verify với robot thật

- Cần chạy `robot.launch.py` + `navigation.launch.py`, và đã localize (AMCL) trên map
  `cty.yaml`. Đặt `rooms.yaml` đúng toạ độ thực của từng phòng.
- Gửi lệnh "đồ ăn làm nóng xong chưa?" → timeline: Planner → đi Kitchen (Nav2 né vật
  cản) → quan sát microwave → kết luận BẬT/TẮT kèm ảnh bbox.
- GPU: chạy python3.12 (torch `+cu130`) để VLM dùng GPU.
