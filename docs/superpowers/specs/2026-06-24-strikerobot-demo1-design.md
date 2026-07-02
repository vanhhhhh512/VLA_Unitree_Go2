# StrikeRobot — Demo1 (Agentic Nav + VLM) — Design Spec

**Ngày:** 2026-06-24
**Trạng thái:** Đã duyệt thiết kế, chờ review spec

## 1. Mục tiêu

Kịch bản demo1: người dùng gõ một câu lệnh tự nhiên (vd *"đồ ăn đã được làm nóng
xong chưa?"*). Hệ thống:

1. **Planner** (Qwen text) suy luận ra phòng cần đến (microwave ở kitchen → đi kitchen),
   vật cần quan sát, và câu hỏi quan sát; hiển thị reasoning.
2. **Navigate** tới toạ độ phòng bằng Nav2 (`BasicNavigator` → `NavigateToPose`),
   vừa đi vừa **né vật cản** (local costmap của Nav2).
3. **Perceive**: tới nơi, chụp camera, dùng VLM **detect microwave** (bbox) và **phán
   đoán BẬT/TẮT**.
4. **Answer**: kết luận lại cho người dùng (vd "Lò đang BẬT — đồ ăn vẫn đang hâm nóng").

Giao diện reasoning theo dạng **timeline trắng, chuyên nghiệp**, có logo **StrikeRobot**.

### Quyết định đã chốt (brainstorming)
- **Runtime (A):** chạy tích hợp ROS2 (python3.12 + ROS sourced). Camera lấy từ topic
  `/camera/image_raw`; điều hướng bằng `nav2_simple_commander.BasicNavigator`; Nav2 lo
  né vật cản. Cần chạy kèm `robot.launch.py` + `navigation.launch.py`.
- **Planner (2A):** LLM planner dùng Qwen (text-only) để chọn phòng + vật + câu hỏi
  quan sát + reasoning. Tổng quát hoá.
- **Logo (3A):** file `vlm/webconsole/webui/assets/logo.jpg` (400×400, đã có).
- **UI (4A):** timeline các bước dọc (icon trạng thái) + camera live + thẻ kết luận lớn,
  nền trắng, accent đen theo logo.
- Toạ độ phòng **init random** trong vùng map, người dùng chỉnh sau trong `rooms.yaml`.

### Môi trường đã xác nhận
`rclpy`, `cv_bridge`, `geometry_msgs`, `nav_msgs`, `nav2_simple_commander` đều import
được dưới python3.12 khi đã `source /opt/ros/jazzy/setup.bash`.

## 2. Hiện trạng liên quan

- Nav2 đầy đủ qua `go2_robot_sdk/launch/navigation.launch.py` (AMCL + bt_navigator).
  Né vật cản: `config/nav2_params.yaml` có `local_costmap` voxel_layer + inflation.
  **Không có** file `navigation.py`; né vật cản là tính năng sẵn của Nav2.
- Camera publish ở `/camera/image_raw`. Map frame `map`, origin `[-2.118, -5.803]`,
  resolution `0.05`, ảnh `cty.pgm` 177×193 → vùng map ~ x∈[-2.118, 6.732],
  y∈[-5.803, 3.847].
- `vlm/webconsole/vlm_engine.py` (Qwen2.5-VL streaming) tái dùng được. Web console
  WebRTC cũ **giữ nguyên**; demo1 là app riêng.

## 3. Kiến trúc

Một tiến trình: FastAPI (asyncio) + một rclpy node (spin trong thread riêng).

```
Browser (UI timeline trắng + logo StrikeRobot)
   │ GET /             → webui/demo1.html
   │ GET /assets/logo.jpg
   │ GET /video_feed   → MJPEG từ frame mới nhất của /camera/image_raw
   │ WS  /ws           → gửi {command}; nhận step-events
   ▼
demo1/agent_server.py  (FastAPI + rclpy)
   ├─ planner.py          chọn phòng + vật + câu hỏi + reasoning (Qwen text)
   ├─ navigator.py        BasicNavigator.goToPose(); Nav2 né vật cản
   ├─ ros_frame_source.py subscribe /camera/image_raw → latest frame (BGR)
   ├─ perception.py       VLM detect + phán đoán BẬT/TẮT + vẽ bbox
   └─ agent.py            orchestrator: plan → nav → perceive → answer
```

## 4. Các module (tách nhỏ, có interface rõ, DI để test)

Đặt trong `vlm/webconsole/demo1/`. Tái dùng `vlm.webconsole.vlm_engine`.

1. **`rooms.py`** — load/đại diện phòng.
   - `Room` dataclass: `name: str`, `x: float`, `y: float`, `yaw: float`,
     `landmarks: list[str]`.
   - `load_rooms(path) -> dict[str, Room]`.
   - `random_rooms() -> dict[str, Room]` — sinh toạ độ random trong vùng map cho
     kitchen / living_room / bedroom (dùng khi chưa có file). Ghi ra `config/rooms.yaml`.

2. **`planner.py`** — `Planner`.
   - `plan(command: str, rooms: dict[str, Room]) -> Plan` với
     `Plan{room: str, target_object: str, observation_question: str, reasoning: str}`.
   - Dùng Qwen text: prompt gồm câu lệnh + danh sách phòng & landmarks, yêu cầu trả
     JSON `{"room","target_object","observation_question","reasoning"}`. Parser tách
     JSON từ output (regex khối `{...}`), validate `room` ∈ rooms; nếu sai → raise
     `PlanError`.
   - Hàm thuần `parse_plan(text, rooms) -> Plan` tách riêng để **unit-test**.

3. **`ros_frame_source.py`** — `RosFrameSource` (rclpy Node hoặc bọc 1 node).
   - subscribe `/camera/image_raw` (`sensor_msgs/Image`), `cv_bridge` → BGR numpy,
     lưu `latest`.
   - `get_latest_frame() -> np.ndarray | None`, `is_connected -> bool` (đã nhận ≥1 frame).

4. **`navigator.py`** — `Nav2Navigator` (bọc `BasicNavigator`).
   - `go_to(room: Room) -> Iterator[NavEvent]`: tạo `PoseStamped` (frame `map`,
     x/y/yaw→quaternion), `goToPose`, vòng lặp `isTaskComplete()` yield
     `NavEvent{kind:"feedback", distance_remaining: float}` rồi
     `NavEvent{kind:"done", success: bool}`. Lỗi/timeout → `kind:"error"`.
   - Định nghĩa interface tối thiểu để orchestrator dùng được `FakeNavigator` khi test.

5. **`perception.py`** — `Perception`.
   - `observe(frame_bgr, target_object, question) -> Iterator[str]` stream token VLM
     (tái dùng `VLMEngine.stream_infer`) với prompt yêu cầu: detect `target_object`
     (trả bbox `[ymin,xmin,ymax,xmax]`) và trả lời `question` (ON/OFF + lý do).
   - `finalize(full_text, frame_bgr) -> Result{state, annotated_jpeg_b64, summary}`:
     `parse_boxes` (tái dùng), vẽ bbox, suy ra `state ∈ {ON, OFF, UNKNOWN}` từ text
     (tìm từ khoá on/off/bật/tắt). Hàm `parse_state(text)` thuần để **unit-test**.

6. **`agent.py`** — orchestrator.
   - `async run(command) -> async-generator[Event]`: phát các Event:
     - `{type:"step", id, status:"running"|"done"|"error", title, data?}`
     - `{type:"token", step, text}` (stream reasoning planner & perceive)
     - `{type:"nav", distance_remaining}`
     - `{type:"image", data}` (ảnh bbox)
     - `{type:"answer", text, state}`
     - `{type:"error", message}`
   - Thứ tự bước: `plan` → `nav` → `perceive` → `answer`. Một bước lỗi → phát error,
     dừng.

7. **`agent_server.py`** — FastAPI + rclpy.
   - `create_agent_app(agent, frame_source) -> FastAPI` (DI để test bằng fake).
   - `GET /` → `webui/demo1.html`; `GET /assets/{f}` → file tĩnh; `GET /status`
     → `{connected, mock}`; `GET /video_feed` → MJPEG (`encode_frame_jpeg` tái dùng);
     `WS /ws` → nhận `{command}`, đẩy mọi Event từ `agent.run` ra client; khoá 1 lệnh/lần.
   - `main()`: `rclpy.init`, tạo `RosFrameSource` + spin thread, dựng `Planner`/
     `Nav2Navigator`/`Perception` (load `VLMEngine`), `Agent`, chạy uvicorn `0.0.0.0:8001`.
   - Cờ `DEMO_MOCK=1`: dùng `FakeNavigator` + ảnh tĩnh + không cần ROS (xem UI/luồng).

8. **`webui/demo1.html`** — UI timeline trắng (CSS/JS inline), logo từ `/assets/logo.jpg`.

## 5. Step-events (hợp đồng WS → UI)

| event | ý nghĩa |
|---|---|
| `{type:"step", id:"plan", status:"running", title}` | bắt đầu planner |
| `{type:"token", step:"plan", text}` | stream reasoning planner |
| `{type:"step", id:"plan", status:"done", data:{room, target_object}}` | xong planner |
| `{type:"step", id:"nav", status:"running", title, data:{room,x,y}}` | bắt đầu đi |
| `{type:"nav", distance_remaining}` | cập nhật khoảng cách |
| `{type:"step", id:"nav", status:"done"}` | đã tới |
| `{type:"step", id:"perceive", status:"running", title}` | bắt đầu quan sát |
| `{type:"token", step:"perceive", text}` | stream reasoning VLM |
| `{type:"image", data}` | ảnh bbox base64 |
| `{type:"step", id:"perceive", status:"done"}` | xong quan sát |
| `{type:"answer", text, state}` | kết luận cuối |
| `{type:"error", message}` | lỗi bất kỳ bước nào |

## 6. rooms.yaml (ví dụ init random, người dùng chỉnh)

```yaml
# frame: map. x,y theo mét; yaw theo radian.
kitchen:     {x: 3.10, y: -1.20, yaw: 0.0,  landmarks: [microwave, fridge, stove, sink]}
living_room: {x: 0.50, y:  1.80, yaw: 1.57, landmarks: [sofa, tv, coffee table]}
bedroom:     {x: 4.80, y:  2.50, yaw: 3.14, landmarks: [bed, wardrobe, lamp]}
```

## 7. Xử lý lỗi

| Tình huống | Hành vi |
|---|---|
| Nav2 chưa active / goal timeout / robot kẹt | step `nav` → error, dừng pipeline, báo UI |
| Không có frame camera khi perceive | step `perceive` → error |
| Planner không ra phòng hợp lệ | error "không xác định được phòng", gợi ý phòng có sẵn |
| VLM không chắc BẬT/TẮT | `state=UNKNOWN`, answer nêu rõ "không chắc" + lý do |
| Gửi lệnh khi đang chạy | khoá 1 lệnh/lần, trả `{type:"error", message:"đang xử lý"}` |

## 8. Kiểm thử

- **Unit:** `load_rooms`/`random_rooms` (toạ độ trong vùng map); `parse_plan` (JSON
  hợp lệ/sai/room không tồn tại); `parse_state` (ON/OFF/bật/tắt/không rõ); schema event.
- **Orchestrator:** `Agent.run` với Fake planner/navigator/perception + frame giả →
  assert đúng thứ tự event `plan→nav→perceive→answer` và có `answer`.
- **Server:** `create_agent_app` với fakes (TestClient): `/` trả HTML, `/status`,
  `/video_feed` content-type, `/ws` chạy 1 lệnh ra đủ chuỗi event tới `answer`.
- **Mock mode:** `DEMO_MOCK=1` chạy full UI không cần ROS/robot.

Lưu ý: test chạy bằng conda (như package hiện tại) với Fake, không cần ROS; module ROS
chỉ import bên trong `Nav2Navigator`/`RosFrameSource` (lazy) để test không cần rclpy.

## 9. Cách chạy (dự kiến)

```bash
# Terminal 1: robot + Nav2
ros2 launch go2_robot_sdk robot.launch.py
ros2 launch go2_robot_sdk navigation.launch.py

# Terminal 2: demo1 (python3.12 + ROS)
cd /home/dsc-labs/ros2_ws/src
./run_demo1.sh            # mở http://localhost:8001
# Xem UI không cần robot:
DEMO_MOCK=1 ./run_demo1.sh
```

## 10. Ngoài phạm vi (YAGNI)

- Đa robot, nhiều kịch bản phức tạp, chỉnh map trên web, điều khiển tay.
- Tự động đặt lại toạ độ phòng (người dùng sửa `rooms.yaml`).
- Giữ nguyên web console WebRTC cũ.
