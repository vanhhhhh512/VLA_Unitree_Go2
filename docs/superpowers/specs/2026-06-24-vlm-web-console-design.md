# VLM Web Console cho Unitree Go2 — Design Spec

**Ngày:** 2026-06-24
**Trạng thái:** Đã duyệt thiết kế, chờ review spec

## 1. Mục tiêu

Xây một **web GUI chạy localhost** để người dùng nhập lệnh tự do, hệ thống lấy ảnh
camera từ robot Unitree Go2 qua WebRTC, đưa vào model Vision-Language (Qwen2.5-VL-3B-Instruct),
và hiển thị **reasoning stream theo thời gian thực** kèm **ảnh đã chú thích bounding box**.
Giao diện theo phong cách **glassmorphism**, đẹp và khác biệt so với ChatGPT.

### Quyết định đã chốt (brainstorming)
- **Phạm vi (1B):** Hỏi-đáp / suy luận tự do trên ảnh. Khi lệnh là tìm/định vị vật thể
  thì vẽ thêm bounding box lên ảnh đính kèm. **Không** điều khiển chuyển động robot.
- **Ảnh hiển thị (2A):** Video **live** luôn chạy; khi gửi lệnh thì "chụp" frame tại
  thời điểm đó đưa vào VLM.
- **Nguồn ảnh (3A):** Kết nối **WebRTC trực tiếp** tới Go2 (biến môi trường `ROBOT_IP`),
  tái dùng `Go2Connection` có sẵn.
- **Thẩm mỹ (4B):** Glassmorphism — gradient tối, panel kính mờ, bo góc lớn, bóng mềm,
  chuyển động mượt.
- **Reasoning (5A):** Stream token-by-token (typewriter) qua WebSocket.
- **Video transport:** MJPEG (đơn giản, ổn định, không cần WebRTC trong trình duyệt).

## 2. Hiện trạng code (folder `vlm`)

- [`run_vlm_webrtc.py`](../../../run_vlm_webrtc.py): kết nối Go2 qua WebRTC, nhận video,
  chạy Qwen2.5-VL, vẽ bbox, hiển thị bằng cửa sổ OpenCV. **Chỉ làm object detection**
  (`Detect {prompt}`), không stream reasoning, không có web UI.
- [`vlm/vlm/vlm_detect_node.py`](../../../vlm/vlm/vlm_detect_node.py): bản ROS2 node, cùng
  logic detection, publish `/vlm/debug_image`.

Hệ thống mới sẽ **tách logic WebRTC và VLM ra module dùng lại được**, không sửa hành vi
hai file trên.

## 3. Kiến trúc

Một tiến trình FastAPI duy nhất:

```
Browser (glassmorphism UI)
   │  GET /            → trang HTML/CSS/JS
   │  GET /video_feed  → MJPEG live stream (frame mới nhất từ Go2)
   │  WS  /ws          → gửi lệnh; nhận token reasoning + ảnh bbox
   ▼
FastAPI server (asyncio)
   ├─ background task: WebRTC → Go2 → cập nhật latest_frame liên tục
   ├─ vlm_engine: Qwen2.5-VL, stream token + parse bbox
   └─ on command: snapshot latest_frame → infer → stream chữ → vẽ bbox → gửi ảnh
```

## 4. Các module

Đặt code mới trong `vlm/webconsole/` (package con), entry point chạy bằng `python -m`.

1. **`webrtc_source.py`** — quản lý kết nối Go2.
   - Tái dùng `Go2Connection`. Chạy nền trong asyncio loop của server.
   - Lưu `latest_frame` (numpy BGR) mỗi khi nhận frame WebRTC.
   - API: `start()`, `get_latest_frame() -> np.ndarray | None`, `is_connected -> bool`.
   - **Mock mode:** nếu không có `ROBOT_IP`, trả về 1 ảnh test tĩnh và `is_connected=False`.

2. **`vlm_engine.py`** — bao bọc Qwen2.5-VL.
   - Load model + processor 1 lần lúc khởi động (GPU nếu có, fallback CPU + cảnh báo).
   - `stream_infer(frame_bgr, prompt) -> generator[str]`: yield từng token text bằng
     `TextIteratorStreamer` (chạy `model.generate` trong thread riêng).
   - Sau khi xong: ghép full text, `parse_boxes(text)` trả list `[(x1,y1,x2,y2,label)]`
     đã scale về kích thước ảnh thực; `draw_boxes(frame, boxes)` trả ảnh đã vẽ.
   - Một hàm `parse_boxes` thuần (regex `\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]`, thứ tự
     `ymin,xmin,ymax,xmax`, chia 1000 nhân kích thước) — **unit-testable độc lập**.

3. **`server.py`** — FastAPI app.
   - `GET /` → trả `webui/index.html`.
   - `GET /video_feed` → `StreamingResponse` MJPEG: vòng lặp encode `latest_frame` JPEG
     ~15–20 fps; nếu không có frame thì khung chờ.
   - `WS /ws` → nhận `{prompt}`; nếu đang xử lý lệnh khác thì trả `{type:"busy"}`;
     ngược lại: snapshot frame → `stream_infer` → gửi `{type:"token", text}` mỗi token →
     khi xong vẽ bbox → gửi `{type:"image", data: <base64 jpeg>}` rồi `{type:"done"}`.
   - Lỗi runtime → `{type:"error", message}`.
   - Khởi tạo `webrtc_source` và `vlm_engine` ở startup event.

4. **`webui/index.html`** — UI 1 file (CSS + JS inline, không cần build step).

## 5. Luồng dữ liệu khi gửi lệnh

1. JS gửi `{prompt}` qua WebSocket.
2. Server snapshot `latest_frame` ngay lúc đó.
3. `stream_infer` yield token → server đẩy `{type:"token"}` → UI gõ chữ typewriter.
4. Khi xong: `parse_boxes` + `draw_boxes` lên frame đã snapshot → encode base64 →
   gửi `{type:"image"}` → hiển thị trong bong bóng reasoning → `{type:"done"}`.

## 6. Giao diện (Glassmorphism)

- Nền gradient tối; panel kính mờ (`backdrop-filter: blur`), bo góc lớn, bóng mềm.
- **Cột trái:** panel video live (`<img src="/video_feed">`) + badge trạng thái kết nối
  Go2 (xanh = connected, đỏ = mock/mất kết nối).
- **Cột phải:** khung chat — bong bóng lệnh người dùng; bong bóng reasoning stream dần;
  ảnh bbox đính kèm khi có.
- Ô nhập lệnh cố định dưới cùng, animation mượt khi gửi.
- Responsive gọn cho 1 màn hình desktop.

## 7. Xử lý lỗi

| Tình huống | Hành vi |
|---|---|
| Chưa set `ROBOT_IP` | Mock mode: video_feed hiện ảnh test, vẫn infer được; badge đỏ |
| Mất kết nối WebRTC | Badge đỏ; frame stale; lệnh vẫn chạy trên frame cuối |
| Gửi lệnh khi đang infer | Khóa 1 lệnh/lần; trả `{type:"busy"}`; UI báo "đang xử lý" |
| Không có GPU | Fallback CPU + cảnh báo chậm trong log |
| Lỗi infer | `{type:"error"}`; UI hiện thông báo lỗi trong chat |

## 8. Kiểm thử

- **Unit:** `parse_boxes` (định dạng đúng/sai, scale toạ độ, nhiều box, không có box).
- **Smoke:** server khởi động; `GET /` trả HTML; `/video_feed` trả `multipart/x-mixed-replace`.
- **Mock mode:** chạy full luồng chat (gửi lệnh → token → ảnh) không cần robot thật.

## 9. Cách chạy (dự kiến)

```bash
export ROBOT_IP=192.168.123.161   # bỏ qua để chạy mock mode
python -m vlm.webconsole.server   # mở http://localhost:8000
```

## 10. Ngoài phạm vi (YAGNI)

- Điều khiển chuyển động robot.
- Nhiều robot / nhiều phiên chat đồng thời.
- Lưu lịch sử chat, xác thực người dùng.
- Tích hợp ROS2 (dùng WebRTC trực tiếp).
