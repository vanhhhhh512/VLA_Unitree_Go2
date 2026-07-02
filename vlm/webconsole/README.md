# VLM Web Console — Unitree Go2

Web GUI localhost: nhập lệnh tự do → lấy ảnh live từ Go2 (WebRTC) → Qwen2.5-VL
stream reasoning theo thời gian thực + vẽ bounding box khi lệnh là tìm/định vị vật thể.
Giao diện glassmorphism.

## Cài đặt

```bash
pip install -r ../../requirements.txt
```

(Lần đầu chạy thật sẽ tự tải model `Qwen/Qwen2.5-VL-3B-Instruct` ~vài GB; cần GPU CUDA
để chạy mượt.)

## Chạy với robot thật (real mode)

Real mode cần `go2_robot_sdk` → phải dùng **python3.12 + ROS đã source** (conda
python3.13 không chạy được phần WebRTC vì `rclpy` build cho 3.12). Có sẵn script:

```bash
cd /home/dsc-labs/ros2_vlm/src
./run_vlm_console.sh 192.168.123.161     # hoặc: ROBOT_IP=... ./run_vlm_console.sh
```

Script tự: source ROS (`/opt/ros/jazzy` + `~/ros2_vlm/install`), set `PYTHONPATH` để
import được `webconsole`, rồi chạy `python3.12 -m webconsole.server`.

Mở http://localhost:8000

- **Panel trái:** video live từ Go2 + badge trạng thái kết nối (xanh = CONNECTED,
  đỏ = MOCK MODE / DISCONNECTED).
- **Panel phải:** gõ lệnh (vd "tìm điện thoại", "mô tả cảnh", "có người không?"),
  xem reasoning stream dần và ảnh có bounding box xuất hiện trong chat.

> ⚠️ **GPU:** máy hiện không có CUDA → model chạy trên CPU sẽ **rất chậm**
> (có thể vài chục giây–vài phút mỗi câu). Nên chạy trên máy có GPU CUDA để mượt.

## Mock mode (xem giao diện, không cần robot)

Không truyền `ROBOT_IP` → server chạy với 1 ảnh test. Chạy nhanh bằng conda:

```bash
cd /home/dsc-labs/ros2_vlm/src
python -m vlm.webconsole.server
```

(Hoặc `./run_vlm_console.sh` không kèm IP.) Mock mode dùng để kiểm tra giao diện và
luồng web. Lưu ý reasoning thật vẫn cần model load (lần đầu tải ~vài GB).

### Chỉ xem GUI, không tải model

Để mở giao diện tức thì mà không tải/nạp model 3B:

```bash
cd /home/dsc-labs/ros2_vlm/src
VLM_SKIP_MODEL=1 python -m vlm.webconsole.server
```

UI và video chạy bình thường; khi gửi lệnh sẽ báo lỗi cho tới khi bỏ cờ này.

## Test

ROS workspace có sẵn vài pytest plugin (`launch_testing`) không tương thích pytest mới,
và `setup.cfg` ở gốc bật coverage. Chạy test của package này bằng:

```bash
cd /home/dsc-labs/ros2_vlm/src
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vlm/webconsole -v
```

(`vlm/webconsole/pytest.ini` đã set `--import-mode=importlib` để import package đúng.)

## Cấu trúc

| File | Trách nhiệm |
|---|---|
| `vlm_engine.py` | Helper bbox/JPEG/messages + `VLMEngine` (Qwen, streaming token) |
| `frame_source.py` | Kết nối Go2 WebRTC / mock mode, giữ frame mới nhất |
| `server.py` | FastAPI: `/`, `/status`, `/video_feed` (MJPEG), `/ws` (chat stream) |
| `webui/index.html` | UI glassmorphism (CSS/JS inline) |
