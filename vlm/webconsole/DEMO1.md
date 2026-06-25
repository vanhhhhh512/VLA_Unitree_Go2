# StrikeRobot — Demo1 (Agentic Nav + VLM)

Gõ câu lệnh → Qwen planner chọn phòng → Nav2 điều hướng (né vật cản) → **YOLO detect +
khoanh vùng vật thể** + **Qwen trả lời tự nhiên** → kết luận. UI timeline trắng, logo StrikeRobot.

## Phát hiện vật thể (YOLO + Qwen)

- **YOLO11n** (ultralytics, model `demo1/models/yolo11n.pt`, 80 lớp COCO) lo **detect + bounding box**
  chính xác (frame-space, không cần scale). Kết quả YOLO được **mớm vào prompt của Qwen** để câu
  trả lời chính xác hơn.
- **Qwen2.5-VL** lo **trả lời câu hỏi** bằng ngôn ngữ tự nhiên (mô tả + kết luận YES/NO).
- Tắt YOLO (quay về Qwen grounding): đặt `USE_YOLO=0`.
- ultralytics cần ở python3.12 (runtime demo1):
  `python3.12 -m pip install --user --break-system-packages --no-deps ultralytics ultralytics-thop`
  (`--no-deps` để không đụng torch `+cu130`/numpy đang chạy).

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
