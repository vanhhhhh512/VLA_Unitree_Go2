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
