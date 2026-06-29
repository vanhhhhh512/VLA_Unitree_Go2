# RUN — Unitree Go2 · Nav + VLM (StrikeRobot Demo 1)

Robot Go2 IP: **192.168.1.2** (cùng mạng với máy).

## Terminal 1 — Kết nối robot + Nav2
```bash
pkill -f go2_driver ; pkill -f nav2 ; pkill -f controller_server ; pkill -f planner_server ; pkill -f bt_navigator ; sleep 2
export ROBOT_IP=192.168.1.2
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch go2_robot_sdk navigation.launch.py
```

## Terminal 2 — GUI nhập lệnh (http://localhost:8001)
```bash
cd ~/ros2_ws/src
./run_demo1.sh
```
Mở trình duyệt: **http://localhost:8001**

Nhập lệnh trong ô GUI:
- Hỏi-đáp + điều hướng: `Is the bottle on the microwave?`
- Lệnh chuyển động tay: `move forward 75 cm` · `move backward 30 cm` · `turn left 90 deg` · `turn right 45 deg`
- Bấm **Stop** để hủy khi đang chạy.

GUI biến thể (tùy chọn, đặt biến trước lệnh):
```bash
DEMO_MOCK=1 ./run_demo1.sh        # xem GUI không cần robot/ROS
USE_YOLO=0  ./run_demo1.sh        # tắt YOLO, dùng Qwen grounding
```

## Terminal 3 — Điều khiển tay bằng bàn phím (tùy chọn)
```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -r /cmd_vel:=/cmd_vel_joy -p stamped:=true -p frame_id:=base_link
```
