# RUN — Unitree Go2 · Nav + VLM (StrikeRobot Demo 1)

Hướng dẫn chạy đầy đủ các node. Mọi lệnh chạy trên máy này.

- Workspace: `~/ros2_ws` (tức `/home/dsc-labs/ros2_ws`)
- Robot Go2: **IP `192.168.1.2`** (cùng mạng `192.168.1.x` với máy). Kiểm tra: `ping 192.168.1.2`
- Runtime: **python3.12 + ROS Jazzy** cho robot/Nav2; demo1 chạy bằng python3.12 (GPU torch `+cu130`).

---

## 0. Một lần đầu (nếu chưa có)

```bash
# Build workspace (nếu chưa build)
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install

# Thư viện cho demo1 (python3.12) — YOLO cài --no-deps để không đụng torch/numpy
python3.12 -m pip install --user --break-system-packages fastapi "uvicorn[standard]"
python3.12 -m pip install --user --break-system-packages --no-deps ultralytics ultralytics-thop
```

---

## 1. Terminal 1 — Robot + Nav2 (bắt buộc)

`navigation.launch.py` đã gồm: driver kết nối Go2 (WebRTC) + camera/odom/scan + AMCL + Nav2 + RViz.

```bash
# dọn node cũ còn sót (tránh kẹt lifecycle / giữ WebRTC)
pkill -f go2_driver ; pkill -f nav2 ; pkill -f controller_server ; pkill -f planner_server ; pkill -f bt_navigator ; sleep 2

export ROBOT_IP=192.168.1.2
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch go2_robot_sdk navigation.launch.py
```

Sau khi RViz mở: nếu robot chưa đúng vị trí trên map → bấm **"2D Pose Estimate"** đặt đúng chỗ robot đứng.

---

## 2. Terminal 2 — GUI nhập lệnh navigation (StrikeRobot Demo 1)

```bash
cd ~/ros2_ws/src
./run_demo1.sh
```
→ mở **http://localhost:8001**. Gõ lệnh tự nhiên, ví dụ:
- `Is the bottle on the microwave?`
- `Is the food done heating yet?`

Robot tự đi tới phòng (Nav2 né vật cản) → YOLO khoanh vật + Qwen trả lời. Bấm **Stop** để hủy khi chờ quá lâu.

Biến môi trường tùy chọn (đặt trước `./run_demo1.sh`):
- `USE_YOLO=0` → tắt YOLO, dùng Qwen grounding.
- `DEMO_MOCK=1` → không cần robot/ROS (nav giả + ảnh tĩnh).
- `VLM_SKIP_MODEL=1` → không tải model (chỉ xem giao diện).

---

## 3. Terminal 3 — Điều khiển tay bằng bàn phím (tùy chọn)

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -r /cmd_vel:=/cmd_vel_joy -p stamped:=true -p frame_id:=base_link
```
Phím: `i`=tiến `,`=lùi `j`/`l`=xoay `k`=dừng · `q/z`=tốc độ. Lệnh tay (priority 10) đè Nav2.

---

## 4. Kiểm tra nhanh (Terminal phụ)

```bash
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash

# Nav2 đã sẵn sàng?
ros2 service call /lifecycle_manager_navigation/is_active std_srvs/srv/Trigger   # success=True

# Robot có stream dữ liệu?
ros2 topic hz /camera/image_raw      # ~5-15 hz
ros2 topic hz /odom
ros2 topic hz /scan
```

---

## 5. Map / toạ độ phòng

```bash
# Xem map + click lấy toạ độ (mét) cho rooms.yaml
python3 ~/view_map.py

# Sửa toạ độ/hướng các phòng
nano ~/ros2_ws/src/vlm/webconsole/config/rooms.yaml
```
Mỗi phòng: `x`, `y`, `yaw` (radian). `yaw`: 0=Đông, 1.57=Bắc, 3.14=Tây, -1.57=Nam.

---

## 6. Test (phát triển)

```bash
cd ~/ros2_ws/src
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vlm/webconsole -q
```

---

## Sự cố thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| Web quay / "No Image" / không có odom | Robot chưa kết nối → kiểm tra `ROBOT_IP=192.168.1.2`, `ping 192.168.1.2`, máy cùng mạng |
| `is_active=False`, không có `/plan`, không nhận lệnh | Node Nav2 cũ còn sót → chạy lại `pkill ...` ở bước 1 rồi launch lại |
| "Executor is already spinning" | (đã fix trong code) — chạy lại Terminal 2 |
| Robot không quay đúng hướng ở đích | Kiểm tra đã "2D Pose Estimate" (AMCL localize) chuẩn chưa |
| Kẹt "X m remaining" mãi | `xy_goal_tolerance` đã nới 0.4; hoặc bấm **Stop** trên web |
| VLM chạy CPU (chậm) | Phải chạy demo bằng python3.12 (run_demo1.sh đã đúng), không phải conda |

---

## Tóm tắt tối thiểu (đã quen)

```bash
# T1
export ROBOT_IP=192.168.1.2
source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch go2_robot_sdk navigation.launch.py

# T2
cd ~/ros2_ws/src && ./run_demo1.sh        # http://localhost:8001
```
