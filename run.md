# RUN — Unitree Go2 · Nav + VLM (StrikeRobot Demo 1)

**Sơ đồ máy:**
- Máy chạy (GPU RTX 5070 Ti): **`dsc-labs` · LAN IP `192.168.1.29`** — chạy ROS + web ở đây.
- Máy của bạn: **`admin1`** — SSH sang 5070, mở trình duyệt xem web.
- Robot Go2: **`192.168.1.7`** (cùng mạng LAN).

> Mọi lệnh dưới đây chạy **trên 5070 (dsc-labs)** qua SSH. RViz/Foxglove tắt vì SSH headless.

---

## Bước 0 — Từ máy admin1, SSH sang 5070 (kèm forward cổng web 8001)
```bash
ssh -L 8001:localhost:8001 dsc-labs@192.168.1.29
```
Mở **2 terminal SSH** như vậy (hoặc dùng `tmux` cho tiện): 1 cho robot, 1 cho web.
Chỉ cần forward 8001 ở **một** lần SSH là đủ để xem web bằng `localhost:8001` trên admin1.

---

## Terminal 1 (SSH) — Kết nối robot + Nav2
```bash
pkill -f go2_driver ; pkill -f nav2 ; sleep 2
export ROBOT_IP=192.168.1.7
source /opt/ros/jazzy/setup.bash
source ~/ros2_vlm/install/setup.bash
ros2 launch go2_robot_sdk robot.launch.py nav2:=false slam:=false rviz2:=false foxglove:=false
```
⚠️ **Bắt buộc `export ROBOT_IP=192.168.1.7`** — thiếu thì driver connect IP rỗng → không camera/odom.
✅ **Không còn Nav2/SLAM** (`nav2:=false slam:=false`) — VLM tự lái, không cần Nav2. Launch này cấp
đủ: camera `/camera/image_raw`, `/odom`, `/scan`, `twist_mux` (`/cmd_vel_joy`). Token mặc định rỗng.

Kiểm tra robot đã lên đủ (terminal khác, nhớ source ROS):
```bash
ros2 topic hz /camera/image_raw    # ~15-30 Hz  -> camera OK (web sẽ hiện hình)
ros2 topic hz /odom                # có dữ liệu -> VLM đi đúng quãng đường
ros2 topic hz /scan                # có dữ liệu -> né vật cản hoạt động
```

---

## Terminal 2 (SSH) — Web GUI nhập lệnh
```bash
cd ~/ros2_vlm/src
./run_demo1.sh
```
Đợi tới dòng `Uvicorn running on http://0.0.0.0:8001` (~18s, model load CUDA).

Trên trình duyệt **máy admin1**, mở:
- `http://localhost:8001` (nếu đã SSH kèm `-L 8001:...` ở Bước 0), **hoặc**
- `http://192.168.1.29:8001` (vào thẳng IP 5070, cùng LAN).

Nhập lệnh trong ô GUI:
- **VLM nav (mặc định)**: gõ mục tiêu ngôn ngữ tự nhiên, vd `đi tới bình nước rồi rẽ phải, gặp ghế thì dừng`.
  VLM nhìn cam → tự suy luận → sinh JSON `{action,value,unit}` → đi từng bước tới khi xong. **Không qua Nav2.**
- **Lệnh tay** (đi thẳng, không qua VLM): `move forward 75 cm` · `turn left 90 deg` · `turn right 45 deg`.
- Bấm **Stop** / **E-Stop** để dừng. Xem `vlm/webconsole/DEMO1.md`.

Chọn não Qwen mạnh hơn cho VLM nav (tùy chọn, đặt trước `./run_demo1.sh`):
```bash
VLA_BRAIN=api VLA_API_URL=<endpoint> VLA_API_KEY=<key> VLA_MODEL=qwen2.5-vl-7b-instruct ./run_demo1.sh
```

---

## Xem web không cần robot (chỉ kiểm tra giao diện)
```bash
cd ~/ros2_vlm/src
DEMO_MOCK=1 ./run_demo1.sh        # nav giả + ảnh tĩnh, vẫn có VLM thật
```

## Terminal 3 (tùy chọn) — Lái tay bằng bàn phím
```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_vlm/install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -r /cmd_vel:=/cmd_vel_joy -p stamped:=true -p frame_id:=base_link
```

---

## Web không hiện camera? — checklist
1. `ros2 topic hz /camera/image_raw` có ra Hz không?
   - **Không** → driver chưa chạy / thiếu `ROBOT_IP` → xem lại Terminal 1.
   - **Có** mà web vẫn trống → reload trang, hoặc restart `./run_demo1.sh`.
2. `ros2 node list` phải thấy `go2_driver_node`. Chỉ thấy `/strikerobot_demo1` = mới có web, chưa có robot.
3. `ping 192.168.1.7` không thông → kiểm tra mạng/robot.

## 🛠️ Debug phần VLM — lỗi gì sửa file nào
Bảng đầy đủ (triệu chứng → file/tham số, bản đồ file, tất cả env) ở:
**`vlm/webconsole/DEMO1.md`** → mục **"🛠️ DEBUG — lỗi gì sửa ở file nào (phần VLM)"**.

Tóm tắt nhanh (sửa `.py` xong chỉ cần `pkill -f agent_server` rồi `./run_demo1.sh`, KHÔNG cần build):
- Suy luận/căn giữa/đi-dừng sai → `demo1/navloop.py` (`_SYSTEM` prompt, `_stop_override`).
- Box YOLO nhiễu → `demo1/annotator.py` (`VLA_YOLO_MIN_CONF`, `VLA_BOX_SMOOTH`).
- Tốc độ đi/xoay, vật cản → `demo1/motion.py` (`MOTION_*`).
- Đổi model VLM → `vlm_engine.py` hoặc env `VLA_BRAIN=api`.
- Web mất cam/disconnect → KHÔNG phải VLM, là driver/robot (token, `ROBOT_IP`, restart driver).

pkill -f agent_server ; pkill -f go2_driver ; pkill -f "ros2 launch" ; sleep 3
export ROBOT_IP=192.168.1.7
source /opt/ros/jazzy/setup.bash && source ~/ros2_vlm/install/setup.bash
ros2 launch go2_robot_sdk robot.launch.py nav2:=false slam:=false rviz2:=false foxglove:=false &
sleep 10
cd ~/ros2_vlm/src
MOTION_FRONT_STOP=0.22 VLA_STOP_BOTTOM_PX=15 VLA_OVERRIDE_STEP_M=0.10 MOTION_LIN_SPEED=0.12 ./run_demo1.sh

pkill -f agent_server ; sleep 1
cd ~/ros2_vlm/src
VLA_CONTROL=servo VLA_YOLO_MIN_CONF=0.6 ./run_demo1.sh

pkill -9 -f "go2_driver|robot.launch|agent_server" 2>/dev/null ; sleep 3
export ROBOT_IP=192.168.1.7
source /opt/ros/jazzy/setup.bash
source ~/ros2_vlm/install/setup.bash
DECODE_LIDAR=false ros2 launch go2_robot_sdk robot.launch.py nav2:=false slam:=false rviz2:=false foxglove:=false joystick:=false

pkill -f agent_server ; sleep 1
cd ~/ros2_vlm/src
VLA_SERVO_HZ=20 VLA_SERVO_WZ_MAX=0.2 VLA_SERVO_WZ_MIN=0.15 VLA_SERVO_KP=3 \
VLA_STOP_BOTTOM_PX=0 VLA_FINAL_PUSH_M=0.4 \
VLA_YOLO_MIN_CONF=0.2 VLA_YOLO_IMGSZ=1280 \
./run_demo1.sh
