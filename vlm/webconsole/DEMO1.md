# StrikeRobot — Demo1 (VLM Nav, không Nav2)

Gõ MỘT câu lệnh ngôn ngữ tự nhiên → **VLM (Qwen2.5-VL) tự suy luận từng bước** từ ảnh camera
→ sinh JSON lệnh di chuyển → tự chấp hành → lặp tới khi xong. **Không dùng Nav2.** UI timeline
trắng, logo StrikeRobot.

## VLM nav — vòng lặp kiểu NaVILA (mặc định cho mọi lệnh)

Gõ mục tiêu, vd `đi tới bình nước rồi rẽ phải, gặp ghế thì dừng`. Vòng lặp:

```
frame cam → prompt → Qwen trả JSON {reasoning, action, value, unit, obstacles_detected,
is_finished} → MotionController (vòng kín /odom + né vật cản /scan + estop) → frame mới → lặp…
```

Tới khi VLM trả `is_finished:true` (đã tới đích) hoặc bấm **Stop/E-Stop**. Mỗi bước ghi ra
`vla_logs/nav_<time>.jsonl` (đổi chỗ bằng `VLA_LOG_DIR`).

### Hai chế độ điều khiển (`VLA_CONTROL`)

**`vlm` (mặc định) — VLM tự suy luận theo SỐ LIỆU bơm vào (State Injection + CoT).**
Mỗi bước, code đo sẵn `lệch ?°`, `đáy cách mép ?px`, `vật cản lidar ?m` rồi **bơm vào prompt**
(VLM khỏi đoán bằng mắt). VLM trả JSON có **chuỗi suy luận** (điền trước action):
```json
{"obstacle_check":"...","bottom_touched":"...","is_centered":"...",
 "reasoning":"...","action":"turn_right","value":15,"unit":"degrees","is_finished":false}
```
`_SYSTEM` ép thứ tự ưu tiên (an toàn → dừng đúng đích → căn giữa ≤5° → tiến) + 3 few-shot.
**THUẦN VLM: code KHÔNG đè quyết định** — mọi logic ưu tiên nằm trong JSON/prompt; code chỉ
bơm số liệu + chấp hành. (An toàn lúc ĐANG đi vẫn còn ở tầng `MotionController`: tự né `/scan`
+ E-Stop; nếu cần tất định tuyệt đối thì dùng mode `servo`.)

**`servo` (CHỌN: `VLA_CONTROL=servo`) — VISUAL SERVO LIÊN TỤC (vòng kín @ `VLA_SERVO_HZ`=4Hz).**
Khi YOLO thấy mục tiêu, bỏ qua VLM, **bơm vận tốc trực tiếp**:
1. **ƯU TIÊN 1 — căn giữa:** |lệch| > `VLA_CENTER_TOL_DEG` (5°) → bơm ωz xoay nhẹ (vx=0), tới khi vào giữa.
2. **ƯU TIÊN 2 — tiến:** đã giữa → bơm vx (`VLA_SERVO_VX`) tiến thẳng (tự chậm lại khi sắp tới).
3. **DỪNG NGAY:** đáy box chạm mép dưới (gap ≤ `VLA_STOP_BOTTOM_PX`) → publish 0 trong cùng chu kỳ (≤0.25s).
Chọn vật **TO NHẤT** khi trùng tên (`VLA_PICK=area`). Mất dấu → nhường VLM quét tìm.

Cả hai: khi **CHƯA thấy** mục tiêu → để VLM nhìn ảnh tự quét tìm. (JSON CoT chỉ dùng ở mode `vlm`;
ở `servo` quyết định thuần hình học, không qua JSON.)

**Định tuyến lệnh** ([demo1/agent_server.py](demo1/agent_server.py)):
- `move/turn ...` → MotionController (lệnh tay, đi thẳng).
- `ngồi/chào/nhảy...` → ActionController (sport API).
- còn lại (ngôn ngữ tự nhiên) → **VLA loop** ([demo1/navloop.py](demo1/navloop.py)) — VLM lái.

**Chọn não Qwen** (env, đặt trước lệnh chạy):
- `VLA_BRAIN=local` (mặc định) — dùng `VLMEngine` (Qwen2.5-VL-3B) đã load.
- `VLA_BRAIN=api VLA_API_URL=<url> VLA_API_KEY=<key> VLA_MODEL=<tên>` — gọi endpoint
  OpenAI-compatible (DashScope / vLLM / Ollama / sglang), ép `response_format json_object`,
  dùng được model lớn hơn (7B/72B) khi 3B suy luận yếu.
- `VLA_MAX_STEPS=20` — số bước tối đa trước khi dừng an toàn.

Logic vòng lặp + chấp hành nằm ở Python nên **giữ lớp an toàn**: đi đúng quãng đường bằng
`/odom`, tự dừng trước vật cản bằng `/scan`, có E-Stop. Gặp vật cản giữa bước → robot không đi,
cho VLM **quan sát lại** để né.

## YOLO — gợi ý vật cản cho VLM (tùy chọn)

- **YOLO11n** (ultralytics, `demo1/models/yolo11n.pt`, 80 lớp COCO) detect vật trong khung hình;
  danh sách nhãn được **mớm vào prompt** để VLM né/định hướng tốt hơn.
- Tắt: `USE_YOLO=0`.
- ultralytics cần ở python3.12 (runtime demo1):
  `python3.12 -m pip install --user --break-system-packages --no-deps ultralytics ultralytics-thop`

## Chạy thật (chỉ cần driver robot, KHÔNG Nav2)

```bash
# Terminal 1 — driver + camera + /odom + /scan + twist_mux (không Nav2/SLAM)
export ROBOT_IP=192.168.1.7
ros2 launch go2_robot_sdk robot.launch.py nav2:=false slam:=false rviz2:=false foxglove:=false
# Terminal 2 — web
cd /home/dsc-labs/ros2_vlm/src
./run_demo1.sh        # http://localhost:8001
```

Chi tiết SSH/nhập lệnh: xem `src/run.md`.

## Xem UI không cần robot

```bash
cd /home/dsc-labs/ros2_vlm/src
DEMO_MOCK=1 ./run_demo1.sh           # ảnh tĩnh, có VLM thật
DEMO_MOCK=1 VLM_SKIP_MODEL=1 python -m vlm.webconsole.demo1.agent_server  # chỉ xem giao diện
```

## Test

```bash
cd /home/dsc-labs/ros2_vlm/src
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest vlm/webconsole -v
```

---

# 🛠️ DEBUG — lỗi gì sửa ở file nào (phần VLM)

**Dashboard số liệu trên GUI** (ngay dưới "Live Camera"): hiện realtime `target / gap px /
offset° / obstacle m / mode`, **poll 5Hz** từ endpoint `/debug` — tách khỏi WebSocket nên
KHÔNG gây lag camera. Nhìn dashboard biết ngay vì sao robot xoay/tiến/dừng:
- `gap` xanh = đáy box đã ≤ ngưỡng dừng. `offset` xanh = đã căn giữa (≤ `VLA_CENTER_TOL_DEG`),
  vàng = đang lệch (sẽ xoay). `obstacle` đỏ = vật cản quá gần.
- Log tiến trình (bên phải) chỉ in khi đổi bước/quyết định → không trôi mất thông tin.


> Toàn bộ phần VLM là **script Python** (`python -m webconsole.demo1.agent_server`),
> **KHÔNG cần `colcon build`** — sửa `.py` xong chỉ cần `pkill -f agent_server` rồi
> chạy lại `./run_demo1.sh`. (Chỉ `go2_robot_sdk`/driver mới cần build.)

## Bản đồ file
| File | Lo việc gì |
|---|---|
| [vlm_engine.py](vlm_engine.py) | **Model VLM** Qwen2.5-VL: load model, `stream_infer(ảnh, prompt)`. Đổi model/size, device. |
| [demo1/navloop.py](demo1/navloop.py) | **Bộ não VLA**: `_SYSTEM` prompt, `build_nav_prompt`, `parse_vla_json`, brain (Local/Api), vòng lặp `NavLoopAgent.run`, **YOLO override điểm dừng** `_stop_override`. |
| [demo1/annotator.py](demo1/annotator.py) | **YOLO vẽ box lên camera** + **lọc nhiễu** (min_conf, EMA, mất dấu), `targets_from_text` (lệnh → lớp COCO). |
| [demo1/yolo_detector.py](demo1/yolo_detector.py) | Model YOLO (ultralytics) + ngưỡng `conf` gốc. |
| [demo1/motion.py](demo1/motion.py) | **Chấp hành** đi/xoay: tốc độ, **chặn vật cản** `front_stop`, vòng kín `/odom`. |
| [demo1/agent_server.py](demo1/agent_server.py) | Web server: định tuyến lệnh (move/turn→motion, action→sport, còn lại→VLA), `/video_feed`, WebSocket. |
| [webui/demo1.html](webui/demo1.html) | Giao diện. |

## Triệu chứng → sửa ở đâu
| Triệu chứng | File / tham số |
|---|---|
| Robot **dừng giữa chừng / dừng sớm** | `navloop.py` `_stop_override` (đã chống mất-dấu); siết `VLA_STOP_BOTTOM_PX`. Mất dấu nhiều → `annotator.py` (tăng `VLA_YOLO_MIN_CONF`, giảm `VLA_BOX_SMOOTH`). |
| **Đáy box chưa chạm mép dưới** khi dừng | `VLA_STOP_BOTTOM_PX` (navloop, nhỏ hơn) + `MOTION_FRONT_STOP` (motion, nhỏ hơn để lại gần). |
| **Mất camera khi đi/xoay** | `motion.py`: `MOTION_ANG_SPEED` (chốt ≤5°/s), `MOTION_LIN_SPEED`. Tăng `VLA_SETTLE_S` (navloop). |
| **Box YOLO nhảy/nhiễu/vỡ** | `annotator.py`: `VLA_YOLO_MIN_CONF` (tăng 0.6-0.8), `VLA_BOX_SMOOTH` (giảm 0.25), `VLA_BOX_MAX_MISS`. |
| **VLM lạm dụng turn / không căn giữa / suy luận sai** | `navloop.py` `_SYSTEM` (sửa luật prompt). |
| **VLM yếu (3B)** | đổi não: env `VLA_BRAIN=api VLA_API_URL=... VLA_MODEL=...` (navloop `make_brain`) hoặc đổi model trong `vlm_engine.py`. |
| **JSON VLM hỏng / parse lỗi** | `navloop.py` `parse_vla_json`. |
| **Box không hiện / bắt sai vật** | `annotator.py` `targets_from_text` (thêm từ Việt); tên vật phải thuộc **80 lớp COCO**. |
| **Robot không lại gần được (bị chặn)** | `motion.py` `front_stop`/`MOTION_FRONT_STOP`; lidar `/scan` range_min ở `navigation/robot.launch.py`. |
| **Bước ép tiến quá to/nhỏ** | `VLA_OVERRIDE_STEP_M` (navloop). |
| **Lệnh tự nhiên không chạy VLA** | `agent_server.py` định tuyến WebSocket. |
| **Web mất cam / "disconnect"** | KHÔNG phải VLM — là **driver/robot** (token rỗng, `ROBOT_IP`, restart driver). Xem `../../run.md`. |

## Tất cả env (đặt trước `./run_demo1.sh`)
| Env | File đọc | Mặc định | Ý nghĩa |
|---|---|---|---|
| `VLA_CONTROL` | navloop | **vlm** | `vlm` = THUẦN VLM (mọi ưu tiên trong JSON) \| `servo` = visual servo code-cứng |
| `VLA_SERVO_HZ` | navloop | 4 | tần số xử lý ảnh + điều khiển servo (Hz) |
| `VLA_SERVO_VX` | navloop | 0.25 | tốc độ tiến servo (m/s) |
| `VLA_SERVO_WZ_MAX` | navloop | 0.175 | ωz tối đa khi căn giữa (rad/s ≈ 10°/s) |
| `VLA_SERVO_KP` | navloop | 1.5 | hệ số P cho ωz (xoay mạnh/nhẹ theo độ lệch) |
| `VLA_CENTER_TOL_DEG` | navloop | 5 | sai số góc cho phép coi là "đã giữa" |
| `VLA_PICK` | annotator | area | trùng tên thì chọn `area`=to nhất \| `conf`=tin cậy nhất |
| `VLA_FRONT_SAFE_M` | navloop | 0.35 | (mode vlm) ngưỡng an toàn lidar (m) |
| `VLA_BRAIN` | navloop | local | `local` (Qwen 3B) \| `api` (OpenAI-compatible) |
| `VLA_API_URL`/`VLA_API_KEY`/`VLA_MODEL` | navloop | — | endpoint khi `VLA_BRAIN=api` |
| `VLA_MAX_STEPS` | navloop | 20 | số bước tối đa |
| `VLA_SETTLE_S` | navloop | 1.0 | dừng lắng sau mỗi bước (s) cho cam ổn định |
| `VLA_STOP_BOTTOM_PX` | navloop | 20 | đáy box cách mép dưới ≤ px này → DỪNG NGAY (ảnh cao 720) |
| `VLA_HFOV_DEG` | navloop | 90 | FOV ngang camera (để quy px→độ lệch) |
| `VLA_YOLO_IMGSZ` | yolo_detector | 640 | kích thước ảnh YOLO; hạ 480/320 nếu lag |
| `VLA_YOLO_DEVICE` | yolo_detector | (auto) | ép `cuda`/`0` nếu YOLO chạy nhầm CPU |
| `VLA_LOG_DIR` | navloop | vla_logs | nơi ghi log JSONL từng bước |
| `VLA_YOLO_MIN_CONF` | annotator | 0.25 | ngưỡng tin cậy lọc box |
| `VLA_BOX_SMOOTH` | annotator | 0.4 | hệ số EMA (nhỏ = mượt hơn, trễ hơn) |
| `VLA_BOX_MAX_MISS` | annotator | 3 | số frame mất dấu mới xoá box |
| `MOTION_LIN_SPEED` | motion | 0.20 | tốc độ đi (m/s) |
| `MOTION_ANG_SPEED` | motion | 0.35 (~20°/s) | tốc độ xoay (rad/s) |
| `MOTION_ANG_MAX_DEG` | motion | 10 | TRẦN tốc độ xoay (°/s) — chốt cứng mọi turn (cả VLM) |
| `VLA_MAX_TURN_DEG` | navloop | 10 | góc TỐI ĐA mỗi lệnh turn của VLM (chặn VLM xuất 90°) |
| `VLA_SEARCH_MAX_DEG` | navloop | 360 | quét tìm tích luỹ quá góc này không thấy mục tiêu -> dừng |
| `MOTION_TURN_SIGN` | motion | 1 | đặt `-1` nếu robot xoay NGƯỢC chiều (càng xoay mục tiêu càng lệch xa) |
| `MOTION_FRONT_STOP` | motion | 0.30 | khoảng cách lidar coi là vật cản (m) |
| `USE_YOLO` | agent_server | 1 | bật/tắt YOLO (0 = tắt) |
| `DEMO_MOCK` / `VLM_SKIP_MODEL` | agent_server | — | chạy không cần robot / không load model |
