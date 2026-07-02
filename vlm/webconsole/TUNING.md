# Demo1 — Bảng THAM SỐ chỉnh được (đặt env trước `./run_demo1.sh`)

Ba "não" (`VLA_BRAIN`): `local` (Qwen), `api` (OpenAI-compatible), `navila` (NaVILA server).
Hai "tay lái" (`VLA_CONTROL`): `servo` (điều khiển liên tục theo YOLO) | `vlm` (theo JSON của não).

---

## 0. Chọn chế độ
| Env | Mặc định | Ý nghĩa |
|---|---|---|
| `VLA_BRAIN` | `local` | `local`=Qwen \| `api` \| `navila` |
| `VLA_CONTROL` | `servo` (đặt trong run_demo1.sh) | `servo` \| `vlm` |
| `USE_YOLO` | `1` | `0` = tắt YOLO (NaVILA không cần) |
| `DECODE_LIDAR` (driver) | `true` | `false` = tắt lidar cho cam mượt (mất né vật cản) |

---

## 1. Chung cho vòng lặp — navloop.py
| Env | Mặc định | Ý nghĩa |
|---|---|---|
| `VLA_MAX_STEPS` | 20 | số bước tối đa mỗi lệnh |
| `VLA_MAX_TURN_DEG` | 10 | **chặn góc xoay tối đa mỗi lệnh** (kể cả VLM/NaVILA). NaVILA hay 15-30° → nâng `35` |
| `VLA_SEARCH_MAX_DEG` | 360 | quét đủ vòng này không thấy mục tiêu → dừng |
| `VLA_HFOV_DEG` | 90 | FOV ngang camera (quy px→độ lệch) |
| `VLA_SETTLE_S` | 1.0 | dừng lắng giữa các bước (vlm) |

## 2. Chế độ SERVO (VLA_CONTROL=servo) — navloop.py
| Env | Mặc định | Ý nghĩa |
|---|---|---|
| `VLA_CENTER_TOL_DEG` | 5 | **|lệch| ≤ đây → bơm vx đi thẳng**; > đây → xoay tại chỗ |
| `VLA_SERVO_HZ` | 20 | nhịp bơm vận tốc (≥15 để Go2 không tự phanh giữa lệnh) |
| `VLA_SERVO_VX` | 0.25 | tốc độ tiến (m/s) |
| `VLA_SERVO_WZ_MAX` | 0.175 | tốc độ xoay tối đa (rad/s ~10°/s) — tăng để xoay nhanh |
| `VLA_SERVO_WZ_MIN` | 0.10 | tốc độ xoay tối thiểu (thắng ì) |
| `VLA_SERVO_KP` | 1.5 | độ nhạy xoay theo độ lệch |
| `VLA_SERVO_TIMEOUT` | 30 | quá lâu 1 chặng (s) |

## 3. DỪNG khi tới vật (servo) — navloop.py
| Env | Mặc định | Ý nghĩa |
|---|---|---|
| `VLA_STOP_BOTTOM_PX` | 20 | đáy box cách mép dưới ≤ px này (đo lúc ĐỨNG YÊN) → dừng. Nhỏ=sát hơn |
| `VLA_STOP_SETTLE_S` | 1.2 | đứng yên bao lâu rồi đo lại frame cuối (chống cam nhấp nhô/gait) |
| `VLA_FINAL_PUSH_M` | 0 | sau khi đáy chạm mép, **tiến thêm N mét** (bù cam nghiêng còn cách vật ~0.5m) |

## 4. Não NaVILA (VLA_BRAIN=navila) — navloop.py + navila_server.py
| Env / arg | Mặc định | Ý nghĩa |
|---|---|---|
| `VLA_NAVILA_URL` | `http://127.0.0.1:8100` | địa chỉ navila_server |
| `VLA_NAVILA_FRAMES` | 8 | số frame lịch sử gửi lên (NaVILA train ở 8) |
| server `--load-4bit` | (bật) | lượng tử 4-bit cho vừa 16GB (đỉnh ~8.3GB) |
| server `--port` | 8100 | cổng HTTP |
| server `--num-frames` | 8 | phải khớp `VLA_NAVILA_FRAMES` |
| server `--conv-mode` | `llama_3` | template hội thoại |
| body `max_new_tokens` | 32 | độ dài action sinh ra |

## 5. Não API (VLA_BRAIN=api) — navloop.py
`VLA_API_URL`, `VLA_API_KEY`, `VLA_MODEL` (mặc định `qwen2.5-vl-7b-instruct`).

## 6. Motion cấp thấp (lệnh tay + chấp hành turn/move của vlm/NaVILA) — motion.py
| Env | Mặc định | Ý nghĩa |
|---|---|---|
| `MOTION_TURN_SIGN` | 1 | **đặt `-1` nếu robot xoay NGƯỢC chiều lệnh** (nguyên nhân "xoay mãi không tới") |
| `MOTION_ANG_MAX_DEG` | 10 | trần tốc độ xoay lệnh rời rạc (°/s). KHÔNG áp cho servo |
| `MOTION_ANG_SPEED` | 0.35 | tốc độ xoay lệnh rời rạc (bị trần trên) |
| `MOTION_LIN_SPEED` | 0.3 | tốc độ tiến lệnh rời rạc (m/s) |
| `MOTION_TURN_TOL_DEG` | 5 | dung sai HOÀN THÀNH 1 lệnh xoay (deadband, KHÔNG phải ngưỡng căn giữa) |
| `MOTION_FRONT_STOP` | 0.30 | lidar coi là vật cản trước mặt (m) — chỉ khi lidar bật |

## 7. YOLO nhận dạng — yolo_detector.py / annotator.py
| Env | Mặc định | Ý nghĩa |
|---|---|---|
| `VLA_YOLO_WEIGHTS` | yolo11n.pt | đổi `yolo11s/m/l/x.pt` để nhận tốt hơn |
| `VLA_YOLO_MIN_CONF` | 0.3 | ngưỡng tin cậy (giảm = bắt vật mờ/xa) |
| `VLA_YOLO_CONF` | =MIN_CONF | ngưỡng gốc YOLO (đừng để cao hơn MIN_CONF) |
| `VLA_YOLO_IMGSZ` | 960 | to = nhận xa hơn (1280 tốt nhất), chậm hơn |
| `VLA_BOX_MAX_MISS` | 3 | giữ box qua bao nhiêu frame rớt |
| `VLA_BOX_SMOOTH` | 0.4 | độ mượt box hiển thị |
| `VLA_PICK` | area | `area`=box to nhất \| `conf`=tin cậy nhất |

## 8. Driver robot — robot.launch.py
`ROBOT_IP` (bắt buộc), `ROBOT_TOKEN` (để RỖNG cho LAN), `DECODE_LIDAR`, `LIDAR_PUBLISH_RATE`.

---

## Cài lại NaVILA-Blackwell từ máy trắng
`bash ~/ros2_vlm/src/navila/setup_navila_blackwell.sh`
