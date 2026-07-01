# NaVILA — Files, chức năng, tham số & DEBUG

Kiến trúc 2 tiến trình (2 conda env khác nhau, nối bằng HTTP):
```
Web GUI 8001 (env ROS) ──HTTP /decide──► navila_server 8100 (env navila, VILA 8B 4-bit)
  NaVilaBrain: đệm 8 frame, gửi ảnh + lệnh          ◄── raw action text ──┘
  parse_navila_action: text -> JSON -> MotionCmd -> MotionController -> cmd_vel
```
> ⚠️ NaVILA pretrained hiện **ra rác trên camera Go2** (fisheye lệch phân bố train). Bộ này để DEBUG/nghiên cứu & sẵn sàng khi fine-tune. Chạy thật dùng `VLA_CONTROL=servo`.

---

## Các file NaVILA & chức năng

| File | Env chạy | Chức năng |
|---|---|---|
| [navila_server.py](navila_server.py) | **navila** | Server suy luận: nạp model 1 lần, nhận 8 frame + lệnh → trả **action text**. Có endpoint debug. |
| [navila_probe.py](../../../navila_probe.py) (ở `~/navila_probe.py`) | **navila** | Test nhanh 1 lần: đo VRAM/latency + xem output. Không cần ROS/robot. |
| [setup_navila_blackwell.sh](setup_navila_blackwell.sh) | base→navila | Cài/port toàn bộ (idempotent) cho GPU Blackwell. |
| [navloop.py](../vlm/webconsole/demo1/navloop.py) — phần `NaVilaBrain` + `parse_navila_action` | **ROS** | Client HTTP: đệm 8 frame, gọi server, đổi text→JSON schema để vòng lặp dùng. |

---

## Tham số theo file

### navila_server.py — tham số dòng lệnh (arg)
| Arg | Mặc định | Ý nghĩa |
|---|---|---|
| `--model-path` | (bắt buộc) | thư mục checkpoint (`~/navila-ckpt`) |
| `--load-4bit` | tắt | lượng tử 4-bit (BẬT để vừa 16GB, đỉnh ~8.3GB) |
| `--port` | 8100 | cổng HTTP |
| `--num-frames` | 8 | số frame model xử lý (khớp `VLA_NAVILA_FRAMES`) |
| `--conv-mode` | llama_3 | template hội thoại |
| `--model-base` | (none) | chỉ dùng khi LoRA |

### navila_server.py — env
| Env | Mặc định | Ý nghĩa |
|---|---|---|
| `NAVILA_CROP_SQUARE` | 1 | crop tâm về vuông trước khi đưa model (bớt méo). `0`=tắt |
| `CUDA_HOME` | (bắt buộc) | trỏ `~/miniconda3/envs/navila` (deepspeed cần) |
| `PYTORCH_CUDA_ALLOC_CONF` | — | đặt `expandable_segments:True` đỡ phân mảnh |

### navila_server.py — body POST /decide
`instruction` (str), `frames` (list base64 JPEG, cũ→mới), `max_new_tokens` (mặc định 32).

### navloop.py (phần NaVILA) — env
| Env | Mặc định | Ý nghĩa |
|---|---|---|
| `VLA_BRAIN` | local | đặt `navila` để dùng NaVILA |
| `VLA_CONTROL` | servo | đặt `vlm` để đi theo não (NaVILA) |
| `VLA_NAVILA_URL` | http://127.0.0.1:8100 | địa chỉ navila_server |
| `VLA_NAVILA_FRAMES` | 8 | số frame gửi lên |
| `NAVILA_FALLBACK` | stop | khi output không parse được: `stop` (an toàn) \| `scan` (xoay tìm) |
| `VLA_MAX_TURN_DEG` | 10 | chặn góc xoay/bước (NaVILA hay 15-30° → nâng 35) |

### setup_navila_blackwell.sh — env
`NAVILA_DIR` (~/NaVILA), `CKPT_DIR` (~/navila-ckpt), `ENV` (navila), `TORCH_VER` (2.7.0), `TV_VER` (0.22.0), `CUDA_IDX` (cu128), `CUDA_NVCC` (12.8).

### navila_probe.py — arg
`--model-path`, `--load-4bit`, `--frames-dir` (8 ảnh thật), `--num-frames`, `--conv-mode`, `--instruction`, `--max-new-tokens`.

---

## 🔍 DEBUG — xem ĐÚNG cái camera/model đọc

**1) Trên GUI (8001)** — panel REASONING mỗi bước giờ hiện:
```
🧠 NaVILA nhận 8/8 frame (1.2s) — xem ảnh model đọc: http://127.0.0.1:8100/last.jpg
📷 RAW NaVILA: 'The next action is ...'
📤 OUTPUT (đã parse): {...}
```
→ thấy ngay **câu thô NaVILA trả** (nếu là rác/không có số → biết model lỗi grounding).

**2) Ảnh ĐÚNG cái model nhận** (sau crop/resize) — mở trình duyệt:
```
http://192.168.1.29:8100/last.jpg      # 8 frame ghép ngang, refresh để cập nhật
```

**3) JSON raw + meta lần gần nhất:**
```
curl http://127.0.0.1:8100/last         # {raw, instruction, latency_s, n_frames}
```

**4) Log server (T2 terminal)** in mỗi lần:
```
[decide] instr='...' n=8 1.2s -> 'The next action is ...'
```

**5) Test tay ngoài GUI** (chộp frame robot rồi hỏi model):
```bash
source /opt/ros/jazzy/setup.bash && source ~/ros2_vlm/install/setup.bash
# (script chộp 8 frame /camera/image_raw -> ~/frames_robot, rồi POST /decide — xem raw)
```

---

## Chạy (3 terminal)
- T1: driver robot.
- T2: `cd ~/NaVILA && CUDA_HOME=$HOME/miniconda3/envs/navila PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True conda run -n navila --no-capture-output python ~/ros2_vlm/src/navila/navila_server.py --model-path ~/navila-ckpt --load-4bit --port 8100`
- T3: `VLA_BRAIN=navila VLA_CONTROL=vlm VLA_MAX_TURN_DEG=35 USE_YOLO=0 ./run_demo1.sh`

Cài lại từ máy trắng: `bash setup_navila_blackwell.sh`.
