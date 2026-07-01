"""Vòng lặp tự hành kiểu NaVILA (VLA): VLM nhìn cam -> sinh JSON lệnh -> chấp hành -> lặp.

Khác với demo1 'agentic' (Qwen planner -> Nav2 -> phòng), module này để **chính VLM lái
từng bước**:

    frame cam -> build_nav_prompt() -> Brain.decide() -> JSON {reasoning, action, value,
    unit, obstacles_detected, is_finished} -> parse_vla_json() -> MotionCmd ->
    MotionController.run() (vòng kín /odom + né vật cản + estop) -> frame mới -> lặp...

Tách 2 cấp như phần còn lại của demo1:
  - parse_vla_json() / build_nav_prompt(): thuần (re/json/math) -> unit-test được.
  - Brain (LocalBrain | ApiBrain) + NavLoopAgent: phần I/O (gọi model, ROS) tách riêng.

JSON schema lấy đúng theo bản nháp GUI của người dùng để khỏi phá vỡ hợp đồng:
  {
    "reasoning": "...",
    "action": "move_forward|move_backward|turn_left|turn_right|stop",
    "value": <số>,
    "unit": "cm|m|degrees|deg|rad",
    "obstacles_detected": ["...", ...],
    "is_finished": true|false
  }
"""
import os
import re
import json
import math
import time
import base64
from dataclasses import dataclass, field

from .motion import MotionCmd

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Cận an toàn cho 1 bước (đề phòng model trả số quá lớn). Giới hạn xoay 90°/bước
# để xoay chậm, không mất camera; muốn quay nhiều thì model xoay nhiều bước.
MAX_STEP_METERS = 1.0
MAX_STEP_RAD = math.radians(90)

_MOVE_ACTIONS = {"move_forward", "move_backward"}
_TURN_ACTIONS = {"turn_left", "turn_right"}
_STOP_ACTIONS = {"stop", "done", "finish"}
VALID_ACTIONS = _MOVE_ACTIONS | _TURN_ACTIONS | _STOP_ACTIONS


class VlaError(Exception):
    """JSON VLM sai cú pháp / thiếu khoá / action lạ."""


@dataclass
class VlaStep:
    action: str                      # đã chuẩn hoá (lower)
    value: float                     # theo unit gốc (0 nếu stop)
    unit: str                        # 'cm'|'m'|'degrees'|'deg'|'rad'
    reasoning: str = ""
    obstacles: list = field(default_factory=list)
    is_finished: bool = False
    raw: str = ""

    @property
    def finished(self):
        return self.is_finished or self.action in _STOP_ACTIONS

    def to_motion_cmd(self):
        """Đổi sang MotionCmd (mét / radian). Trả None nếu là lệnh dừng."""
        if self.finished:
            return None
        unit = (self.unit or "").lower()
        if self.action in _MOVE_ACTIONS:
            if unit.startswith("cm") or unit.startswith("centi"):
                meters = self.value / 100.0
            elif unit == "mm":
                meters = self.value / 1000.0
            else:                       # m / met / meter / rỗng -> coi là mét
                meters = self.value
            meters = abs(meters)
            if self.action == "move_backward":
                meters = -meters
            meters = max(-MAX_STEP_METERS, min(MAX_STEP_METERS, meters))
            return MotionCmd("move", meters, self.raw)
        # turn
        if unit.startswith("rad"):
            rad = self.value
        else:                           # degrees / deg / ° / rỗng -> coi là độ
            rad = math.radians(self.value)
        rad = abs(rad)
        if self.action == "turn_right":
            rad = -rad
        rad = max(-MAX_STEP_RAD, min(MAX_STEP_RAD, rad))
        return MotionCmd("turn", rad, self.raw)

    def describe(self):
        """Mô tả ngắn cho GUI / history."""
        if self.finished:
            return "stop (đã tới đích)"
        if self.action in _MOVE_ACTIONS:
            mc = self.to_motion_cmd()
            lbl = "forward" if mc.value >= 0 else "backward"
            return f"move {lbl} {abs(mc.value):.2f} m"
        mc = self.to_motion_cmd()
        lbl = "left" if mc.value >= 0 else "right"
        return f"turn {lbl} {math.degrees(abs(mc.value)):.0f}°"


def parse_vla_json(text):
    """Parse output VLM -> VlaStep. Raise VlaError nếu hỏng.

    Chịu được model bọc JSON trong ```json ... ``` hoặc kèm chữ thừa quanh JSON.
    """
    raw = text or ""
    m = _JSON_RE.search(raw)
    if not m:
        raise VlaError("Không tìm thấy JSON trong output của VLM.")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise VlaError(f"JSON không hợp lệ: {e}")
    if not isinstance(data, dict):
        raise VlaError("JSON không phải object.")

    action = str(data.get("action", "")).strip().lower()
    is_finished = bool(data.get("is_finished", False))

    # 'is_finished' mà không có action hợp lệ -> coi là stop.
    if not action and is_finished:
        action = "stop"
    if action not in VALID_ACTIONS:
        raise VlaError(f"Action '{action}' không hợp lệ. Cho phép: {sorted(VALID_ACTIONS)}")

    value = 0.0
    if action not in _STOP_ACTIONS:
        try:
            value = float(data.get("value"))
        except (TypeError, ValueError):
            raise VlaError(f"value '{data.get('value')}' không phải số.")

    obstacles = data.get("obstacles_detected") or data.get("obstacles") or []
    if not isinstance(obstacles, list):
        obstacles = [str(obstacles)]

    return VlaStep(
        action=action,
        value=value,
        unit=str(data.get("unit", "")),
        reasoning=str(data.get("reasoning", "")),
        obstacles=[str(o) for o in obstacles],
        is_finished=is_finished,
        raw=raw,
    )


_SYSTEM = (
    "Bạn là bộ não điều hướng VLA của robot 4 chân (Unitree Go2). Dựa vào SỐ LIỆU CẢM BIẾN "
    "(đã đo sẵn ở dưới — DÙNG SỐ NÀY, KHÔNG tự đoán bằng mắt) và ảnh, chọn MỘT hành động.\n"
    "CHỈ TRẢ VỀ JSON, không markdown, không chữ ngoài JSON. Cấu trúc bắt buộc (suy luận theo "
    "thứ tự, điền 3 trường kiểm tra TRƯỚC rồi mới tới action):\n"
    "{\n"
    '  "obstacle_check": "<có vật cản chắn đường < ngưỡng an toàn không? theo số lidar>",\n'
    '  "bottom_touched": "<đáy box mục tiêu đã chạm mép dưới chưa? so gap với ngưỡng>",\n'
    '  "is_centered": "<mục tiêu đã ở giữa (|lệch| <= 5 độ) chưa? theo số lệch>",\n'
    '  "reasoning": "<1 câu kết luận theo ưu tiên>",\n'
    '  "action": "move_forward" | "move_backward" | "turn_left" | "turn_right" | "stop",\n'
    '  "value": <số>,\n'
    '  "unit": "cm" | "m" | "degrees",\n'
    '  "is_finished": true | false\n'
    "}\n"
    "ÁP DỤNG ĐÚNG THỨ TỰ ƯU TIÊN (dừng ở ưu tiên đầu tiên thoả mãn):\n"
    "1) AN TOÀN: nếu vật cản phía trước < ngưỡng an toàn VÀ nó KHÔNG phải mục tiêu -> "
    "action='stop' (hoặc turn để né). \n"
    "2) DỪNG ĐÚNG ĐÍCH: nếu đáy box mục tiêu đã chạm mép dưới (gap <= ngưỡng) -> "
    "action='stop', is_finished=true.\n"
    "3) CĂN GIỮA (chỉ khi |lệch| > 5 độ): turn_left/turn_right value = số độ lệch nhưng "
    "TỐI ĐA 10° mỗi lần (vd lệch 8° -> turn 8; lệch 30° -> turn 10, lặp lại bước sau).\n"
    "4) TIẾN (BẮT BUỘC khi |lệch| <= 5 độ và chưa chạm đáy): action=move_forward 15-25 cm. "
    "TUYỆT ĐỐI KHÔNG turn khi |lệch| <= 5 độ — phải đi thẳng tới vật.\n"
    "Nếu CHƯA thấy mục tiêu (không có số liệu mục tiêu) -> turn_left/right 10 degrees quét tìm "
    "(bước nhỏ để không lướt qua mục tiêu).\n"
    "\nVÍ DỤ:\n"
    'Cảm biến: "mục tiêu lệch 2° trái; đáy cách mép dưới 10px (ngưỡng 20px); vật cản 1.2m".\n'
    '=> {"obstacle_check":"không (1.2m)","bottom_touched":"rồi (10<=20)","is_centered":"rồi (2<=5)",'
    '"reasoning":"đáy chạm mép -> dừng","action":"stop","value":0,"unit":"cm","is_finished":true}\n'
    'Cảm biến: "mục tiêu lệch 8° phải; đáy cách mép dưới 200px (ngưỡng 20px); vật cản 1.5m".\n'
    '=> {"obstacle_check":"không","bottom_touched":"chưa (200>20)","is_centered":"chưa (8>5)",'
    '"reasoning":"lệch >5 -> căn giữa trước","action":"turn_right","value":8,"unit":"degrees","is_finished":false}\n'
    'Cảm biến: "mục tiêu lệch 3° trái; đáy cách mép dưới 120px (ngưỡng 20px); vật cản 1.0m".\n'
    '=> {"obstacle_check":"không","bottom_touched":"chưa","is_centered":"rồi (3<=5)",'
    '"reasoning":"đã giữa (3<=5), chưa chạm đáy -> ĐI THẲNG","action":"move_forward","value":20,"unit":"cm","is_finished":false}'
)


def build_nav_prompt(instruction, history=None, obstacle_hint=None, state=None):
    """Ghép prompt 1 bước: mục tiêu + SỐ LIỆU CẢM BIẾN (state injection) + lịch sử + YOLO hint."""
    parts = [_SYSTEM, "", f'Mục tiêu của người dùng: "{instruction}"']
    if state:
        parts.append(f"SỐ LIỆU CẢM BIẾN (đã đo): {state}")
    if obstacle_hint:
        parts.append(f"Vật YOLO thấy trong khung: {obstacle_hint}")
    if history:
        parts.append("Các bước đã làm: " + "; ".join(history[-6:]))
    parts.append("Trả về JSON cho bước kế tiếp.")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Brain: nguồn quyết định (local transformers HOẶC API OpenAI-compatible).
# --------------------------------------------------------------------------- #
class LocalBrain:
    """Dùng VLMEngine (Qwen2.5-VL local) sẵn có."""

    def __init__(self, engine):
        self.engine = engine

    def decide(self, frame_bgr, prompt):
        return "".join(self.engine.stream_infer(frame_bgr, prompt))


class ApiBrain:
    """Gọi endpoint OpenAI-compatible (DashScope / vLLM / Ollama...). Ép json_object.

    Dùng urllib chuẩn thư viện -> không thêm dependency. Gửi frame dạng base64 JPEG.
    """

    def __init__(self, url, key, model, temperature=0.1, timeout=60):
        self.url = url
        self.key = key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def decide(self, frame_bgr, prompt):
        import urllib.request
        from ..vlm_engine import encode_frame_jpeg

        b64 = base64.b64encode(encode_frame_jpeg(frame_bgr)).decode("ascii")
        body = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            "temperature": self.temperature,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


_NAV_NUM = r"([-+]?\d*\.?\d+)"


def parse_navila_action(text):
    """Đổi câu action ngôn ngữ tự nhiên của NaVILA -> JSON theo schema demo1 (để
    parse_vla_json dùng lại nguyên vòng lặp). Vd:
      'The next action is move forward 25 cm.' -> move_forward 25 cm
      'turn left 15 degrees'                   -> turn_left 15 degrees
      'stop'                                    -> stop (is_finished)
    """
    t = (text or "").lower()
    reason = (text or "").strip()
    if re.search(r"\bstop\b|completed|finished|task is done|reached", t):
        return json.dumps({"action": "stop", "value": 0, "unit": "cm",
                           "reasoning": reason, "is_finished": True})
    m = re.search(r"turn\s+(left|right)\D*" + _NAV_NUM + r"\s*(deg|degree|°)", t)
    if m:
        act = "turn_left" if m.group(1) == "left" else "turn_right"
        return json.dumps({"action": act, "value": float(m.group(2)), "unit": "degrees",
                           "reasoning": reason, "is_finished": False})
    m = re.search(r"(forward|backward|back|ahead)\D*" + _NAV_NUM +
                  r"\s*(cm|centimet\w*|mm|m|met\w*|meter\w*)", t)
    if m:
        act = "move_backward" if m.group(1) in ("backward", "back") else "move_forward"
        u = m.group(3)
        unit = "cm" if u.startswith(("cm", "centi")) else ("mm" if u == "mm" else "m")
        return json.dumps({"action": act, "value": float(m.group(2)), "unit": unit,
                           "reasoning": reason, "is_finished": False})
    # không parse được -> DỪNG cho an toàn (đừng xoay mù vô tận). Đặt NAVILA_FALLBACK=scan
    # nếu muốn quét tìm thay vì dừng.
    if os.getenv("NAVILA_FALLBACK", "stop") == "scan":
        return json.dumps({"action": "turn_left", "value": 10, "unit": "degrees",
                           "reasoning": "NaVILA không parse được: " + reason, "is_finished": False})
    return json.dumps({"action": "stop", "value": 0, "unit": "cm",
                       "reasoning": "NaVILA không cho action hợp lệ (dừng an toàn): " + reason,
                       "is_finished": False})


class NaVilaBrain:
    """Client tới navila_server (chạy env 'navila'). Giữ đệm 8 frame, gửi HTTP mỗi bước,
    trả JSON đã chuẩn hoá để vòng lặp dùng như brain thường."""

    def __init__(self, url, num_frames=8, timeout=30):
        self.url = url.rstrip("/") + "/decide"
        self.num_frames = num_frames
        self.timeout = timeout
        self._buf = []            # base64 JPEG, cũ -> mới
        self.instruction = ""
        self.last = {}            # debug: raw + meta lần decide gần nhất (hiện lên GUI)

    def set_instruction(self, instruction):
        self.instruction = instruction
        self._buf = []            # reset lịch sử mỗi lệnh mới

    def decide(self, frame_bgr, prompt):
        import urllib.request
        from ..vlm_engine import encode_frame_jpeg
        if frame_bgr is not None:
            self._buf.append(base64.b64encode(encode_frame_jpeg(frame_bgr)).decode("ascii"))
            self._buf = self._buf[-self.num_frames:]
        body = {"instruction": self.instruction or prompt, "frames": self._buf}
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        self.last = {"raw": data.get("raw", ""), "latency_s": data.get("latency_s"),
                     "n_frames": data.get("n_frames"), "sent": len(self._buf)}
        return parse_navila_action(data.get("raw", ""))


def make_brain(engine=None):
    """Chọn brain theo env VLA_BRAIN ('local' mặc định | 'api' | 'navila')."""
    mode = os.getenv("VLA_BRAIN", "local").lower()
    if mode == "navila":
        url = os.getenv("VLA_NAVILA_URL", "http://127.0.0.1:8100")
        nf = int(os.getenv("VLA_NAVILA_FRAMES", "8"))
        return NaVilaBrain(url, num_frames=nf)
    if mode == "api":
        url = os.getenv("VLA_API_URL")
        key = os.getenv("VLA_API_KEY", "")
        model = os.getenv("VLA_MODEL", "qwen2.5-vl-7b-instruct")
        if not url:
            raise RuntimeError("VLA_BRAIN=api nhưng thiếu VLA_API_URL.")
        return ApiBrain(url, key, model)
    if engine is None:
        raise RuntimeError("LocalBrain cần VLMEngine đã load.")
    return LocalBrain(engine)


# --------------------------------------------------------------------------- #
# NavLoopAgent: vòng lặp quan sát -> quyết định -> chấp hành.
# --------------------------------------------------------------------------- #
class NavLoopAgent:
    """Phát event theo schema demo1 (step/token/nav/answer/error) để tái dùng GUI."""

    def __init__(self, brain, frame_source, motion, detector=None,
                 max_steps=20, log_dir=None, settle_s=None, stop_bottom_px=None,
                 annotator=None, control=None):
        self.brain = brain
        self.frame_source = frame_source
        self.motion = motion
        self.detector = detector
        # annotator: nguồn box YOLO ĐÃ LÀM MƯỢT (EMA) -> điểm dừng ổn định.
        self.annotator = annotator
        self.max_steps = int(os.getenv("VLA_MAX_STEPS", max_steps))
        self.log_dir = log_dir or os.getenv("VLA_LOG_DIR", "vla_logs")
        # Dừng lắng sau mỗi lần đi/xoay để camera WebRTC ổn định trước khi VLM quan sát.
        self.settle_s = (float(os.getenv("VLA_SETTLE_S", "1.0"))
                         if settle_s is None else settle_s)
        # YOLO override: khi VLM báo dừng nhưng đáy hộp mục tiêu còn cách đáy ảnh
        # > ngưỡng px này -> ép đi tiếp (toán cứng đảm bảo điểm dừng).
        # Camera 720px cao: 20px ≈ 2.8% -> đáy box phải gần sát mép dưới mới cho dừng.
        self.stop_bottom_px = (int(os.getenv("VLA_STOP_BOTTOM_PX", "20"))
                               if stop_bottom_px is None else stop_bottom_px)
        # Số liệu bơm cho VLM + servo: sai số góc & FOV.
        self.center_tol_deg = float(os.getenv("VLA_CENTER_TOL_DEG", "5"))   # sai số góc cho phép
        self.hfov_deg = float(os.getenv("VLA_HFOV_DEG", "90"))             # FOV ngang camera
        self.yolo_min_conf = float(os.getenv("VLA_YOLO_MIN_CONF", "0.25"))  # lọc box điều khiển
        self._targets = set()             # lớp COCO mục tiêu (tính 1 lần/lệnh)
        # Quét quá 1 vòng (360°) không thấy mục tiêu -> dừng (history[-6:] khiến VLM "quên"
        # đã xoay bao nhiêu -> xoay vô tận). Bộ đếm cứng chặn lại.
        self.search_max_deg = float(os.getenv("VLA_SEARCH_MAX_DEG", "360"))
        # Chế độ: 'vlm' (MẶC ĐỊNH) = THUẦN VLM — mọi logic ưu tiên nằm trong JSON/prompt,
        # code KHÔNG đè quyết định (chỉ bơm số liệu + chấp hành). 'servo' = visual servo
        # liên tục code-cứng (chọn khi cần tất định/an toàn).
        self.control = (os.getenv("VLA_CONTROL", "vlm").lower()
                        if control is None else control)
        self.front_safe_m = float(os.getenv("VLA_FRONT_SAFE_M", "0.35"))   # ngưỡng an toàn lidar
        # --- Visual servo liên tục ---
        # Tần số vòng servo = nhịp BƠM lệnh vận tốc. PHẢI cao (≥15Hz) nếu không Go2 có
        # watchdog cmd_vel sẽ tự về 0 giữa 2 lệnh -> robot đứng ì dù servo vẫn ra wz.
        self.servo_hz = float(os.getenv("VLA_SERVO_HZ", "20"))            # nhịp publish vận tốc
        self.servo_vx = float(os.getenv("VLA_SERVO_VX", "0.35"))          # vx tiến (m/s)
        # ωz tối đa = 10°/s (0.175 rad/s) — chốt cứng tốc độ xoay servo.
        self.servo_wz_max = float(os.getenv("VLA_SERVO_WZ_MAX", "0.175"))
        # Góc TỐI ĐA mỗi lệnh turn (kể cả VLM tự quyết) — VLM hay xuất 90°, chặn lại.
        self.max_turn_deg = float(os.getenv("VLA_MAX_TURN_DEG", "10"))
        self.servo_wz_min = float(os.getenv("VLA_SERVO_WZ_MIN", "0.15"))  # ωz tối thiểu (thắng ì)
        self.servo_kp = float(os.getenv("VLA_SERVO_KP", "1.5"))           # hệ số P cho ωz
        # Servo mất dấu THOÁNG QUA (YOLO rớt 1-2 frame do conf thấp) -> đứng yên quan sát lại
        # tối đa N lần (KHÔNG cho VLM xoay quét), quá thì mới QUÉT TÌM bằng servo.
        self.lost_reobserve_max = int(os.getenv("VLA_LOST_REOBS_MAX", "8"))
        self.servo_timeout = float(os.getenv("VLA_SERVO_TIMEOUT", "30"))  # giây
        self.turn_sign = float(os.getenv("MOTION_TURN_SIGN", "1"))        # đảo chiều xoay nếu cần
        # Bơm vận tốc trong 1 XUNG ngắn rồi DỪNG + chờ frame mới -> không xoay/đi mù khi
        # camera chậm (tránh vọt quá deadband). Tăng refresh nếu camera fps thấp.
        self.servo_pulse_s = float(os.getenv("VLA_SERVO_PULSE_S", "0.15"))
        self.servo_refresh_s = float(os.getenv("VLA_SERVO_REFRESH_S", "0.4"))
        # Nghi tới đích -> DỪNG hẳn, chờ cam ổn định (hết nhấp nhô do gait) rồi ĐO LẠI gap
        # trên frame TĨNH; chỉ chốt dừng nếu gap tĩnh vẫn ≤ ngưỡng (chống dừng sớm do bob).
        self.stop_settle_s = float(os.getenv("VLA_STOP_SETTLE_S", "1.2"))
        # Đáy box chạm mép nhưng cam nghiêng nên vật vẫn còn cách ~nửa mét -> sau khi xác
        # nhận chạm mép, TIẾN THÊM đoạn này (mét, đo bằng /odom) cho sát vật. 0 = không tiến.
        self.final_push_m = float(os.getenv("VLA_FINAL_PUSH_M", "0.2"))

    # -- logging ----------------------------------------------------------- #
    def _open_log(self, instruction):
        try:
            from datetime import datetime
            os.makedirs(self.log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.log_dir, f"nav_{ts}.jsonl")
            f = open(path, "w", encoding="utf-8")
            f.write(json.dumps({"event": "start", "instruction": instruction},
                               ensure_ascii=False) + "\n")
            f.flush()
            return f, path
        except Exception:
            return None, None

    def _log(self, logf, i, step):
        if logf is None:
            return
        try:
            logf.write(json.dumps({
                "step": i,
                "action": step.action,
                "value": step.value,
                "unit": step.unit,
                "reasoning": step.reasoning,
                "obstacles_detected": step.obstacles,
                "is_finished": step.finished,
                "raw": step.raw,
            }, ensure_ascii=False) + "\n")
            logf.flush()
        except Exception:
            pass

    # -- gợi ý vật từ YOLO: lấy từ annotator (KHÔNG detect riêng -> đỡ nghẽn) -------- #
    def _obstacle_hint(self, frame):
        if self.annotator is not None and getattr(self.annotator, "raw_box", None) is not None:
            return getattr(self.annotator, "label", None)
        return None

    def _target_box(self, frame, instruction):
        """Box mục tiêu cho điều khiển. DÙNG CHUNG 1 nguồn YOLO với annotator (box thô mới
        nhất) -> KHÔNG chạy YOLO trùng -> đỡ nghẽn CPU/GPU -> camera mượt.
        Chỉ tự detect khi KHÔNG có annotator (test/fallback)."""
        if self.annotator is not None:
            rt = getattr(self.annotator, "raw_target", None)
            return rt() if callable(rt) else self.annotator.target_box()
        if self.detector is not None and frame is not None and self._targets:
            try:
                from .annotator import pick_best
                dets = self.detector.detect(frame, wanted=self._targets)
                best = pick_best(dets, self.yolo_min_conf, by="area")
                if best is not None:
                    return (int(best[0]), int(best[1]), int(best[2]), int(best[3]))
            except Exception:
                pass
            return None
        return None

    def _offset_deg(self, box, w):
        """Độ lệch ngang của tâm box so với tâm ảnh (độ; + = lệch phải)."""
        cx = (box[0] + box[2]) / 2.0
        return ((cx - w / 2.0) / (w / 2.0)) * (self.hfov_deg / 2.0)

    def _state_text(self, box, w, h, obstacle_dist):
        """Bơm SỐ LIỆU đo được vào text cho VLM đọc (State Injection)."""
        parts = []
        if box is not None and w and h:
            gap = h - int(box[3])
            off = self._offset_deg(box, w)
            side = "phải" if off > 0 else "trái"
            centered = "RỒI -> ĐI THẲNG" if abs(off) <= self.center_tol_deg else "CHƯA -> xoay"
            parts.append(f"mục tiêu lệch {abs(off):.0f}° {side} "
                         f"(ngưỡng căn giữa ±{self.center_tol_deg:.0f}° => đã giữa: {centered}); "
                         f"đáy cách mép dưới {gap}px (ngưỡng dừng {self.stop_bottom_px}px)")
        else:
            parts.append("KHÔNG CÓ số liệu mục tiêu (chưa thấy trong khung) — BẮT BUỘC dùng "
                         "luật QUÉT TÌM (turn), điền 'N/A' cho bottom_touched và is_centered, "
                         "TUYỆT ĐỐI KHÔNG move_forward")
        if obstacle_dist is not None:
            parts.append(f"vật cản phía trước {obstacle_dist:.2f}m "
                         f"(ngưỡng an toàn {self.front_safe_m:.2f}m)")
        return "; ".join(parts)

    def _drive(self, cmd, cancel):
        """Chấp hành 1 cmd: yield event GUI; yield CUỐI là {'_status': ok|obstacle|fatal}."""
        obstacle = False
        for ev in self.motion.run(cmd, cancel):
            etype = ev.get("type")
            if etype == "answer":
                continue                                   # nuốt 'Done' của motion
            if etype == "error":
                if "Vật cản" in ev.get("message", ""):
                    obstacle = True
                    continue
                yield ev
                yield {"_status": "fatal"}
                return
            yield ev
        yield {"_status": "obstacle" if obstacle else "ok"}

    # -- VISUAL SERVO LIÊN TỤC (bơm vx/ωz @ servo_hz) ---------------------- #
    def _visual_servo(self, instruction, cancel):
        """Vòng kín liên tục theo box YOLO:
          ƯU TIÊN 1: |lệch| > tol -> bơm ωz xoay nhẹ (vx=0) đến khi về giữa (≤ tol°).
          ƯU TIÊN 2: đã giữa -> bơm vx tiến thẳng; đáy box chạm mép -> DỪNG NGAY.
        Yield event GUI; yield CUỐI {'_servo': done|lost|cancel|timeout|obstacle}.
        """
        period = 1.0 / max(1.0, self.servo_hz)
        t0 = time.time()
        last_phase = None
        lost = 0
        search_acc = 0.0              # tổng góc đã QUÉT TÌM khi mất dấu (cap = search_max_deg)
        gap_ema = None                # đáy box↔mép đã làm mượt (chống cam nhấp nhô do gait)
        pub = getattr(self.motion, "_publish", None)
        stop = getattr(self.motion, "_stop", None)
        if not callable(pub) or not callable(stop):
            yield {"_servo": "lost"}                    # motion không hỗ trợ -> nhường VLM
            return
        while True:
            if cancel is not None and cancel.is_set():
                stop(); yield {"_servo": "cancel"}; return
            if time.time() - t0 > self.servo_timeout:
                stop(); yield {"_servo": "timeout"}; return

            frame = self.frame_source.get_latest_frame()
            box = self._target_box(frame, instruction)     # DETECT TƯƠI cùng frame
            if box is None:
                lost += 1
                if lost <= self.lost_reobserve_max:
                    # MẤT DẤU THOÁNG QUA -> ĐỨNG YÊN giữ hướng, chờ thấy lại (không xoay loạn).
                    stop()
                    if lost == 1:
                        last_phase = "lost"
                        yield {"type": "step", "id": "vla", "status": "running",
                               "title": "Servo · mất dấu thoáng qua → đứng yên chờ thấy lại"}
                    time.sleep(period); continue
                # MẤT DẤU LÂU -> QUÉT TÌM bằng servo (xoay đều), cộng dồn tới khi đủ 1 vòng.
                search_acc += abs(math.degrees(self.servo_wz_max)) * period
                if search_acc > self.search_max_deg:
                    stop(); yield {"_servo": "lost"}; return
                pub(0.0, self.servo_wz_max * self.turn_sign)
                if last_phase != "search":
                    last_phase = "search"
                    yield {"type": "step", "id": "vla", "status": "running",
                           "title": f"Servo · QUÉT TÌM mục tiêu ({search_acc:.0f}°)"}
                time.sleep(period); continue
            lost = 0
            search_acc = 0.0
            if frame is not None:
                h, w = frame.shape[0], frame.shape[1]
            else:
                w, h = (self.annotator.frame_width() or 1280,
                        self.annotator.frame_height() or 720)
            gap_raw = h - int(box[3])         # đáy box cách mép dưới ảnh (px) — TÍN HIỆU DỪNG
            # EMA chỉ để LỌC trigger sớm khi đang đi (cam nhấp nhô do gait); quyết định CUỐI
            # dựa trên NHIỀU mẫu lúc ĐỨNG YÊN + median -> không dừng vì 1 box nhiễu.
            gap_ema = gap_raw if gap_ema is None else 0.4 * gap_raw + 0.6 * gap_ema
            gap = int(gap_ema)

            # Phanh khẩn nếu có vật LẠ quá gần (mục tiêu chưa tới đáy).
            fd = getattr(self.motion, "front_distance", None)
            if callable(fd):
                d = fd()
                if d is not None and d < 0.20 and gap > self.stop_bottom_px:
                    stop()
                    yield {"type": "token", "step": "vla",
                           "text": f"\n🛡️ Vật lạ {d:.2f}m quá gần → DỪNG khẩn.\n"}
                    yield {"_servo": "obstacle"}; return

            # NGHI đáy chạm mép: DỪNG hẳn -> ĐỨNG YÊN hết stop_settle_s cho robot hết "ưỡn"
            # (gait nâng cam lên giả) -> đo LẠI 1 frame CUỐI khi cam đã hạ về thật.
            if gap <= self.stop_bottom_px:
                stop()
                yield {"type": "token", "step": "vla",
                       "text": f"\n⏸️ Nghi đáy chạm mép (gap {gap}px) → đứng yên "
                               f"{self.stop_settle_s:.1f}s rồi đo lại frame cuối…\n"}
                time.sleep(self.stop_settle_s)             # đứng YÊN hẳn (cam hạ về vị trí thật)
                if cancel is not None and cancel.is_set():
                    yield {"_servo": "cancel"}; return
                fb = self.frame_source.get_latest_frame()
                bb = self._target_box(fb, instruction)
                if bb is None:
                    yield {"_servo": "done"}; return        # mất dấu khi đứng -> coi như tới
                hh = (fb.shape[0] if fb is not None
                      else (self.annotator.frame_height() if self.annotator else h)) or h
                gap2 = hh - int(bb[3])
                if gap2 <= self.stop_bottom_px:
                    yield {"type": "token", "step": "vla",
                           "text": f"\n🛑 Đứng yên đo lại: đáy chạm mép (gap {gap2}px ≤ "
                                   f"{self.stop_bottom_px}px).\n"}
                    # Cam nghiêng -> đáy chạm mép mà vật còn cách; TIẾN THÊM cho sát.
                    if self.final_push_m > 0:
                        yield {"type": "token", "step": "vla",
                               "text": f"➡️ Tiến thêm {self.final_push_m:.2f}m cho sát vật…\n"}
                        mv = getattr(self.motion, "_move", None)
                        if callable(mv):
                            for mev in mv(self.final_push_m, cancel):
                                k = mev.get("kind")
                                if k == "cancel":
                                    stop(); yield {"_servo": "cancel"}; return
                                if k == "obstacle":
                                    yield {"type": "token", "step": "vla",
                                           "text": "🛡️ Gặp vật cản khi tiến thêm → dừng.\n"}
                                    break
                        else:                                  # không có /odom -> bơm theo thời gian
                            tpush = self.final_push_m / max(0.05, self.servo_vx)
                            tp0 = time.time()
                            while time.time() - tp0 < tpush:
                                if cancel is not None and cancel.is_set():
                                    stop(); yield {"_servo": "cancel"}; return
                                pub(self.servo_vx, 0.0); time.sleep(period)
                        stop()
                    yield {"type": "token", "step": "vla", "text": "🛑 Đã tới sát mục tiêu → DỪNG.\n"}
                    yield {"_servo": "done"}; return
                yield {"type": "token", "step": "vla",
                       "text": f"\n↩️ Đứng yên đo lại đáy {gap2}px > {self.stop_bottom_px}px "
                               f"(lúc đi cam bị nâng giả) → CHƯA chạm mép, đi tiếp.\n"}
                gap_ema = float(gap2)
                last_phase = None
                continue

            offset = self._offset_deg(box, w)
            if abs(offset) <= self.center_tol_deg:
                # ĐÃ GIỮA (|lệch| ≤ ngưỡng) -> BƠM vx ĐI THẲNG (ωz=0). Đây là điều kiện DUY
                # NHẤT được tiến: lệch còn > ngưỡng thì TUYỆT ĐỐI không tiến.
                vx = self.servo_vx
                if gap < 3 * self.stop_bottom_px:
                    vx *= 0.4                             # sắp chạm mép -> chậm lại (đỡ lố)
                wz = 0.0
                phase = "forward"
            else:
                # CHƯA giữa -> XOAY TẠI CHỖ căn giữa (vx=0), liên tục tới khi ≤ ngưỡng.
                wz = max(-self.servo_wz_max,
                         min(self.servo_wz_max, self.servo_kp * math.radians(offset)))
                if abs(wz) < self.servo_wz_min:
                    wz = self.servo_wz_min if wz >= 0 else -self.servo_wz_min
                wz = -wz * self.turn_sign                 # offset phải -> xoay phải (ωz âm)
                vx = 0.0
                phase = "turn"
            pub(vx, wz)

            if phase != last_phase:
                last_phase = phase
                lbl = ("CĂN GIỮA (lệch %.0f° > %.0f°)" % (abs(offset), self.center_tol_deg)) \
                    if phase == "turn" else \
                    ("ĐI THẲNG (lệch %.0f° ≤ %.0f°, đáy %dpx)"
                     % (abs(offset), self.center_tol_deg, gap))
                yield {"type": "step", "id": "vla", "status": "running",
                       "title": f"Servo · {lbl}"}
            time.sleep(period)                            # liên tục (không stop giữa chừng)

    # -- vòng lặp chính ---------------------------------------------------- #
    def run(self, instruction, cancel=None):
        def cancelled():
            return cancel is not None and cancel.is_set()

        logf, path = self._open_log(instruction)
        if path:
            yield {"type": "step", "id": "vla", "status": "running",
                   "title": f"Auto-Nav bắt đầu · log: {os.path.basename(path)}"}
        if hasattr(self.brain, "set_instruction"):   # NaVILA: cấp lệnh + reset đệm frame
            self.brain.set_instruction(instruction)
        history = []
        seen_target = False          # YOLO đã từng thấy mục tiêu trong phiên này chưa
        # Lớp COCO mục tiêu (1 lần/lệnh) cho detect tươi điều khiển.
        self._targets = set()
        if self.detector is not None:
            try:
                from .annotator import targets_from_text
                self._targets = targets_from_text(
                    instruction, getattr(self.detector, "names", []) or [])
            except Exception:
                self._targets = set()
        search_turn = 0.0            # tổng góc đã quét tìm (reset khi thấy mục tiêu)
        try:
            for i in range(1, self.max_steps + 1):
                if cancelled():
                    yield {"type": "error", "message": "⏹ Đã dừng theo yêu cầu."}
                    return

                frame = self.frame_source.get_latest_frame()
                if frame is None:
                    yield {"type": "step", "id": "vla", "status": "error"}
                    yield {"type": "error", "message": "Không có ảnh camera."}
                    return

                box = self._target_box(frame, instruction)   # DETECT TƯƠI (đỡ lệch nhịp)
                if box is not None:
                    seen_target = True
                    search_turn = 0.0                         # thấy mục tiêu -> reset bộ đếm quét
                    servo_lost = 0
                    w, h = frame.shape[1], frame.shape[0]
                else:
                    w = h = None
                obstacle_dist = (self.motion.front_distance()
                                 if hasattr(self.motion, "front_distance") else None)

                # Servo: ĐÃ từng thấy mục tiêu mà giờ mất dấu THOÁNG QUA -> ĐỨNG YÊN quan sát
                # lại (KHÔNG nhường VLM xoay quét, đỡ "Turning left" vô cớ khi vật vẫn ở đó).
                if self.control == "servo":
                    # ===== SERVO LO TRỌN GÓI: bám / căn giữa / TIẾN ngay ở ≤5° / QUÉT TÌM khi
                    # mất dấu. KHÔNG bao giờ nhường VLM -> ngưỡng đi thẳng luôn = center_tol
                    # (5°), không bị motion._turn "kết thúc ở 2°" như nhánh VLM. =====
                    yield {"type": "step", "id": "vla", "status": "running",
                           "title": f"Bước {i} · Visual servo (bám mục tiêu)"}
                    sstatus = None
                    for ev in self._visual_servo(instruction, cancel):
                        if "_servo" in ev:
                            sstatus = ev["_servo"]
                        else:
                            yield ev
                    if sstatus == "done":
                        yield {"type": "answer",
                               "text": "Đã tới sát mục tiêu (đáy box chạm mép dưới ảnh).",
                               "state": "YES"}
                        return
                    if sstatus == "cancel":
                        yield {"type": "error", "message": "⏹ Đã dừng theo yêu cầu."}
                        return
                    if sstatus == "obstacle":
                        yield {"type": "error", "message": "⛔ Vật cản lạ quá gần — đã dừng."}
                        return
                    if sstatus == "timeout":
                        if self.settle_s > 0 and not cancelled():
                            time.sleep(self.settle_s)
                        continue                              # chạy quá lâu 1 chặng -> tiếp
                    # lost = quét đủ 1 vòng vẫn không thấy -> dừng (KHÔNG nhường VLM xoay).
                    yield {"type": "answer",
                           "text": "Không tìm thấy mục tiêu sau khi quét 1 vòng.",
                           "state": "NO"}
                    return
                else:
                    # ===== VLM tự suy luận theo SỐ LIỆU bơm vào (CoT, đúng bản chất VLM) =====
                    if isinstance(self.brain, NaVilaBrain):
                        # NaVILA tự dựng prompt + dùng 8-frame buffer -> bỏ YOLO/state/hint.
                        prompt = instruction
                    else:
                        state = self._state_text(box, w, h, obstacle_dist)
                        hint = self._obstacle_hint(frame)
                        prompt = build_nav_prompt(instruction, history, hint, state)
                    yield {"type": "step", "id": "vla", "status": "running",
                           "title": f"Bước {i} · VLM suy luận…"}
                    yield {"type": "token", "step": "vla",
                           "text": f"\n══════════ BƯỚC {i} ══════════\n📥 INPUT:\n{prompt}\n"}
                    try:
                        raw = self.brain.decide(frame, prompt)
                    except Exception as e:
                        yield {"type": "step", "id": "vla", "status": "error"}
                        yield {"type": "error", "message": f"VLM lỗi: {e}"}
                        return
                    # DEBUG NaVILA: hiện ĐÚNG câu model trả + số frame gửi/nhận + link ảnh.
                    if isinstance(self.brain, NaVilaBrain) and self.brain.last:
                        d = self.brain.last
                        yield {"type": "token", "step": "vla",
                               "text": (f"\n🧠 NaVILA nhận {d.get('n_frames')}/{d.get('sent')} frame "
                                        f"({d.get('latency_s')}s) — xem ảnh model đọc: "
                                        f"{os.getenv('VLA_NAVILA_URL','http://127.0.0.1:8100')}/last.jpg\n"
                                        f"📷 RAW NaVILA: {d.get('raw')!r}\n")}
                    yield {"type": "token", "step": "vla",
                           "text": f"\n📤 OUTPUT (đã parse):\n{raw}\n"}
                    if cancelled():
                        yield {"type": "error", "message": "⏹ Đã dừng theo yêu cầu."}
                        return
                    try:
                        step = parse_vla_json(raw)
                    except VlaError as e:
                        yield {"type": "step", "id": "vla", "status": "error"}
                        yield {"type": "error", "message": f"JSON VLM hỏng: {e}"}
                        return
                    self._log(logf, i, step)
                    yield {"type": "step", "id": "vla", "status": "done",
                           "title": f"Bước {i} · {step.describe()}"}
                    if step.finished:
                        # THUẦN VLM: VLM quyết dừng -> tin VLM (số liệu đã bơm vào prompt).
                        yield {"type": "answer",
                               "text": step.reasoning or "Đã tới đích.",
                               "state": "YES"}
                        return
                    cmd = step.to_motion_cmd()
                    desc = step.describe()
                    # CHẶN GÓC: kể cả VLM tự quyết turn lớn (vd 90°) -> giới hạn ≤ max_turn_deg.
                    if cmd is not None and cmd.kind == "turn":
                        cap = math.radians(self.max_turn_deg)
                        if abs(cmd.value) > cap:
                            cmd.value = cap if cmd.value > 0 else -cap
                            desc = (f"turn {'left' if cmd.value >= 0 else 'right'} "
                                    f"{self.max_turn_deg:.0f}° [chặn từ {abs(step.value):.0f}°]")

                    # Bộ đếm QUÉT: đang quét tìm (chưa thấy mục tiêu) mà xoay -> cộng dồn;
                    # đủ 1 vòng (search_max_deg) mà vẫn không thấy -> dừng, đỡ xoay vô tận.
                    if box is None and cmd is not None and cmd.kind == "turn":
                        search_turn += abs(math.degrees(cmd.value))
                        if search_turn >= self.search_max_deg:
                            yield {"type": "token", "step": "vla",
                                   "text": f"\n🛡️ Đã quét {search_turn:.0f}° (≥ "
                                           f"{self.search_max_deg:.0f}°) mà không thấy mục tiêu.\n"}
                            yield {"type": "error",
                                   "message": "⛔ Không tìm thấy mục tiêu sau khi quét đủ vòng."}
                            return

                # Chấp hành (dùng chung _drive: MotionController vẫn tự né /scan khi đi).
                status = None
                for ev in self._drive(cmd, cancel):
                    if "_status" in ev:
                        status = ev["_status"]
                    else:
                        yield ev
                if status == "fatal":
                    return
                if status == "obstacle":
                    yield {"type": "token", "step": "vla",
                           "text": "\n⛔ Vật cản phía trước — dừng an toàn, quan sát lại."}
                    history.append(desc + " [BỊ CHẶN bởi vật cản]")
                else:
                    history.append(desc)

                # Dừng lắng cho camera ổn định sau khi đi/xoay.
                if self.settle_s > 0 and not cancelled():
                    yield {"type": "token", "step": "vla",
                           "text": f"\n⏳ Dừng {self.settle_s:.1f}s cho camera ổn định…"}
                    time.sleep(self.settle_s)

            yield {"type": "answer",
                   "text": f"Đã chạy hết {self.max_steps} bước mà chưa tới đích.",
                   "state": "UNKNOWN"}
        finally:
            if logf is not None:
                try:
                    logf.close()
                except Exception:
                    pass
