"""Hành động robot Go2: gõ tên hành động trên GUI -> publish WebRtcReq (sport API).

ACTIONS: bảng hành động (api_id + tên/ mô tả tiếng Việt + từ khoá nhận lệnh) — dùng cho
cả matcher lẫn bảng Note trên GUI. match_action() thuần (re) -> unit-test được;
ActionController import ROS lazy.
"""
import re
import unicodedata

SPORT_TOPIC = "rt/api/sport/request"


def _norm(s):
    """Chuẩn hoá Unicode NFC + lower — tránh lệch NFC/NFD của tiếng Việt."""
    return unicodedata.normalize("NFC", s or "").lower()

# warn=True: động tác mạnh/nguy hiểm, cần không gian rộng.
ACTIONS = [
    {"api_id": 1004, "vi": "Đứng dậy", "desc": "Robot đứng dậy, khoá khớp tư thế cao.",
     "keys": ["stand up", "standup", "đứng dậy", "dung day", "đứng lên"]},
    {"api_id": 1005, "vi": "Nằm xuống", "desc": "Hạ người nằm xuống.",
     "keys": ["stand down", "standdown", "nằm xuống", "nam xuong", "nằm"]},
    {"api_id": 1002, "vi": "Đứng thăng bằng", "desc": "Đứng cân bằng, sẵn sàng đi.",
     "keys": ["balance stand", "balance", "thăng bằng", "thang bang"]},
    {"api_id": 1006, "vi": "Phục hồi đứng", "desc": "Tự đứng dậy sau khi ngã.",
     "keys": ["recovery", "phục hồi", "phuc hoi"]},
    {"api_id": 1003, "vi": "Dừng di chuyển", "desc": "Dừng mọi chuyển động sport.",
     "keys": ["stop move", "stopmove", "dừng lại", "dung lai", "đứng yên"]},
    {"api_id": 1001, "vi": "Thả lỏng", "desc": "Thả lỏng/xả khớp (mềm người).",
     "keys": ["damp", "thả lỏng", "tha long", "xả khớp"]},
    {"api_id": 1009, "vi": "Ngồi", "desc": "Robot ngồi xuống.",
     "keys": ["sit down", "sit", "ngồi", "ngoi"]},
    {"api_id": 1010, "vi": "Đứng dậy từ ngồi", "desc": "Đứng lên từ tư thế ngồi.",
     "keys": ["rise sit", "rise", "thôi ngồi"]},
    {"api_id": 1016, "vi": "Chào (vẫy tay)", "desc": "Robot vẫy tay chào.",
     "keys": ["hello", "say hello", "xin chào", "chào", "chao", "vẫy tay", "vay tay"]},
    {"api_id": 1017, "vi": "Vươn vai", "desc": "Vươn vai / duỗi người.",
     "keys": ["stretch", "vươn vai", "vuon vai", "duỗi"]},
    {"api_id": 1036, "vi": "Cảm ơn / Thả tim", "desc": "Cử chỉ thả tim — dùng để cảm ơn.",
     "keys": ["cảm ơn", "cám ơn", "cam on", "thank you", "thanks", "thank",
              "finger heart", "thả tim", "tha tim", "heart"]},
    {"api_id": 1033, "vi": "Lắc hông", "desc": "Lắc hông vui nhộn.",
     "keys": ["wiggle", "lắc hông", "lac hong"]},
    {"api_id": 1022, "vi": "Nhảy 1", "desc": "Bài nhảy số 1.",
     "keys": ["dance 1", "dance1", "nhảy 1", "nhay 1", "dance", "nhảy", "nhay"]},
    {"api_id": 1023, "vi": "Nhảy 2", "desc": "Bài nhảy số 2.",
     "keys": ["dance 2", "dance2", "nhảy 2", "nhay 2"]},
    {"api_id": 1045, "vi": "Đi tự do", "desc": "Chế độ đi tự do.",
     "keys": ["free walk", "freewalk", "đi tự do"]},
    {"api_id": 1051, "vi": "Đi chéo", "desc": "Dáng đi chéo (cross walk).",
     "keys": ["cross walk", "crosswalk", "đi chéo"]},
    {"api_id": 1305, "vi": "Moonwalk", "desc": "Đi kiểu moonwalk.",
     "keys": ["moonwalk", "moon walk"]},
    {"api_id": 1030, "vi": "Lộn nhào trước ⚠️", "desc": "⚠️ Lộn nhào về trước — cần không gian rộng, dễ ngã.",
     "keys": ["front flip", "frontflip", "lộn nhào", "lon nhao", "backflip"]},
    {"api_id": 1031, "vi": "Nhảy bật trước ⚠️", "desc": "⚠️ Bật nhảy về phía trước.",
     "keys": ["front jump", "nhảy bật", "nhay bat"]},
    {"api_id": 1032, "vi": "Chồm trước ⚠️", "desc": "⚠️ Chồm/lao về phía trước.",
     "keys": ["front pounce", "chồm", "pounce"]},
    {"api_id": 1301, "vi": "Trồng cây chuối ⚠️", "desc": "⚠️ Trồng chuối — rất khó, dễ ngã.",
     "keys": ["handstand", "trồng cây chuối", "trong cay chuoi", "chuối"]},
]

# (key, action) sắp theo độ dài key giảm dần -> ưu tiên khớp cụ thể ("dance 2" > "dance").
_KEYS = sorted(
    [(_norm(k), a) for a in ACTIONS for k in a["keys"]],
    key=lambda t: -len(t[0]),
)


def match_action(command):
    """Trả action dict nếu câu lệnh là 1 hành động; None nếu không.
    Bỏ qua câu hỏi (có dấu '?') -> để agentic xử lý."""
    t = _norm(command).strip()
    if not t or "?" in t:
        return None
    for key, act in _KEYS:
        if re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", t):
            return act
    return None


class ActionController:
    def __init__(self, node, out_topic="/webrtc_req", sport_topic=SPORT_TOPIC):
        from go2_interfaces.msg import WebRtcReq
        self._WebRtcReq = WebRtcReq
        self.sport_topic = sport_topic
        self.pub = node.create_publisher(WebRtcReq, out_topic, 10)

    def send(self, api_id):
        msg = self._WebRtcReq()
        msg.id = 0
        msg.topic = self.sport_topic
        msg.api_id = int(api_id)
        msg.parameter = ""
        msg.priority = 0
        self.pub.publish(msg)

    def run(self, act, cancel=None):
        import time
        yield {"type": "step", "id": "action", "status": "running",
               "title": f"Action: {act['vi']}"}
        self.send(act["api_id"])
        time.sleep(0.2)
        yield {"type": "step", "id": "action", "status": "done"}
        yield {"type": "answer", "text": f"Đã gửi lệnh: {act['vi']}.", "state": "UNKNOWN"}
