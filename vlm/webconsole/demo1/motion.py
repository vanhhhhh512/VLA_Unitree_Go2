"""Chế độ lệnh chuyển động rời rạc (NaVILA-style mid-level action).

2 cấp:
  - parse_motion(): "não/parser" — tách câu lệnh -> MotionCmd (quãng đường / góc).
  - MotionController: "tủy sống" — chạy vòng kín bằng /odom, publish TwistStamped vào
    /cmd_vel_joy (priority cao, đè Nav2), tự dừng đúng quãng đường/góc; chặn /scan +
    timeout + cancel để an toàn.

parse_motion thuần (chỉ re/math) -> unit-test được. ROS import nằm trong MotionController
(lazy) nên module import được khi không có ROS.
"""
import re
import math
from dataclasses import dataclass

_NUM = r"([-+]?\d*\.?\d+)"
_MOVE_DIR = re.compile(r"\b(forward|ahead|backward|back|tiến|lùi|tien|lui)\b")
_MOVE_VERB = re.compile(r"\b(move|go|drive|forward|backward|back|ahead|tiến|lùi|tien|lui)\b")
_MOVE_UNIT = re.compile(_NUM + r"\s*(cm|centimet\w*|mm|m|met\w*|meter\w*)\b")
_TURN_VERB = re.compile(r"\b(turn|rotate|spin|xoay|quay)\b")
_TURN_DIR = re.compile(r"\b(left|right|trái|phải|trai|phai)\b")
_TURN_UNIT = re.compile(_NUM + r"\s*(deg|degree\w*|°|rad|radian\w*)")


@dataclass
class MotionCmd:
    kind: str       # "move" | "turn"
    value: float    # mét (move; + tiến, - lùi) | radian (turn; + trái/CCW, - phải/CW)
    raw: str


def parse_motion(command):
    """Trả MotionCmd nếu câu lệnh là lệnh chuyển động; None nếu không phải."""
    t = (command or "").lower().strip()

    # TURN
    if _TURN_VERB.search(t) or _TURN_DIR.search(t):
        mu = _TURN_UNIT.search(t)
        if mu and (_TURN_DIR.search(t) or _TURN_VERB.search(t)):
            val = float(mu.group(1))
            unit = mu.group(2)
            rad = val if unit.startswith("rad") else math.radians(val)
            if re.search(r"\b(right|phải|phai)\b", t):
                rad = -abs(rad)
            else:
                rad = abs(rad)
            return MotionCmd("turn", rad, command)

    # MOVE
    mu = _MOVE_UNIT.search(t)
    if mu and _MOVE_VERB.search(t):
        val = float(mu.group(1))
        unit = mu.group(2)
        if unit == "cm" or unit.startswith("centi"):
            meters = val / 100.0
        elif unit == "mm":
            meters = val / 1000.0
        else:
            meters = val
        if re.search(r"\b(back|backward|lùi|lui)\b", t):
            meters = -abs(meters)
        else:
            meters = abs(meters)
        return MotionCmd("move", meters, command)

    return None


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y),
                      1 - 2 * (q.y * q.y + q.z * q.z))


class MotionController:
    def __init__(self, node, cmd_topic="/cmd_vel_joy", lin_speed=0.3, ang_speed=0.6,
                 rate_hz=20.0, scan_topic="/scan", front_stop=0.45, front_fov_deg=20.0,
                 timeout_s=25.0):
        from geometry_msgs.msg import TwistStamped
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import LaserScan
        from rclpy.qos import qos_profile_sensor_data
        self.node = node
        self.lin_speed = lin_speed
        self.ang_speed = ang_speed
        self.rate_hz = rate_hz
        self.front_stop = front_stop
        self.front_fov = math.radians(front_fov_deg)
        self.timeout_s = timeout_s
        self._TwistStamped = TwistStamped
        self._odom = None
        self._scan = None
        self.pub = node.create_publisher(TwistStamped, cmd_topic, 10)
        node.create_subscription(Odometry, "/odom", self._on_odom, 10)
        node.create_subscription(LaserScan, scan_topic, self._on_scan,
                                 qos_profile_sensor_data)

    def _on_odom(self, msg):
        self._odom = msg.pose.pose

    def _on_scan(self, msg):
        self._scan = msg

    def _front_clear(self):
        s = self._scan
        if s is None:
            return True  # không có scan -> không chặn được (cẩn thận: đi chậm)
        vals = []
        for i, r in enumerate(s.ranges):
            a = s.angle_min + i * s.angle_increment
            if -self.front_fov <= a <= self.front_fov and r > 0.05 and r == r:
                vals.append(r)
        return (min(vals) > self.front_stop) if vals else True

    def _publish(self, vx, wz):
        msg = self._TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = float(vx)
        msg.twist.angular.z = float(wz)
        self.pub.publish(msg)

    def _stop(self):
        for _ in range(3):
            self._publish(0.0, 0.0)

    def _move(self, distance, cancel):
        import time
        if self._odom is None:
            yield {"kind": "nodom"}
            return
        x0, y0 = self._odom.position.x, self._odom.position.y
        target = abs(distance)
        vx = self.lin_speed if distance >= 0 else -self.lin_speed
        t0 = time.time()
        while True:
            if cancel is not None and cancel.is_set():
                yield {"kind": "cancel"}
                return
            if distance > 0 and not self._front_clear():
                yield {"kind": "obstacle"}
                return
            od = self._odom
            cur = math.hypot(od.position.x - x0, od.position.y - y0)
            rem = target - cur
            if rem <= 0.03:
                yield {"kind": "done"}
                return
            if time.time() - t0 > self.timeout_s:
                yield {"kind": "timeout"}
                return
            self._publish(vx, 0.0)
            yield {"kind": "progress", "remaining": rem}
            time.sleep(1.0 / self.rate_hz)

    def _turn(self, angle, cancel):
        import time
        if self._odom is None:
            yield {"kind": "nodom"}
            return
        prev = _yaw(self._odom.orientation)
        acc = 0.0
        target = abs(angle)
        wz = self.ang_speed if angle >= 0 else -self.ang_speed
        t0 = time.time()
        while True:
            if cancel is not None and cancel.is_set():
                yield {"kind": "cancel"}
                return
            cur = _yaw(self._odom.orientation)
            d = (cur - prev + math.pi) % (2 * math.pi) - math.pi
            acc += abs(d)
            prev = cur
            rem = target - acc
            if rem <= math.radians(2):
                yield {"kind": "done"}
                return
            if time.time() - t0 > self.timeout_s:
                yield {"kind": "timeout"}
                return
            self._publish(0.0, wz)
            yield {"kind": "progress", "remaining": rem}
            time.sleep(1.0 / self.rate_hz)

    def run(self, cmd, cancel=None):
        """Generator phát step-events cho GUI (tái dùng schema demo1)."""
        if cmd.kind == "move":
            label = "forward" if cmd.value >= 0 else "backward"
            title = f"Moving {label} {abs(cmd.value):.2f} m"
            steps = self._move(cmd.value, cancel)
        else:
            label = "left" if cmd.value >= 0 else "right"
            title = f"Turning {label} {math.degrees(abs(cmd.value)):.0f}°"
            steps = self._turn(cmd.value, cancel)

        yield {"type": "step", "id": "motion", "status": "running", "title": title}
        result = {"kind": "fail"}
        for ev in steps:
            if ev["kind"] == "progress":
                if cmd.kind == "move":
                    yield {"type": "nav", "distance_remaining": ev["remaining"]}
                else:
                    yield {"type": "step", "id": "motion", "status": "running",
                           "title": f"Turning {label} — {math.degrees(ev['remaining']):.0f}° left"}
            else:
                result = ev
        self._stop()

        kind = result["kind"]
        if kind == "done":
            yield {"type": "step", "id": "motion", "status": "done"}
            yield {"type": "answer", "text": f"Done — {title.lower()}.",
                   "state": "UNKNOWN"}
        elif kind == "obstacle":
            yield {"type": "step", "id": "motion", "status": "error"}
            yield {"type": "error", "message": "⛔ Vật cản phía trước — đã dừng an toàn."}
        elif kind == "cancel":
            yield {"type": "step", "id": "motion", "status": "error"}
            yield {"type": "error", "message": "⏹ Đã dừng theo yêu cầu."}
        elif kind == "nodom":
            yield {"type": "step", "id": "motion", "status": "error"}
            yield {"type": "error", "message": "Không có /odom — robot chưa kết nối?"}
        else:  # timeout / fail
            yield {"type": "step", "id": "motion", "status": "error"}
            yield {"type": "error", "message": "Quá thời gian chuyển động."}
