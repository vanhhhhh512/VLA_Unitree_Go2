"""Nav2Navigator: bọc BasicNavigator (nav2_simple_commander)."""
import math
import time


def _yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def yaw_toward(x, y, fx, fy):
    """Hướng (rad) từ điểm (x,y) nhìn vào điểm (fx,fy)."""
    return math.atan2(fy - y, fx - x)


def goal_yaw(room):
    """Yaw đích: nếu room.face đặt và khác vị trí robot -> quay mặt vào face;
    ngược lại dùng room.yaw."""
    face = getattr(room, "face", None)
    if face is not None:
        dx, dy = float(face[0]) - float(room.x), float(face[1]) - float(room.y)
        if (dx * dx + dy * dy) > 1e-6:  # tránh vector 0
            return math.atan2(dy, dx)
    return float(room.yaw)


class Nav2Navigator:
    def __init__(self, timeout_sec=120.0):
        from nav2_simple_commander.robot_navigator import BasicNavigator
        self.nav = BasicNavigator()
        self.timeout_sec = timeout_sec

    def cancel(self):
        """Hủy goal đang chạy (gọi từ thread khác khi người dùng bấm Stop)."""
        try:
            self.nav.cancelTask()
        except Exception:
            pass

    def _make_pose(self, room):
        from geometry_msgs.msg import PoseStamped
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = float(room.x)
        pose.pose.position.y = float(room.y)
        qx, qy, qz, qw = _yaw_to_quat(goal_yaw(room))
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def go_to(self, room):
        from nav2_simple_commander.robot_navigator import TaskResult
        pose = self._make_pose(room)
        pose.header.stamp = self.nav.get_clock().now().to_msg()
        self.nav.goToPose(pose)
        start = time.time()
        while not self.nav.isTaskComplete():
            fb = self.nav.getFeedback()
            if fb is not None:
                yield {"kind": "feedback",
                       "distance_remaining": float(fb.distance_remaining)}
            if time.time() - start > self.timeout_sec:
                self.nav.cancelTask()
                yield {"kind": "error", "message": "Quá thời gian điều hướng."}
                return
            time.sleep(0.5)
        result = self.nav.getResult()
        yield {"kind": "done", "success": result == TaskResult.SUCCEEDED}
