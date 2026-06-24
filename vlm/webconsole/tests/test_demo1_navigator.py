import math
from vlm.webconsole.demo1.navigator import yaw_toward, goal_yaw
from vlm.webconsole.demo1.rooms import Room


def test_yaw_toward_cardinal():
    assert math.isclose(yaw_toward(0, 0, 1, 0), 0.0)           # +x
    assert math.isclose(yaw_toward(0, 0, 0, 1), math.pi / 2)   # +y
    assert math.isclose(yaw_toward(0, 0, -1, 0), math.pi)      # -x


def test_goal_yaw_uses_face_when_set():
    # đứng tại (-0.6,-0.26), nhìn vào lò (0.1,-0.26) -> hướng +x -> yaw 0
    room = Room("kitchen", -0.6, -0.26, 1.23, ["microwave"], face=[0.1, -0.26])
    assert math.isclose(goal_yaw(room), 0.0, abs_tol=1e-6)


def test_goal_yaw_falls_back_to_yaw_without_face():
    room = Room("bedroom", 1.0, 2.0, 0.62, ["bed"])
    assert math.isclose(goal_yaw(room), 0.62)


def test_goal_yaw_face_equal_position_falls_back():
    # face trùng vị trí -> vector 0 -> dùng yaw
    room = Room("kitchen", 0.1, -0.26, 0.5, ["microwave"], face=[0.1, -0.26])
    assert math.isclose(goal_yaw(room), 0.5)
