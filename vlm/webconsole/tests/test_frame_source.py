import asyncio
import numpy as np
from vlm.webconsole.frame_source import FrameSource


def test_mock_mode_when_no_ip():
    src = FrameSource(robot_ip=None)
    assert src.is_mock is True
    assert src.is_connected is False


def test_mock_start_provides_frame():
    src = FrameSource(robot_ip="")
    asyncio.run(src.start())
    frame = src.get_latest_frame()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3 and frame.shape[2] == 3


def test_real_mode_no_frame_before_start():
    src = FrameSource(robot_ip="192.168.1.10")
    assert src.is_mock is False
    assert src.get_latest_frame() is None
    assert src.is_connected is False


def test_failed_connection_does_not_crash():
    # IP không nối được (hoặc thiếu go2 sdk) -> start() không raise,
    # vẫn có frame thông báo, badge disconnected.
    src = FrameSource(robot_ip="10.255.255.1")
    asyncio.run(src.start())
    assert src.is_connected is False
    frame = src.get_latest_frame()
    assert isinstance(frame, np.ndarray)
