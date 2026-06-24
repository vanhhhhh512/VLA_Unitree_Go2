"""Nguồn frame: WebRTC tới Go2 hoặc mock mode."""
import sys
import asyncio
import json

import numpy as np
import cv2

sys.path.insert(0, "/home/dsc-labs/ros2_ws/src/go2_robot_sdk")


def _make_text_frame(text, color=(0, 180, 255)):
    frame = np.full((480, 640, 3), 30, dtype=np.uint8)
    cv2.putText(frame, text, (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame


def _make_mock_frame():
    return _make_text_frame("MOCK MODE - no ROBOT_IP")


class FrameSource:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        self._latest = None
        self._connected = False
        self._conn = None

    @property
    def is_mock(self):
        return not self.robot_ip

    @property
    def is_connected(self):
        return self._connected

    def get_latest_frame(self):
        return self._latest

    async def start(self):
        if self.is_mock:
            self._latest = _make_mock_frame()
            return
        try:
            await self._connect_webrtc()
        except Exception as e:
            # Robot tắt / sai IP / khác mạng: KHÔNG làm sập server.
            # Vẫn mở GUI, badge sẽ là DISCONNECTED.
            print(f"[FrameSource] Không kết nối được robot {self.robot_ip}: {e}")
            self._connected = False
            self._latest = _make_text_frame(
                f"KHONG KET NOI ROBOT {self.robot_ip}", color=(80, 80, 255)
            )

    async def _connect_webrtc(self):
        from go2_robot_sdk.infrastructure.webrtc.go2_connection import Go2Connection
        from go2_robot_sdk.domain.constants import RTC_TOPIC

        async def on_video_frame(track, robot_id):
            while True:
                try:
                    frame = await track.recv()
                    self._latest = frame.to_ndarray(format="bgr24")
                except Exception as e:
                    print(f"[FrameSource] video stream closed: {e}")
                    break

        def on_validated(robot_num):
            self._connected = True
            asyncio.create_task(self._conn.disableTrafficSaving(True))
            try:
                for topic in RTC_TOPIC.values():
                    self._conn.data_channel.send(
                        json.dumps({"type": "subscribe", "topic": topic})
                    )
            except Exception as e:
                print(f"[FrameSource] subscribe failed: {e}")

        self._conn = Go2Connection(
            robot_ip=self.robot_ip, robot_num=0, token="",
            on_validated=on_validated, on_video_frame=on_video_frame,
            decode_lidar=False,
        )
        await self._conn.connect()

    async def stop(self):
        self._connected = False
        if self._conn is not None:
            await self._conn.disconnect()
