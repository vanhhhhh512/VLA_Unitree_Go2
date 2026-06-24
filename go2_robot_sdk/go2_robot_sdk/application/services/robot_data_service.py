# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

import logging
import math
import time
from typing import Dict, Any

from ...domain.entities import RobotData, RobotState, IMUData, OdometryData, JointData, LidarData
from ...domain.interfaces import IRobotDataPublisher
from ...domain.constants import RTC_TOPIC

logger = logging.getLogger(__name__)


class RobotDataService:
    """Service for processing and validating robot data"""

    def __init__(self, publisher: IRobotDataPublisher, lidar_max_rate: float = 15.0):
        self.publisher = publisher
        # The robot streams LiDAR voxel arrays far faster than any SLAM needs
        # (observed ~1000 Hz / ~350 MB/s), which saturates CPU/DDS and starves
        # the rest of the pipeline. Throttle to lidar_max_rate Hz; <=0 disables.
        self._lidar_min_interval = (1.0 / lidar_max_rate) if lidar_max_rate > 0 else 0.0
        self._last_lidar_time = 0.0
        # Đo tần số robot gửi voxel-map thật (trước throttle) -> biết trần nguồn dữ liệu
        self._raw_lidar_count = 0
        self._raw_log_t = time.monotonic()

    def process_webrtc_message(self, msg: Dict[str, Any], robot_id: str) -> None:
        """Process WebRTC message"""
        try:
            topic = msg.get('topic')
            robot_data = RobotData(robot_id=robot_id, timestamp=0.0)

            if topic == RTC_TOPIC["ULIDAR_ARRAY"]:
                # Mỗi frame tới đây = robot đã gửi 1 voxel-map (WASM decode chạy eager
                # ở tầng WebRTC trước đó). Đếm để biết robot gửi bao nhiêu Hz.
                self._raw_lidar_count += 1
                _t = time.monotonic()
                if _t - self._raw_log_t > 5.0:
                    try:
                        self.publisher.node.get_logger().info(
                            f"[lidar-raw] robot gui ~"
                            f"{self._raw_lidar_count / (_t - self._raw_log_t):.1f} Hz (truoc throttle)")
                    except Exception:
                        pass
                    self._raw_lidar_count = 0
                    self._raw_log_t = _t
                # Drop excess frames BEFORE the expensive WASM voxel decode.
                if self._lidar_min_interval > 0.0:
                    now = time.monotonic()
                    if (now - self._last_lidar_time) < self._lidar_min_interval:
                        return
                    self._last_lidar_time = now
                self._process_lidar_data(msg, robot_data)
                self.publisher.publish_lidar_data(robot_data)
                self.publisher.publish_voxel_data(robot_data)

            elif topic == RTC_TOPIC["ROBOTODOM"]:
                self._process_odometry_data(msg, robot_data)
                self.publisher.publish_odometry(robot_data)

            elif topic == RTC_TOPIC["LF_SPORT_MOD_STATE"]:
                self._process_sport_mode_state(msg, robot_data)
                self.publisher.publish_robot_state(robot_data)

            elif topic == RTC_TOPIC["LOW_STATE"]:
                self._process_low_state(msg, robot_data)
                self.publisher.publish_joint_state(robot_data)

        except Exception as e:
            logger.error(f"Error processing WebRTC message: {e}")

    def _process_lidar_data(self, msg: Dict[str, Any], robot_data: RobotData) -> None:
        """Process lidar data"""
        try:
            decoded_data = msg.get("decoded_data", {})
            data = msg.get("data", {})
            
            robot_data.lidar_data = LidarData(
                positions=decoded_data.get("positions"),
                uvs=decoded_data.get("uvs"),
                resolution=data.get("resolution", 0.0),
                origin=data.get("origin", [0.0, 0.0, 0.0]),
                stamp=data.get("stamp", 0.0),
                width=data.get("width"),
                src_size=data.get("src_size"),
                compressed_data=msg.get("compressed_data")
            )
        except Exception as e:
            logger.error(f"Error processing lidar data: {e}")

    def _process_odometry_data(self, msg: Dict[str, Any], robot_data: RobotData) -> None:
        """Process odometry data"""
        try:
            pose_data = msg['data']['pose']
            position = pose_data['position']
            orientation = pose_data['orientation']

            # Data validation
            pos_vals = [position['x'], position['y'], position['z']]
            rot_vals = [orientation['x'], orientation['y'], orientation['z'], orientation['w']]

            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pos_vals + rot_vals):
                logger.warning("Invalid odometry data - skipping")
                return

            robot_data.odometry_data = OdometryData(
                position=position,
                orientation=orientation
            )
        except Exception as e:
            logger.error(f"Error processing odometry data: {e}")

    def _process_sport_mode_state(self, msg: Dict[str, Any], robot_data: RobotData) -> None:
        """Process sport mode state"""
        try:
            data = msg["data"]

            # Data validation
            if not self._validate_float_list(data.get("position", [])):
                return
            if not self._validate_float_list(data.get("range_obstacle", [])):
                return
            if not self._validate_float_list(data.get("foot_position_body", [])):
                return
            if not self._validate_float_list(data.get("foot_speed_body", [])):
                return
            if not self._validate_float(data.get("body_height")):
                return

            robot_data.robot_state = RobotState(
                mode=data["mode"],
                progress=data["progress"],
                gait_type=data["gait_type"],
                position=data["position"],
                body_height=data["body_height"],
                velocity=data["velocity"],
                range_obstacle=data["range_obstacle"],
                foot_force=data["foot_force"],
                foot_position_body=data["foot_position_body"],
                foot_speed_body=data["foot_speed_body"]
            )

            # Process IMU data
            imu_data = data["imu_state"]
            if (self._validate_float_list(imu_data.get("quaternion", [])) and
                self._validate_float_list(imu_data.get("accelerometer", [])) and
                self._validate_float_list(imu_data.get("gyroscope", [])) and
                self._validate_float_list(imu_data.get("rpy", []))):
                
                robot_data.imu_data = IMUData(
                    quaternion=imu_data["quaternion"],
                    accelerometer=imu_data["accelerometer"],
                    gyroscope=imu_data["gyroscope"],
                    rpy=imu_data["rpy"],
                    temperature=imu_data["temperature"]
                )

        except Exception as e:
            logger.error(f"Error processing sport mode state: {e}")

    def _process_low_state(self, msg: Dict[str, Any], robot_data: RobotData) -> None:
        """Process low state data"""
        try:
            low_state_data = msg['data']
            robot_data.joint_data = JointData(
                motor_state=low_state_data['motor_state']
            )
        except Exception as e:
            logger.error(f"Error processing low state: {e}")

    def _validate_float_list(self, data: list) -> bool:
        """Validate a list of float values"""
        return all(isinstance(x, (int, float)) and math.isfinite(x) for x in data)

    def _validate_float(self, value: Any) -> bool:
        """Validate a float value"""
        return isinstance(value, (int, float)) and math.isfinite(value) 