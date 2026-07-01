#!/usr/bin/env bash
# StrikeRobot demo1 (python3.12 + ROS). DEMO_MOCK=1 để chạy không cần robot.
set -e
WS=/home/dsc-labs/ros2_vlm
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true
export PYTHONPATH="$WS/src/vlm:$PYTHONPATH"
# Mặc định SERVO (điều khiển liên tục: TIẾN ngay khi |lệch| ≤ center_tol, không "turn mãi"
# như VLM rời rạc). Đặt VLA_CONTROL=vlm trước khi chạy nếu muốn thuần VLM.
export VLA_CONTROL="${VLA_CONTROL:-servo}"
exec python3.12 -m webconsole.demo1.agent_server
