#!/usr/bin/env bash
# StrikeRobot demo1 (python3.12 + ROS). DEMO_MOCK=1 để chạy không cần robot.
set -e
WS=/home/dsc-labs/ros2_ws
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true
export PYTHONPATH="$WS/src/vlm:$PYTHONPATH"
exec python3.12 -m webconsole.demo1.agent_server
