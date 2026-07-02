#!/usr/bin/env bash
# Khởi động VLM Web Console cho Unitree Go2 (real mode: python3.12 + ROS).
#
# Dùng:
#   ROBOT_IP=192.168.123.161 ./run_vlm_console.sh
#   ./run_vlm_console.sh 192.168.123.161        # truyền IP qua tham số
#   ./run_vlm_console.sh                         # không IP -> MOCK MODE
#
# Mở http://localhost:8000
set -e

WS=/home/dsc-labs/ros2_ws

# 1) Môi trường ROS (cần cho go2_robot_sdk / go2_interfaces / rclpy)
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash" 2>/dev/null || true

# 2) Cho phép import package webconsole (nằm trong src/vlm) mà không bị bản
#    'vlm' đã build trong install/ che mất.
export PYTHONPATH="$WS/src/vlm:$PYTHONPATH"

# 3) ROBOT_IP từ tham số thứ nhất nếu có
if [ -n "$1" ]; then
  export ROBOT_IP="$1"
fi

exec python3.12 -m webconsole.server
