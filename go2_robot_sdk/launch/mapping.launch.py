# Mapping launch file - optimized for SLAM and map creation
# Usage: ros2 launch go2_robot_sdk mapping.launch.py

import os
from typing import List
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import FrontendLaunchDescriptionSource, PythonLaunchDescriptionSource


def generate_launch_description():
    """Generate launch description for Go2 mapping mode"""
    
    # Environment variables
    robot_token = os.getenv('ROBOT_TOKEN', 'af4195d67dd4d585f161f7e0932c2aa8')
    robot_ip = os.getenv('ROBOT_IP', '')
    robot_ip_list = robot_ip.replace(" ", "").split(",") if robot_ip else []
    map_name = os.getenv('MAP_NAME', 'my_map')
    save_map = os.getenv('MAP_SAVE', 'true')
    conn_type = os.getenv('CONN_TYPE', 'webrtc')
    
    # Determine connection mode
    conn_mode = "single" if len(robot_ip_list) == 1 and conn_type != "cyclonedds" else "multi"
    
    # Package paths
    package_dir = get_package_share_directory('go2_robot_sdk')
    urdf_file = 'go2.urdf' if conn_mode == 'single' else 'multi_go2.urdf'
    rviz_config = 'single_robot_conf.rviz' if conn_mode == 'single' else 'multi_robot_conf.rviz'
    
    config_paths = {
        'joystick': os.path.join(package_dir, 'config', 'joystick.yaml'),
        'twist_mux': os.path.join(package_dir, 'config', 'twist_mux.yaml'),
        'slam': os.path.join(package_dir, 'config', 'mapper_params_online_async.yaml'),
        'rviz': os.path.join(package_dir, 'config', rviz_config),
        'urdf': os.path.join(package_dir, 'urdf', urdf_file),
    }
    
    print(f"🗺️  Go2 Mapping Mode:")
    print(f"   Robot IPs: {robot_ip_list}")
    print(f"   Connection: {conn_type} ({conn_mode})")
    print(f"   Map name: {map_name}")
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    with_rviz = LaunchConfiguration('rviz', default='true')
    with_foxglove = LaunchConfiguration('foxglove', default='true')
    with_joystick = LaunchConfiguration('joystick', default='true')
    
    launch_args = [
        DeclareLaunchArgument('rviz', default_value='true', description='Launch RViz2'),
        DeclareLaunchArgument('foxglove', default_value='true', description='Launch Foxglove Bridge'),
        DeclareLaunchArgument('joystick', default_value='true', description='Launch joystick control'),
    ]
    
    # Load URDF
    with open(config_paths['urdf'], 'r') as file:
        robot_desc = file.read()
    
    # Core nodes
    core_nodes = [
        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='go2_robot_state_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'robot_description': robot_desc
            }],
        ),
        # Main robot driver
        Node(
            package='go2_robot_sdk',
            executable='go2_driver_node',
            name='go2_driver_node',
            output='screen',
            parameters=[{
                'robot_ip': robot_ip,
                'token': robot_token,
                'conn_type': conn_type,
                'lidar_publish_rate': 15.0,  # chặn firehose LiDAR (robot bắn ~1000Hz)
                'lidar_voxel_size': 0.06,    # voxel 6cm ở driver: giảm điểm -> nhẹ CPU/băng thông hơn nữa
            }],
        ),
        # LƯU Ý: lidar_to_pointcloud (gộp cloud đa-robot) chỉ thêm ở MULTI mode,
        # xem khối `if conn_mode != 'single'` bên dưới. Ở single mode driver đã
        # publish /point_cloud2 trực tiếp -> thêm node này sẽ tự-lặp (sub+pub cùng
        # /point_cloud2) gây firehose ~126Hz/227MB/s, nghẽn CPU, đứng /scan.
        # Point cloud aggregator - maximized for full coverage
        Node(
    package='lidar_processor_cpp',
    executable='pointcloud_aggregator_node',
    name='pointcloud_aggregator',
    remappings=[
        ('cloud_in', '/point_cloud2'),  # single: nhận thẳng từ driver; multi: từ lidar_to_pointcloud
    ],
    parameters=[{
        'max_range': 10.0,
        'min_range': 0.15,
        'height_filter_min': 0.2,   # chỉ giữ lát 0.2-0.4m (z trong frame odom ~ so với mặt đất)
        'height_filter_max': 0.4,
        'downsample_rate': 1,
        'publish_rate': 20.0,
        'voxel_leaf_size': 0.05,    # 5cm: single mode bỏ lidar_to_pointcloud nên phải voxel ở đây
        'sor_enable': True,         # tắt (False) nếu cần giảm CPU thêm
        'sor_mean_k': 12,           # nhẹ hơn mặc định cũ (30)
        'sor_std_dev': 1.0,
        'ror_enable': True,         # lọc điểm lẻ loi: quanh nó < ror_min_neighbors điểm thì bỏ
        'ror_radius': 0.15,         # bán kính tìm hàng xóm (m)
        'ror_min_neighbors': 3,     # cần >=3 điểm trong bán kính, không thì coi là nhiễu -> bỏ
    }],
),
        # PointCloud to LaserScan converter - maximum coverage
        Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='go2_pointcloud_to_laserscan',
            remappings=[
                ('cloud_in', '/pointcloud/filtered'),
                ('scan', '/scan'),
            ],
            parameters=[{
                'target_frame': 'base_link',
                'max_height': 2.0,   # để rộng: lọc chiều cao đã làm ở aggregator (lát 0.2-0.4m)
                'min_height': -1.0,  # tránh cắt 2 lần lệch frame (laserscan lọc theo base_link)
                'angle_min': -3.14159,
                'angle_max': 3.14159,
                'angle_increment': 0.00872665,
                'scan_time': 0.1,
                'range_min': 0.35,   # loại chân/thân robot (chân Go2 ~0.2-0.3m) khỏi scan -> hết đốm ảo quanh robot
                'range_max': 10.0,
                'use_inf': True,
                'concurrency_level': 2,
            }],
            output='screen',
        ),
    ]

    # Chỉ MULTI mode mới cần gộp các cloud robot{i}/point_cloud2 -> /point_cloud2.
    # Single mode: bỏ qua (driver đã publish /point_cloud2; thêm vào sẽ tự-lặp).
    if conn_mode != 'single':
        core_nodes.append(
            Node(
                package='lidar_processor_cpp',
                executable='lidar_to_pointcloud_node',
                name='lidar_to_pointcloud',
                parameters=[{
                    'robot_ip_lst': robot_ip_list,
                    'map_name': map_name,
                    'map_save': save_map,
                }],
            )
        )

    # Teleop nodes
    teleop_nodes = [
        Node(
            package='joy',
            executable='joy_node',
            condition=IfCondition(with_joystick),
            parameters=[config_paths['joystick']]
        ),
        Node(
            package='teleop_twist_joy',
            executable='teleop_node',
            name='go2_teleop_node',
            condition=IfCondition(with_joystick),
            parameters=[config_paths['twist_mux']],
        ),
        Node(
            package='twist_mux',
            executable='twist_mux',
            output='screen',
            condition=IfCondition(with_joystick),
            parameters=[
                {'use_sim_time': use_sim_time},
                config_paths['twist_mux']
            ],
        ),
    ]
    
    # Visualization nodes
    viz_nodes = [
        Node(
            package='rviz2',
            executable='rviz2',
            condition=IfCondition(with_rviz),
            name='go2_rviz2',
            output='screen',
            arguments=['-d', config_paths['rviz']],
            parameters=[{'use_sim_time': False}]
        ),
    ]
    
    # Include launches
    foxglove_launch = os.path.join(
        get_package_share_directory('foxglove_bridge'),
        'launch', 'foxglove_bridge_launch.xml'
    )
    
    include_launches = [
        # Foxglove Bridge
        IncludeLaunchDescription(
            FrontendLaunchDescriptionSource(foxglove_launch),
            condition=IfCondition(with_foxglove),
        ),
        # SLAM Toolbox for mapping
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('slam_toolbox'),
                            'launch', 'online_async_launch.py')
            ]),
            launch_arguments={
                'slam_params_file': config_paths['slam'],
                'use_sim_time': use_sim_time,
            }.items(),
        ),
    ]
    
    return LaunchDescription(
        launch_args +
        core_nodes +
        teleop_nodes +
        viz_nodes +
        include_launches
    )