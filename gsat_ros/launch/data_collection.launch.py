"""GSAT data collection (ROS 2): sw_sync (LiDAR + odometry sync) and the data collection node.
ros2 launch gsat_ros data_collection.launch.py [config:=/path/to/data_collection.yaml] [use_sim_time:=true]"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_cfg = os.path.join(get_package_share_directory('gsat_ros'), 'config', 'data_collection.yaml')
    cfg = LaunchConfiguration('config')
    sim = {'use_sim_time': LaunchConfiguration('use_sim_time')}
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default_cfg),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(package='gsat_ros', executable='sw_sync', name='data_sw_sync', output='screen', parameters=[cfg, sim]),
        Node(package='gsat_ros', executable='data_collection', name='data_collection_node', output='screen',
             parameters=[cfg, sim]),
    ])
