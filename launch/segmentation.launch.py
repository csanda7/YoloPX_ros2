from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolopx',
            executable='seg_sub_trt',  # ugyanaz, ami ros2 run-nál működik
            name='yolopx_trt',
            output='screen',
            emulate_tty=True
        )
    ])
