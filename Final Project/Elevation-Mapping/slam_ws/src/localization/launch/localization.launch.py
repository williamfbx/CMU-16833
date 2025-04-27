from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='localization',
            executable='localization_node',
            name='localization_node'
        )
    ])
