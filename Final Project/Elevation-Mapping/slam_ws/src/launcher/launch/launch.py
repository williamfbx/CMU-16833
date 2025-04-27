from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os

def generate_launch_description():

    sensing_launch = IncludeLaunchDescription(
      PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory('sensing'), 'launch/sensing_launch.py')
        )
    )

    localization_launch = IncludeLaunchDescription(
      PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory('localization'), 'launch/localization.launch.py')
        )
    )

    mapping_launch = IncludeLaunchDescription(
      PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory('mapping'), 'launch/mapping.launch.py')
        )
    )

    visualization_launch = IncludeLaunchDescription(
      PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory('visualization'), 'launch/visualization.launch.py')
        )
    )

    rviz_config_path = os.path.join(get_package_share_directory('launcher'), 'launch/gridmap.rviz')


    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )

    return LaunchDescription([
        sensing_launch,
        localization_launch,
        mapping_launch,
        visualization_launch,
        rviz_node,
    ])