from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os
import yaml

def generate_launch_description():
  included_launch = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
      os.path.join(get_package_share_directory('realsense2_camera'), 'examples/launch_params_from_file/rs_launch_get_params_from_yaml.py')
    ), 
  )  

  # Static transform parameters
  realsense_static_tf_params_path = os.path.join(
      get_package_share_directory('realsense'),
      'config',
      'realsense_static_tf_params.yaml'
  )
  with open(realsense_static_tf_params_path, 'r') as stream:
    realsense_static_tf_params = yaml.load(
        stream,
        Loader=yaml.SafeLoader
    )['realsense_static_transform_publisher']['ros__parameters']

  # Return the launch file
  return LaunchDescription([
      included_launch,
      # Static tf publisher
      # Order of args is [x, y, z, yaw, pitch, roll, parent_frame, child_frame]
      # http://docs.ros.org/en/galactic/Tutorials/Tf2/Writing-A-Tf2-Static-Broadcaster-Cpp.html#the-proper-way-to-publish-static-transforms
      Node(
          package='tf2_ros',
          executable='static_transform_publisher',
          arguments=[
              realsense_static_tf_params['x'], realsense_static_tf_params['y'], realsense_static_tf_params['z'],
              realsense_static_tf_params['yaw'], realsense_static_tf_params['pitch'], realsense_static_tf_params['roll'],
              realsense_static_tf_params['parent_frame_id'], realsense_static_tf_params['child_frame_id']
          ]
      )
  ])

