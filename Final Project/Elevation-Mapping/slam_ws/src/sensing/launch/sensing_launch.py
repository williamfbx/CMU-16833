from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def generate_launch_description():
    
    camera_model_arg = DeclareLaunchArgument(
        "camera_model",
        default_value="zed2i",
        description="Model of the ZED camera"
    )

    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('zed_wrapper'), 'launch/zed_camera.launch.py')
        ),
        launch_arguments={"camera_model": LaunchConfiguration("camera_model")}.items()
    )

    # Static transform parameters
    zed_static_tf_params_path = os.path.join(
        get_package_share_directory('sensing'),
        'config',
        'zed_static_tf_params.yaml'
    )
    with open(zed_static_tf_params_path, 'r') as stream:
      zed_static_tf_params = yaml.load(
          stream,
          Loader=yaml.SafeLoader
      )['zed_static_transform_publisher']['ros__parameters']

    zed_static_tf_node = Node(
          package='tf2_ros',
          executable='static_transform_publisher',
          arguments=[
              zed_static_tf_params['x'], zed_static_tf_params['y'], zed_static_tf_params['z'],
              zed_static_tf_params['yaw'], zed_static_tf_params['pitch'], zed_static_tf_params['roll'],
              zed_static_tf_params['parent_frame_id'], zed_static_tf_params['child_frame_id']
          ]
      )

    # realsense_launch = IncludeLaunchDescription(
    #   PythonLaunchDescriptionSource(
    #     os.path.join(get_package_share_directory('realsense'), 'launch/realsense_launch.py')
    #     )
    # )

    return LaunchDescription([
        camera_model_arg,
        zed_launch,
        zed_static_tf_node,
        # realsense_launch,
    ])