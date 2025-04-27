# Elevation Mapping
This is intended for submission as the final course project for 16-833: Robot Localization and Mapping at CMU

## Get Started

### Setup drivers

For this project, we used a ZED 2i stereo camera. Make sure the prerequisites on this page are installed:
https://www.stereolabs.com/docs/ros2

Once the prerequisites are installed, run the following script (changing `ros2_ws` to your desired workspace):

```
cd ~/ros2_ws/src/
git clone https://github.com/stereolabs/zed-ros2-wrapper.git
cd ..
sudo apt update
# Install the required dependencies
rosdep install --from-paths src --ignore-src -r -y
# Build the wrapper
colcon build
```

We will also need to install the ROS packages to visualize in RViz:

```
cd ~/ros2_ws/src/
git clone https://github.com/stereolabs/zed-ros2-examples.git
cd ../
sudo apt update
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build
```

For 3D visualization, we use the RViz plugins from Elevation Mapping:
1. `grid_map`
```
git clone https://github.com/anybotics/grid_map.git --branch humble
cd ../
rosdep install -y --ignore-src --from-paths src
colcon build
```

2. `kindr`
```
git clone https://github.com/ANYbotics/kindr.git
cd kindr/
mkdir build
cd build
cmake .. -DUSE_CMAKE=true
sudo make install
```

3. `kindr_ros`
```
git clone -b galactic https://github.com/SivertHavso/kindr_ros.git
cd ../
rosdep install -y --ignore-src --from-paths src
colcon build
```

A RealSense camera can alternatively be used. However, you will need to implement a seperate localization source. The following script installs the required RealSense libaries and wrapper dependencies
```
cd ~/ros2_ws/src/
sudo apt install ros-humble-librealsense2* # Libaries
git clone https://github.com/IntelRealSense/realsense-ros.git -b ros2-master # Wrapper

cd ~/ros2_ws
sudo apt-get install python3-rosdep -y
sudo rosdep init # "sudo rosdep init --include-eol-distros" for Foxy and earlier
rosdep update # "sudo rosdep update --include-eol-distros" for Foxy and earlier
rosdep install -i --from-path src --rosdistro $ROS_DISTRO --skip-keys=librealsense2 -y

colcon build
```

## How to Launch
```
cd ~/slam_ws/
colcon build
source install/setup.{bash/zsh/etc}
ros2 launch launcher launch.py
```

## Common Issues
If the depth viewer is not interfacing correctly with your device, it may be because OpenGL is not set up correctly. This article shows a solution to this issue:
https://support.stereolabs.com/hc/en-us/articles/8422008229143-How-can-I-solve-ZED-SDK-OpenGL-issues-under-Ubuntu

When running `colcon build` on `kindr_msgs`, some built-in interfaces may fail to be located. If this is the case, edit the `CMakeLists.txt` file and include the following dependencies under `rosidl_generate_interfaces`:
```
rosidl_generate_interfaces(${PROJECT_NAME}
	"msg/VectorAtPosition.msg"
	ADD_LINTER_TESTS
	DEPENDENCIES builtin_interfaces std_msgs
	DEPENDENCIES builtin_interfaces geometry_msgs
)
```
