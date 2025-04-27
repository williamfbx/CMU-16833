######
# Initially written for SLAM project by Boxiang Fu (boxiangf@andrew.cmu.edu)
# Modified by INSERT NAME HERE
######

cmake_minimum_required(VERSION 3.8)
project(localization)

# Default to C++20
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2_ros REQUIRED)

include_directories(include)

include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(localization_node 
  src/localization.cpp
  src/localization_node.cpp
)
ament_target_dependencies(localization_node
  rclcpp
  std_msgs
  sensor_msgs
  cv_bridge
  tf2_ros
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

install(TARGETS localization_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
