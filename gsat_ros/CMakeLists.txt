cmake_minimum_required(VERSION 2.8.3)
project(gsat_ros)
add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  grid_map_msgs
  grid_map_ros
  grid_map_visualization
  message_filters
  pcl_conversions
  tf2
  tf2_sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES transform_pkg supervision
  CATKIN_DEPENDS
    geometry_msgs nav_msgs roscpp sensor_msgs std_msgs
    grid_map_msgs grid_map_ros grid_map_visualization
    message_filters pcl_conversions tf2 tf2_sensor_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(transform_pkg
  src/transform/transform_utils.cpp
)
add_library(supervision
  src/supervision/convert_pseudo_label.cpp
)

add_executable(sw_sync src/sw_sync.cpp)
target_link_libraries(sw_sync
  transform_pkg
  ${catkin_LIBRARIES}
)


add_executable(data_collection src/data_collection.cpp)
target_link_libraries(data_collection
  supervision
  ${catkin_LIBRARIES}
)