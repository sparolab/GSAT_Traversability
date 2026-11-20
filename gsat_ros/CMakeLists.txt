cmake_minimum_required(VERSION 3.8)
project(gsat_ros)
set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include ${EIGEN3_INCLUDE_DIR})

add_library(transform_pkg src/transform/transform_utils.cpp)
ament_target_dependencies(transform_pkg nav_msgs sensor_msgs geometry_msgs tf2 tf2_sensor_msgs)

add_library(supervision src/supervision/convert_pseudo_label.cpp)
ament_target_dependencies(supervision geometry_msgs)

add_executable(sw_sync src/sw_sync.cpp)
target_link_libraries(sw_sync transform_pkg)
ament_target_dependencies(sw_sync rclcpp sensor_msgs nav_msgs message_filters)

add_executable(data_collection src/data_collection.cpp)
target_link_libraries(data_collection supervision)
ament_target_dependencies(data_collection rclcpp sensor_msgs nav_msgs geometry_msgs message_filters tf2 tf2_geometry_msgs)

install(TARGETS transform_pkg supervision sw_sync data_collection
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
ament_package()
