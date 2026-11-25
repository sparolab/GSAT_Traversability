// transform_utils.h
#pragma once

#include <string>
#include <Eigen/Geometry>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace transform
{
    Eigen::Isometry3d odomToIsometry(const nav_msgs::msg::Odometry& odom);

    sensor_msgs::msg::PointCloud2 transformCloudTF2(const sensor_msgs::msg::PointCloud2& cloud,
                                                    const Eigen::Isometry3d& T,
                                                    const std::string& target_frame);
}
