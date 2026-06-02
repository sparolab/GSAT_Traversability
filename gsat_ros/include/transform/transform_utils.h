// transform_utils.h
#pragma once

#include <string>
#include <Eigen/Geometry>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>

namespace transform
{
    Eigen::Isometry3d odomToIsometry(const nav_msgs::Odometry& odom);

    sensor_msgs::PointCloud2 transformCloudTF2(const sensor_msgs::PointCloud2& cloud,
                                            const Eigen::Isometry3d& T,
                                            const std::string& target_frame);
}