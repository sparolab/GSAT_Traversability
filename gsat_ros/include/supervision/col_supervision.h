//include/supervision/col_supervision.h

#pragma once

#include <geometry_msgs/msg/twist.hpp>

struct Pose {
    double x;
    double y;
    double yaw;
};

struct Supervision{
    Pose pose;
    double pseudo_label;
};