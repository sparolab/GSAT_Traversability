//include/supervision/col_supervision.h

#pragma once

#include <geometry_msgs/Twist.h>

struct Pose {
    double x;
    double y;
    double yaw;
};

struct Supervision{
    Pose pose;
    double pseudo_label;
};