//include/supervision/convert_pseudo_label.h

#pragma once

#include <supervision/col_supervision.h>

double convert_label_trac_image(const Pose& pre_pose,
                                const Pose& cur_pose,
                                const geometry_msgs::Twist& cmd_vel,
                                double dt,
                                double eta,
                                double v_th);
