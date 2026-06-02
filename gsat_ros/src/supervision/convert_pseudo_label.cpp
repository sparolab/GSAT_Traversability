#include "supervision/convert_pseudo_label.h"
#include <cmath>

static inline double sigmoid01(double x)
{
  return 1.0 / (1.0 + std::exp(-x));
}

double convert_label_trac_image(const Pose& pre_pose,
                               const Pose& cur_pose,
                               const geometry_msgs::Twist& cmd_vel,
                               double dt,
                               double eta,
                               double v_th)
{
  if (dt <= 1e-6) return 0.0;
  if (eta <= 0.0) eta = 1.0;

  const double dx_world = cur_pose.x - pre_pose.x;
  const double dy_world = cur_pose.y - pre_pose.y;

  const double cos_yaw = std::cos(pre_pose.yaw);
  const double sin_yaw = std::sin(pre_pose.yaw);

  const double v_x = (dx_world * cos_yaw + dy_world * sin_yaw) / dt;
  const double v_y = (-dx_world * sin_yaw + dy_world * cos_yaw) / dt;

  const double v_x_star = cmd_vel.linear.x;
  const double v_y_star = cmd_vel.linear.y;

  const double err_x = v_x - v_x_star;
  const double err_y = v_y - v_y_star;
  const double v_error = 0.5 * (err_x * err_x + err_y * err_y);

  const double sigmoid_input = -eta * (v_error - v_th);
  const double traction = sigmoid01(sigmoid_input);

  return traction;
}