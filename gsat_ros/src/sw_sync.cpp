#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "transform/transform_utils.h"

class LidarOdomSyncNode : public rclcpp::Node
{
public:
  LidarOdomSyncNode()
    : Node("lidar_odom_sync_node")
  {
    // Input topics were hard-coded in the ROS 1 node; same defaults, now parameters.
    const std::string lidar_topic = declare_parameter<std::string>("lidar_topic", "/os1_cloud_node/points");
    const std::string odom_topic  = declare_parameter<std::string>("odom_topic", "/gt_odom");
    queue_size_ = declare_parameter<int>("queue_size", 6);
    const double max_interval_sec = declare_parameter<double>("max_interval_sec", 0.01);

    lidar_sub_.subscribe(this, lidar_topic, rclcpp::SensorDataQoS().get_rmw_qos_profile());
    odom_sub_.subscribe(this, odom_topic);

    cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("filter/points", 1);
    odom_pub_  = create_publisher<nav_msgs::msg::Odometry>("filter/odom", 10);
    global_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("global/points", 1);

    approx_sync_ = std::make_shared<message_filters::Synchronizer<ApproxPolicy>>(
      ApproxPolicy(queue_size_), lidar_sub_, odom_sub_);

    approx_sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(max_interval_sec));

    approx_sync_->registerCallback(
      std::bind(&LidarOdomSyncNode::callback, this, std::placeholders::_1, std::placeholders::_2));
  }

private:
  using ApproxPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::PointCloud2, nav_msgs::msg::Odometry>;

  void callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud,
                const nav_msgs::msg::Odometry::ConstSharedPtr& odom)
  {
    sensor_msgs::msg::PointCloud2 filter_points = *cloud;
    filter_points.header.stamp = odom->header.stamp;

    cloud_pub_->publish(filter_points);
    odom_pub_->publish(*odom);

    Eigen::Isometry3d T = transform::odomToIsometry(*odom);

    sensor_msgs::msg::PointCloud2 global_cloud = transform::transformCloudTF2(*cloud, T, "odom");

    global_cloud_pub_->publish(global_cloud);
  }

  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> lidar_sub_;
  message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;

  int queue_size_{6};

  std::shared_ptr<message_filters::Synchronizer<ApproxPolicy>> approx_sync_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_cloud_pub_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarOdomSyncNode>());
  rclcpp::shutdown();
  return 0;
}
