#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "transform/transform_utils.h"

class LidarOdomSyncNode
{
public:
  LidarOdomSyncNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh),
      lidar_sub_(nh_, "/os1_cloud_node/points", 1),
      odom_sub_(nh_, "/gt_odom", 6)
  {
    pnh_.param("queue_size", queue_size_, 6);

    double max_interval_sec;
    pnh_.param("max_interval_sec", max_interval_sec, 0.01);

    cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("filter/points", 1);
    odom_pub_  = nh_.advertise<nav_msgs::Odometry>("filter/odom", 10);
    global_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("global/points", 1);

    approx_sync_.reset(new message_filters::Synchronizer<ApproxPolicy>(
      ApproxPolicy(queue_size_), lidar_sub_, odom_sub_));

    approx_sync_->setMaxIntervalDuration(ros::Duration(max_interval_sec));

    approx_sync_->registerCallback(
      boost::bind(&LidarOdomSyncNode::callback, this, _1, _2));
  }

private:
  using ApproxPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::PointCloud2, nav_msgs::Odometry>;

  void callback(const sensor_msgs::PointCloud2ConstPtr& cloud,
                const nav_msgs::OdometryConstPtr& odom)
  {
    double dt = (cloud->header.stamp - odom->header.stamp).toSec();

    sensor_msgs::PointCloud2 filter_points = *cloud;
    filter_points.header.stamp = odom->header.stamp;

    cloud_pub_.publish(filter_points);
    odom_pub_.publish(*odom);

    Eigen::Isometry3d T = transform::odomToIsometry(*odom);

    sensor_msgs::PointCloud2 global_cloud = transform::transformCloudTF2(*cloud, T, "odom");

    global_cloud_pub_.publish(global_cloud);
  }

  ros::NodeHandle nh_, pnh_;

  message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub_;
  message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;

  int queue_size_{6};

  std::shared_ptr<message_filters::Synchronizer<ApproxPolicy>> approx_sync_;

  ros::Publisher cloud_pub_;
  ros::Publisher odom_pub_;

  ros::Publisher global_cloud_pub_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "lidar_odom_sync_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  LidarOdomSyncNode node(nh, pnh);
  ros::spin();
  return 0;
}
