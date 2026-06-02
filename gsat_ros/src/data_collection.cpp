// Data_collection_node

#include <ros/ros.h>

#include <supervision/col_supervision.h>
#include <supervision/convert_pseudo_label.h>

#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <sensor_msgs/point_cloud2_iterator.h>
#include <iomanip>
#include <sstream>

class Data_generate
{
public:
  using ApproxPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::PointCloud2, nav_msgs::Odometry>;

  Data_generate(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : nh_(nh), pnh_(pnh)
  {
    std::string dataset_name;
    pnh_.param("dataset_name", dataset_name, std::string("gazebo_hill"));
    const std::string prefix = dataset_name + "/";

    pnh_.param(prefix + "output_dir", output_dir_,
               std::string("/root/gsat_ws/src/collect_data/gazebo/hill/original"));
    pnh_.param(prefix + "sampling_time", sampling_time_, 0.3);
    pnh_.param(prefix + "queue_size", queue_size_, 10);
    pnh_.param(prefix + "max_interval_sec", max_interval_sec_, 0.02);
    pnh_.param(prefix + "use_pseudo_label", use_pseudo_label_, true);
    pnh_.param(prefix + "eta", eta_, 20.0);
    pnh_.param(prefix + "v_th", v_th_, 0.05);

    std::string lidar_topic, odom_topic, cmd_vel_topic;
    pnh_.param(prefix + "lidar_topic", lidar_topic, std::string("filter/points"));
    pnh_.param(prefix + "odom_topic", odom_topic, std::string("filter/odom"));
    pnh_.param(prefix + "cmd_vel_topic", cmd_vel_topic, std::string("cmd_vel"));

    lidar_dir_ = output_dir_ + "/lidar";
    csv_path_  = output_dir_ + "/supervision.csv";

    std::filesystem::create_directories(output_dir_);
    std::filesystem::create_directories(lidar_dir_);

    if (std::filesystem::exists(csv_path_))
    {
      std::error_code ec;
      std::filesystem::remove(csv_path_, ec);
    }

    csv_.open(csv_path_, std::ios::out);

    csv_ << "TIMESTAMP,"
         << "robot_posi_x,"
         << "robot_posi_y,"
         << "robot_posi_z,"
         << "robot_ori_x,"
         << "robot_ori_y,"
         << "robot_ori_z,"
         << "robot_ori_w,"
         << "Travel_label\n";
    csv_.flush();

    ROS_INFO_STREAM("Dataset: " << dataset_name);
    ROS_INFO_STREAM("Output dir: " << output_dir_);
    ROS_INFO_STREAM("CSV path  : " << csv_path_);
    ROS_INFO_STREAM("Sampling  : " << sampling_time_ << " sec");
    ROS_INFO_STREAM("Use pseudo label: " << (use_pseudo_label_ ? "true" : "false (Travel_label=1)"));
    if (use_pseudo_label_)
      ROS_INFO_STREAM("Label eta: " << eta_ << ", v_th: " << v_th_);
    ROS_INFO_STREAM("Topics    : " << lidar_topic << ", " << odom_topic << ", " << cmd_vel_topic);

    lidar_sub_.reset(new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh_, lidar_topic, 1));
    odom_sub_.reset(new message_filters::Subscriber<nav_msgs::Odometry>(nh_, odom_topic, 1));
    cmd_sub_ = nh_.subscribe(cmd_vel_topic, 10, &Data_generate::cmdCallback, this);

    sync_.reset(new message_filters::Synchronizer<ApproxPolicy>(
        ApproxPolicy(queue_size_), *lidar_sub_, *odom_sub_));

    sync_->setMaxIntervalDuration(ros::Duration(max_interval_sec_));
    sync_->registerCallback(boost::bind(&Data_generate::generate_callback, this, _1, _2));
  }

  ~Data_generate()
  {
    if (csv_.is_open())
      csv_.close();
  }

private:
  Pose convert_odom(const nav_msgs::OdometryConstPtr& odom)
  {
    Pose p;
    p.x = odom->pose.pose.position.x;
    p.y = odom->pose.pose.position.y;

    tf2::Quaternion q;
    tf2::fromMsg(odom->pose.pose.orientation, q);
    double roll, pitch;
    tf2::Matrix3x3(q).getRPY(roll, pitch, p.yaw);
    return p;
  }

  bool saveCloudBinXYI(const sensor_msgs::PointCloud2ConstPtr& cloud,
                      const std::string& bin_path)
  {
    if (!cloud || cloud->width * cloud->height == 0) return false;

    const size_t N = static_cast<size_t>(cloud->width) * static_cast<size_t>(cloud->height);

    bool has_intensity = false;
    for (const auto& f : cloud->fields) {
      if (f.name == "intensity") { has_intensity = true; break; }
    }

    std::ofstream out(bin_path, std::ios::binary);
    if (!out.is_open()) {
      ROS_ERROR_STREAM("Failed to open bin: " << bin_path);
      return false;
    }
    sensor_msgs::PointCloud2ConstIterator<float> it_x(*cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> it_y(*cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> it_z(*cloud, "z");

    if (has_intensity) {
      sensor_msgs::PointCloud2ConstIterator<float> it_i(*cloud, "intensity");
      for (size_t k = 0; k < N; ++k, ++it_x, ++it_y, ++it_z, ++it_i) {
        float v[4] = {*it_x, *it_y, *it_z, *it_i};
        out.write(reinterpret_cast<const char*>(v), sizeof(v));
      }
    } else {
      for (size_t k = 0; k < N; ++k, ++it_x, ++it_y, ++it_z) {
        float v[4] = {*it_x, *it_y, *it_z, 0.0f};
        out.write(reinterpret_cast<const char*>(v), sizeof(v));
      }
    }

    out.close();
    return true;
  }


  void cmdCallback(const geometry_msgs::TwistConstPtr& msg)
  {
    command_vel_ = *msg;  // latest command
  }

  void generate_callback(const sensor_msgs::PointCloud2ConstPtr& cloud,
                         const nav_msgs::OdometryConstPtr& odom)
  {
    if (!csv_.is_open())
      return;

    ros::Time now = ros::Time::now();
    cur_pose_ = convert_odom(odom);

    if (!initialized_)
    {
      last_time_   = now;
      initialized_ = true;
      pre_pose_    = cur_pose_;
      return;  // first frame: no dt
    }

    double dt = (now - last_time_).toSec();
    if (dt < sampling_time_)
      return;


    if (std::abs(command_vel_.linear.x) < 1e-6)
      return;

    //--------[Pseudo label from cmd_vel vs odom, or fixed 1.0]--------#
    const double travel_label = use_pseudo_label_
        ? convert_label_trac_image(pre_pose_, cur_pose_, command_vel_, dt, eta_, v_th_)
        : 1.0;

    const uint32_t sec  = odom->header.stamp.sec;
    const uint32_t nsec = odom->header.stamp.nsec;
    
    uint64_t timestamp = sec * 1000000000ULL + nsec;

    const auto& pos = odom->pose.pose.position;
    const auto& ori = odom->pose.pose.orientation;

    csv_ << timestamp << ","
         << pos.x << ","
         << pos.y << ","
         << pos.z << ","
         << ori.x << ","
         << ori.y << ","
         << ori.z << ","
         << ori.w << ","
         << travel_label
         << "\n";
    csv_.flush();

    ROS_INFO_STREAM("dt=" << dt << " Travel_label=" << travel_label);

    std::string bin_path = lidar_dir_ + "/" + std::to_string(timestamp) + ".bin";

    if (!saveCloudBinXYI(cloud, bin_path)) {
      ROS_WARN_STREAM("Failed to save cloud bin: " << bin_path);

    }
    last_time_ = now;
    pre_pose_  = cur_pose_;
  }

private:
  ros::NodeHandle nh_, pnh_;

  std::shared_ptr<message_filters::Subscriber<sensor_msgs::PointCloud2>> lidar_sub_;
  std::shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub_;
  ros::Subscriber cmd_sub_;

  std::shared_ptr<message_filters::Synchronizer<ApproxPolicy>> sync_;

  std::string output_dir_;
  std::string lidar_dir_;
  std::string csv_path_;

  Pose pre_pose_, cur_pose_;
  geometry_msgs::Twist command_vel_;
  ros::Time last_time_;
  bool initialized_{false};

  // params
  double sampling_time_{0.3};
  int queue_size_{10};
  double max_interval_sec_{0.02};
  bool use_pseudo_label_{true};
  double eta_{20.0};
  double v_th_{0.05};

  std::ofstream csv_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "data_collection_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  Data_generate node(nh, pnh);
  ros::spin();
  return 0;
}
