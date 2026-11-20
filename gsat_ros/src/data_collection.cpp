// Data_collection_node

#include <rclcpp/rclcpp.hpp>

#include <supervision/col_supervision.h>
#include <supervision/convert_pseudo_label.h>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <iomanip>
#include <sstream>

class Data_generate : public rclcpp::Node
{
public:
  using ApproxPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::PointCloud2, nav_msgs::msg::Odometry>;

  Data_generate()
    : Node("data_collection_node")
  {
    // ROS 1 read "<dataset_name>/<key>" from the private namespace; here the YAML nests the same keys under
    // the dataset name, which ROS 2 flattens to "<dataset_name>.<key>".
    const std::string dataset_name = declare_parameter<std::string>("dataset_name", "gazebo_hill");
    const std::string prefix = dataset_name + ".";

    output_dir_       = declare_parameter<std::string>(prefix + "output_dir",
                                                       "/root/gsat_ws/src/collect_data/gazebo/hill/original");
    sampling_time_    = declare_parameter<double>(prefix + "sampling_time", 0.3);
    queue_size_       = declare_parameter<int>(prefix + "queue_size", 10);
    max_interval_sec_ = declare_parameter<double>(prefix + "max_interval_sec", 0.02);
    use_pseudo_label_ = declare_parameter<bool>(prefix + "use_pseudo_label", true);
    eta_              = declare_parameter<double>(prefix + "eta", 20.0);
    v_th_             = declare_parameter<double>(prefix + "v_th", 0.05);

    const std::string lidar_topic   = declare_parameter<std::string>(prefix + "lidar_topic", "filter/points");
    const std::string odom_topic    = declare_parameter<std::string>(prefix + "odom_topic", "filter/odom");
    const std::string cmd_vel_topic = declare_parameter<std::string>(prefix + "cmd_vel_topic", "cmd_vel");

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

    RCLCPP_INFO_STREAM(get_logger(), "Dataset: " << dataset_name);
    RCLCPP_INFO_STREAM(get_logger(), "Output dir: " << output_dir_);
    RCLCPP_INFO_STREAM(get_logger(), "CSV path  : " << csv_path_);
    RCLCPP_INFO_STREAM(get_logger(), "Sampling  : " << sampling_time_ << " sec");
    RCLCPP_INFO_STREAM(get_logger(), "Use pseudo label: " << (use_pseudo_label_ ? "true" : "false (Travel_label=1)"));
    if (use_pseudo_label_)
      RCLCPP_INFO_STREAM(get_logger(), "Label eta: " << eta_ << ", v_th: " << v_th_);
    RCLCPP_INFO_STREAM(get_logger(), "Topics    : " << lidar_topic << ", " << odom_topic << ", " << cmd_vel_topic);

    lidar_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(
        this, lidar_topic, rclcpp::SensorDataQoS().get_rmw_qos_profile());
    odom_sub_ = std::make_shared<message_filters::Subscriber<nav_msgs::msg::Odometry>>(this, odom_topic);
    cmd_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        cmd_vel_topic, 10, std::bind(&Data_generate::cmdCallback, this, std::placeholders::_1));

    sync_ = std::make_shared<message_filters::Synchronizer<ApproxPolicy>>(
        ApproxPolicy(queue_size_), *lidar_sub_, *odom_sub_);

    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(max_interval_sec_));
    sync_->registerCallback(std::bind(&Data_generate::generate_callback, this,
                                      std::placeholders::_1, std::placeholders::_2));
  }

  ~Data_generate()
  {
    if (csv_.is_open())
      csv_.close();
  }

private:
  Pose convert_odom(const nav_msgs::msg::Odometry::ConstSharedPtr& odom)
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

  bool saveCloudBinXYI(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud,
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
      RCLCPP_ERROR_STREAM(get_logger(), "Failed to open bin: " << bin_path);
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


  void cmdCallback(const geometry_msgs::msg::Twist::ConstSharedPtr& msg)
  {
    command_vel_ = *msg;  // latest command
  }

  void generate_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud,
                         const nav_msgs::msg::Odometry::ConstSharedPtr& odom)
  {
    if (!csv_.is_open())
      return;

    rclcpp::Time now = this->now();
    cur_pose_ = convert_odom(odom);

    if (!initialized_)
    {
      last_time_   = now;
      initialized_ = true;
      pre_pose_    = cur_pose_;
      return;  // first frame: no dt
    }

    double dt = (now - last_time_).seconds();
    if (dt < sampling_time_)
      return;


    if (std::abs(command_vel_.linear.x) < 1e-6)
      return;

    // Pseudo label from cmd_vel vs odom, or fixed 1.0
    const double travel_label = use_pseudo_label_
        ? convert_label_trac_image(pre_pose_, cur_pose_, command_vel_, dt, eta_, v_th_)
        : 1.0;

    const uint32_t sec  = odom->header.stamp.sec;
    const uint32_t nsec = odom->header.stamp.nanosec;
    
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

    RCLCPP_INFO_STREAM(get_logger(), "dt=" << dt << " Travel_label=" << travel_label);

    std::string bin_path = lidar_dir_ + "/" + std::to_string(timestamp) + ".bin";

    if (!saveCloudBinXYI(cloud, bin_path)) {
      RCLCPP_WARN_STREAM(get_logger(), "Failed to save cloud bin: " << bin_path);

    }
    last_time_ = now;
    pre_pose_  = cur_pose_;
  }

private:
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> lidar_sub_;
  std::shared_ptr<message_filters::Subscriber<nav_msgs::msg::Odometry>> odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;

  std::shared_ptr<message_filters::Synchronizer<ApproxPolicy>> sync_;

  std::string output_dir_;
  std::string lidar_dir_;
  std::string csv_path_;

  Pose pre_pose_, cur_pose_;
  geometry_msgs::msg::Twist command_vel_;
  rclcpp::Time last_time_;
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
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Data_generate>());
  rclcpp::shutdown();
  return 0;
}
