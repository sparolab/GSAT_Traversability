// transform_utils.cpp
#include "transform/transform_utils.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/PCLPointCloud2.h>

#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <geometry_msgs/TransformStamped.h>

namespace transform
{
    Eigen::Isometry3d odomToIsometry(const nav_msgs::Odometry& odom)
    {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

        T.translation() <<
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z;

        Eigen::Quaterniond q(
            odom.pose.pose.orientation.w,
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z);

        T.linear() = q.toRotationMatrix();
        return T;
    }


    sensor_msgs::PointCloud2 transformCloudTF2(const sensor_msgs::PointCloud2& cloud,
                                            const Eigen::Isometry3d& T,
                                            const std::string& target_frame)
    {
        geometry_msgs::TransformStamped tf;
        tf.header.stamp = cloud.header.stamp;
        tf.header.frame_id = target_frame;
        tf.child_frame_id = cloud.header.frame_id;

        tf.transform.translation.x = T.translation().x();
        tf.transform.translation.y = T.translation().y();
        tf.transform.translation.z = T.translation().z();

        Eigen::Quaterniond q(T.rotation());
        q.normalize();
        tf.transform.rotation.x = q.x();
        tf.transform.rotation.y = q.y();
        tf.transform.rotation.z = q.z();
        tf.transform.rotation.w = q.w();

        sensor_msgs::PointCloud2 out;
        tf2::doTransform(cloud, out, tf);
        out.header.frame_id = target_frame;
        return out;
    }

}
