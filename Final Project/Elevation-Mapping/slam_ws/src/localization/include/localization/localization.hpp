#ifndef LOCALIZATION_H
#define LOCALIZATION_H

#include "rclcpp/rclcpp.hpp"
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

class Localization : public rclcpp::Node
{
    private:
        // Variables & pointers

        // Subscribers
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr camera_subscriber_;

        // Broadcasters
        std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

        // Functions
        void setupCommunications();
        void zedPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

    public:
        // Constructor and destructor
        Localization();
        ~Localization(){};
};  

#endif