#include "localization/localization.hpp"

Localization::Localization() : Node("localization_node")
{   

    // Setup Communications
    setupCommunications();

    RCLCPP_INFO(this->get_logger(), "Localization initialized");
}

void Localization::setupCommunications(){
    // QoS profile
    rclcpp::QoS qos_profile = rclcpp::QoS(rclcpp::SensorDataQoS()).reliability(rclcpp::ReliabilityPolicy::BestEffort);

    // Subscribers
    camera_subscriber_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/zed/zed_node/pose",
        qos_profile,
        std::bind(&Localization::zedPoseCallback, this, std::placeholders::_1));

    // Broadcasters
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
}

void Localization::zedPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    geometry_msgs::msg::TransformStamped tf_msg;

    tf_msg.header.stamp = msg->header.stamp;
    tf_msg.header.frame_id = "map";
    tf_msg.child_frame_id = "base_link";

    tf_msg.transform.translation.x = msg->pose.position.x;
    tf_msg.transform.translation.y = msg->pose.position.y;
    tf_msg.transform.translation.z = msg->pose.position.z;

    tf_msg.transform.rotation = msg->pose.orientation;

    tf_broadcaster_->sendTransform(tf_msg);
}