#ifndef GLOBAL_MAPPING_H
#define GLOBAL_MAPPING_H

#include "rclcpp/rclcpp.hpp"
#include "bayes_filter.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <thread>
#include <vector>

class WorldModel : public rclcpp::Node
{
    private:
        // Variables & pointers
        const double MAP_DIMENSION = 8.0;
        const double MAP_RESOLUTION = 0.1;
        const double ELEVATION_SCALE = 100;
        const double TRAVERSABILITY_THRESHOLD = 20;
        nav_msgs::msg::OccupancyGrid global_map_, 
                                     filtered_global_map_,
                                     traversability_map_;

        // Bayes Filter
        std::vector<BayesFilter> bayes_filter_;
        
        // Transforms
        std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

        // Subscribers
        std::thread fuse_map_thread_;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr transformed_pcl_subscriber_;

        // Publishers
        rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr global_map_publisher_;
        rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr filtered_global_map_publisher_;
        rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr traversability_map_publisher_;

        // Wall Timer
        rclcpp::TimerBase::SharedPtr timer_global_map_;

        // Functions
        void setupCommunications();
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformMap(const sensor_msgs::msg::PointCloud2::SharedPtr);
        void configureMaps();
        void transformedPCLCallback(const sensor_msgs::msg::PointCloud2::SharedPtr );
        void fuseMap(const sensor_msgs::msg::PointCloud2::SharedPtr );
        void filterMap();
        void computeTraversability();
        void publishGlobalMap();

    public:
        // Constructor and destructor
        WorldModel();
        ~WorldModel(){};
};  

#endif