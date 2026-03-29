#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <quest3carv_interfaces/msg/keyframe_data.hpp>

// We will include your algorithm header here once we clean it up!
#include "quest3carv_cpp/FreespaceDelaunayAlgorithm.h"

class CarvingNode : public rclcpp::Node {
public:
    CarvingNode() : Node("carving_node") {
        // Subscribe to the custom KeyframeData message from spatial_node.py
        sub_kf_ = this->create_subscription<quest3carv_interfaces::msg::KeyframeData>(
            "quest3carv/keyframe", 10, 
            std::bind(&CarvingNode::keyframe_callback, this, std::placeholders::_1));

        // Publisher for the final RViz Mesh
        pub_mesh_ = this->create_publisher<visualization_msgs::msg::Marker>("quest3carv/carved_mesh", 10);

        RCLCPP_INFO(this->get_logger(), "Freespace Carving Node Initialized.");
    }

private:
    void keyframe_callback(const quest3carv_interfaces::msg::KeyframeData::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "Received Keyframe with %zu points. Ready to carve!", msg->points.size());
        
        // TODO: Convert ROS message to Eigen::Vector3d
        // TODO: Pass into carver_.IterateTetrahedronMethod()
        // TODO: Extract mesh with carver_.tetsToTris_naive()
        // TODO: Publish visualization_msgs::msg::Marker
    }

    rclcpp::Subscription<quest3carv_interfaces::msg::KeyframeData>::SharedPtr sub_kf_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_mesh_;
    
    // The core algorithm class from the Lovi paper
    dlovi::FreespaceDelaunayAlgorithm carver_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CarvingNode>());
    rclcpp::shutdown();
    return 0;
}
