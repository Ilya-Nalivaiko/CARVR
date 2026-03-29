#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <quest3carv_interfaces/msg/keyframe_data.hpp>

#include <Eigen/Dense>
#include "quest3carv_cpp/FreespaceDelaunayAlgorithm.h"

class CarvingNode : public rclcpp::Node {
public:
    CarvingNode() : Node("carving_node") {
        sub_kf_ = this->create_subscription<quest3carv_interfaces::msg::KeyframeData>(
            "quest3carv/keyframe", 10, 
            std::bind(&CarvingNode::keyframe_callback, this, std::placeholders::_1));

        pub_mesh_ = this->create_publisher<visualization_msgs::msg::Marker>("quest3carv/carved_mesh", 10);

        RCLCPP_INFO(this->get_logger(), "Freespace Carving Node Initialized! Waiting for Keyframes...");
    }

private:
    void keyframe_callback(const quest3carv_interfaces::msg::KeyframeData::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "--- NEW KEYFRAME RECEIVED (%zu points) ---", msg->points.size());

        try {
            // 1. Extract Camera Optic Center (O)
            Eigen::Vector3d cam_center(
                msg->camera_pose.position.x,
                msg->camera_pose.position.y,
                msg->camera_pose.position.z
            );

            // Calculate Principle Ray (Viewing Direction) using the Quaternion
            Eigen::Quaterniond q(
                msg->camera_pose.orientation.w,
                msg->camera_pose.orientation.x,
                msg->camera_pose.orientation.y,
                msg->camera_pose.orientation.z
            );
            Eigen::Vector3d look_dir = q * Eigen::Vector3d(0, 0, 1);

            // 2. Feed the Carver State via Proper API
            carver_.addCamCenter(cam_center); // This safely creates the corresponding visibility list!
            int current_cam_idx = carver_.numCams() - 1;

            // Handle the missing "add" methods via fetch-append-set
            auto current_cams = carver_.getCams();
            current_cams.push_back(cam_center); // Dummy copy
            carver_.setCams(current_cams);

            auto current_rays = carver_.getPrincipleRays();
            current_rays.push_back(look_dir);
            carver_.setPrincipleRays(current_rays);

            // Add Points and tie them to this camera's visibility list
            for (const auto& pt : msg->points) {
                carver_.addPoint(Eigen::Vector3d(pt.x, pt.y, pt.z));
                int current_pt_idx = carver_.numPoints() - 1;
                carver_.addVisibilityPair(current_cam_idx, current_pt_idx);
            }

            // 3. Build the CGAL Delaunay Triangulation
            dlovi::FreespaceDelaunayAlgorithm::Delaunay3 dt;
            const auto& all_points = carver_.getPoints();
            for (const auto& p : all_points) {
                dt.insert(dlovi::FreespaceDelaunayAlgorithm::PointD3(p.x(), p.y(), p.z()));
            }

            // 4. Run the Carving Algorithm!
            RCLCPP_INFO(this->get_logger(), "Carving tetrahedra...");
            carver_.TetrahedronBatchMethod(dt, false);

            // 5. Extract Isosurface (The "Skin")
            std::list<Eigen::Vector3d> tris;
            int vote_threshold = 1; // Number of rays required to carve a tetrahedron
            
            // tetsToTris requires a non-const vector, so we pass a copy to protect internal state
            std::vector<Eigen::Vector3d> points_copy = carver_.getPoints();
            carver_.tetsToTris(dt, points_copy, tris, vote_threshold);
            
            RCLCPP_INFO(this->get_logger(), "Mesh extracted! %zu triangles generated.", tris.size());

            // 6. Save and publish
            std::string obj_path = "/workspace/carved_room.obj"; 
            carver_.writeObj(obj_path, points_copy, tris);
            RCLCPP_INFO(this->get_logger(), "Saved mesh to: %s", obj_path.c_str());

            publish_mesh(tris);

        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Runtime Exception: %s", e.what());
        }
    }

    void publish_mesh(const std::list<Eigen::Vector3d>& tris) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = this->get_clock()->now();
        marker.ns = "carved_mesh";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        // Marker configuration (Light Blue, Semi-Transparent)
        marker.scale.x = 1.0; marker.scale.y = 1.0; marker.scale.z = 1.0;
        marker.color.r = 0.3; marker.color.g = 0.8; marker.color.b = 0.9;
        marker.color.a = 0.6; 

        // Map the triangle indices back to actual 3D coordinates using the secure getter
        int max_idx = carver_.numPoints();
        for (const auto& tri : tris) {
            int i0 = std::round(tri.x());
            int i1 = std::round(tri.y());
            int i2 = std::round(tri.z());

            // Safety check against bad CGAL indices
            if(i0 < 0 || i1 < 0 || i2 < 0 || i0 >= max_idx || i1 >= max_idx || i2 >= max_idx) {
                continue;
            }

            Eigen::Vector3d vec0 = carver_.getPoint(i0);
            Eigen::Vector3d vec1 = carver_.getPoint(i1);
            Eigen::Vector3d vec2 = carver_.getPoint(i2);

            geometry_msgs::msg::Point p0, p1, p2;
            p0.x = vec0.x(); p0.y = vec0.y(); p0.z = vec0.z();
            p1.x = vec1.x(); p1.y = vec1.y(); p1.z = vec1.z();
            p2.x = vec2.x(); p2.y = vec2.y(); p2.z = vec2.z();

            marker.points.push_back(p0);
            marker.points.push_back(p1);
            marker.points.push_back(p2);
        }

        pub_mesh_->publish(marker);
    }

    rclcpp::Subscription<quest3carv_interfaces::msg::KeyframeData>::SharedPtr sub_kf_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_mesh_;
    
    dlovi::FreespaceDelaunayAlgorithm carver_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CarvingNode>());
    rclcpp::shutdown();
    return 0;
}