#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <quest3carv_interfaces/msg/keyframe_data.hpp>

#include <Eigen/Dense>
#include "quest3carv_cpp/FreespaceDelaunayAlgorithm.h"
#include <unordered_map>
#include <vector>

class CarvingNode : public rclcpp::Node {
public:
    CarvingNode() : Node("carving_node"), keyframe_count_(0), process_every_n_frames_(3) {
        sub_kf_ = this->create_subscription<quest3carv_interfaces::msg::KeyframeData>(
            "quest3carv/keyframe", 10, 
            std::bind(&CarvingNode::keyframe_callback, this, std::placeholders::_1));

        pub_mesh_ = this->create_publisher<visualization_msgs::msg::Marker>("quest3carv/carved_mesh", 10);
        pub_points_ = this->create_publisher<visualization_msgs::msg::Marker>("quest3carv/carved_points", 10);

        RCLCPP_INFO(this->get_logger(), "Incremental Freespace Carving Node Initialized.");
    }

private:
    void keyframe_callback(const quest3carv_interfaces::msg::KeyframeData::SharedPtr msg) {
        keyframe_count_++;

        try {
            // 1. Extract Camera Optic Center
            Eigen::Vector3d cam_center(
                msg->camera_pose.position.x,
                msg->camera_pose.position.y,
                msg->camera_pose.position.z
            );

            Eigen::Quaterniond q(
                msg->camera_pose.orientation.w,
                msg->camera_pose.orientation.x,
                msg->camera_pose.orientation.y,
                msg->camera_pose.orientation.z
            );
            Eigen::Vector3d look_dir = q * Eigen::Vector3d(0, 0, -1);

            // 2. Feed the Carver State
            carver_.addCamCenter(cam_center); 
            int current_cam_idx = carver_.numCams() - 1;

            auto current_cams = carver_.getCams();
            current_cams.push_back(cam_center); 
            carver_.setCams(current_cams);

            auto current_rays = carver_.getPrincipleRays();
            current_rays.push_back(look_dir);
            carver_.setPrincipleRays(current_rays);

            // Process Incoming Points
            for (size_t i = 0; i < msg->points.size(); ++i) {
                uint32_t global_id = msg->point_ids[i];
                int local_idx;

                if (global_id_to_local_idx_.count(global_id) > 0) {
                    local_idx = global_id_to_local_idx_[global_id];
                    obs_count_[local_idx]++;
                    last_seen_kf_[local_idx] = keyframe_count_;
                } else {
                    carver_.addPoint(Eigen::Vector3d(msg->points[i].x, msg->points[i].y, msg->points[i].z));
                    local_idx = carver_.numPoints() - 1;
                    global_id_to_local_idx_[global_id] = local_idx; 
                    
                    obs_count_.push_back(1);
                    last_seen_kf_.push_back(keyframe_count_);
                    is_dead_.push_back(false);
                }
                carver_.addVisibilityPair(current_cam_idx, local_idx);
            }

            // --- 3. THE INCREMENTAL DELAUNAY ENGINE (Algorithm 1 & 2) ---
            // This updates the persistent dt_ mesh instead of rebuilding it from scratch
            carver_.IterateTetrahedronMethod(dt_, vecVertexHandles_, current_cam_idx);

            // --- 4. OUTLIER DELETION (Algorithm 3) ---
            int ghosts_culled = 0;
            for (size_t i = 0; i < obs_count_.size(); ++i) {
                if (!is_dead_[i] && obs_count_[i] <= 3 && (keyframe_count_ - last_seen_kf_[i]) > 5) {
                    // This physically rips the point out of the Delaunay mesh,
                    // allowing the carving rays to flood the hole and erase the spiderweb.
                    carver_.removeVertex(dt_, vecVertexHandles_, i);
                    is_dead_[i] = true; 
                    ghosts_culled++;
                }
            }
            if (ghosts_culled > 0) {
                RCLCPP_INFO(this->get_logger(), "Algorithm 3: Erased %d outlier points and recarved holes.", ghosts_culled);
            }

            // --- 5. THE THROTTLE GATE ---
            // (Placed after the state updates so the Delaunay engine never desyncs from the tracker)
            if (keyframe_count_ % process_every_n_frames_ != 0) {
                return; 
            }

            // 6. Extract Isosurface (The "Skin")
            std::list<Eigen::Vector3d> tris;
            std::vector<Eigen::Vector3d> points_copy = carver_.getPoints();
            carver_.tetsToTris(dt_, points_copy, tris, 1);
            
            // Standard Near-Field Clipper
            double near_clip_dist = 0.45; 
            std::list<Eigen::Vector3d> filtered_tris;
            const auto& cams = carver_.getCamCenters(); 
            
            for (const auto& tri : tris) {
                int i0 = std::round(tri.x());
                int i1 = std::round(tri.y());
                int i2 = std::round(tri.z());
                
                Eigen::Vector3d centroid = (points_copy[i0] + points_copy[i1] + points_copy[i2]) / 3.0;
                
                double min_cam_dist = std::numeric_limits<double>::max();
                for (const auto& cam : cams) {
                    double dist = (centroid - cam).norm();
                    if (dist < min_cam_dist) min_cam_dist = dist;
                }
                
                if (min_cam_dist > near_clip_dist) {
                    filtered_tris.push_back(tri);
                }
            }
            tris = filtered_tris; 

            // 7. Publish
            publish_mesh(tris);
            publish_points(msg);

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
        
        marker.scale.x = 1.0; marker.scale.y = 1.0; marker.scale.z = 1.0;
        marker.color.r = 0.3; marker.color.g = 0.8; marker.color.b = 0.9;
        marker.color.a = 0.6; 

        int max_idx = carver_.numPoints();
        for (const auto& tri : tris) {
            int i0 = std::round(tri.x());
            int i1 = std::round(tri.y());
            int i2 = std::round(tri.z());

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

    void publish_points(const quest3carv_interfaces::msg::KeyframeData::SharedPtr& msg) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = this->get_clock()->now();
        marker.ns = "carved_points";
        marker.id = 1; 
        
        marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        marker.scale.x = 0.02; marker.scale.y = 0.02; marker.scale.z = 0.02;
        marker.color.r = 1.0; marker.color.g = 1.0; marker.color.b = 0.0; marker.color.a = 1.0; 

        for (const auto& pt : msg->points) {
            marker.points.push_back(pt);
        }
        pub_points_->publish(marker);
    }

    rclcpp::Subscription<quest3carv_interfaces::msg::KeyframeData>::SharedPtr sub_kf_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_mesh_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_points_;
    
    // --- THE FIX: Persistent Incremental State ---
    dlovi::FreespaceDelaunayAlgorithm carver_;
    dlovi::FreespaceDelaunayAlgorithm::Delaunay3 dt_;
    std::vector<dlovi::FreespaceDelaunayAlgorithm::Delaunay3::Vertex_handle> vecVertexHandles_;

    std::unordered_map<uint32_t, int> global_id_to_local_idx_; 
    
    std::vector<int> obs_count_;
    std::vector<int> last_seen_kf_;
    std::vector<bool> is_dead_;
    
    int keyframe_count_;
    int process_every_n_frames_; 
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CarvingNode>());
    rclcpp::shutdown();
    return 0;
}