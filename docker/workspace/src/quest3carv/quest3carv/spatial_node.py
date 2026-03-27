import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import os

from quest3carv.tracker import StereoPointTracker

class SpatialReconstructionNode(Node):
    def __init__(self):
        super().__init__('spatial_recon')
        self.bridge = CvBridge()
        
        # Initialize Tracker (Assume K and Baseline are known for Quest 3)
        K = np.array([[320, 0, 320], [0, 320, 320], [0, 0, 1]]) # TODO get true focal length with the lab script
        self.tracker = StereoPointTracker(K, baseline=0.064)
        
        # Keyframe Logic State
        self.last_kf_pose = None
        self.keyframes = [] # List of (Image, Pose, Points)
        self.dist_threshold = 0.15 # 15cm
        self.rot_threshold = 20.0  # 20 degrees

        # Synchronized Subscribers
        self.sub_l = message_filters.Subscriber(self, Image, 'quest3/left/raw')
        self.sub_r = message_filters.Subscriber(self, Image, 'quest3/right/raw')
        self.sub_p = message_filters.Subscriber(self, PoseStamped, 'quest3/pose')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_l, self.sub_r, self.sub_p], queue_size=10, slop=0.05
        )
        self.ts.registerCallback(self.process_bundle)

    def save_debug_keyframe(self, img, points_3d, points_2d, ages):
        """
        Saves an annotated keyframe image with point distances and ages.
        """
        # Upscale factor for better text resolution
        scale = 2.0
        h, w = img.shape[:2]
        debug_img = cv2.resize(img, (int(w * scale), int(h * scale)))

        for i in range(len(points_2d)):
            # Scale 2D coordinates
            pt = (int(points_2d[i][0] * scale), int(points_2d[i][1] * scale))
            
            # Calculate Euclidean distance from the camera origin (0,0,0)
            dist = np.linalg.norm(points_3d[i])
            age = ages[i]
            
            # Draw point marker
            cv2.circle(debug_img, pt, 4, (0, 255, 0), -1)
            
            # Format label: "Dist: X.Xm | Age: Y"
            label = f"{dist:.2f}m | {age}"
            cv2.putText(debug_img, label, (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Save to disk
        kf_idx = len(self.keyframes)
        filename = f"keyframe_{kf_idx:03d}.jpg"
        cv2.imwrite(filename, debug_img)
        self.get_logger().info(f"Saved debug keyframe: {filename}")

    def is_significant_move(self, current_pose_msg):
        if self.last_kf_pose is None: return True
        
        # 1. Translation check
        curr_p = np.array([current_pose_msg.position.x, current_pose_msg.position.y, current_pose_msg.position.z])
        last_p = np.array([self.last_kf_pose.position.x, self.last_kf_pose.position.y, self.last_kf_pose.position.z])
        if np.linalg.norm(curr_p - last_p) > self.dist_threshold:
            return True
            
        # 2. Rotation check
        curr_q = [current_pose_msg.orientation.x, current_pose_msg.orientation.y, current_pose_msg.orientation.z, current_pose_msg.orientation.w]
        last_q = [self.last_kf_pose.orientation.x, self.last_kf_pose.orientation.y, self.last_kf_pose.orientation.z, self.last_kf_pose.orientation.w]
        
        r_curr = R.from_quat(curr_q)
        r_last = R.from_quat(last_q)
        relative_rot = r_curr.inv() * r_last
        if relative_rot.magnitude() > np.radians(self.rot_threshold):
            return True
            
        return False

    def process_bundle(self, msg_l, msg_r, msg_p):
        img_l = self.bridge.imgmsg_to_cv2(msg_l, "bgr8")
        img_r = self.bridge.imgmsg_to_cv2(msg_r, "bgr8")
        
        # Convert PoseStamped to 4x4 for the tracker
        curr_q = [msg_p.pose.orientation.x, msg_p.pose.orientation.y, msg_p.pose.orientation.z, msg_p.pose.orientation.w]
        curr_t = [msg_p.pose.position.x, msg_p.pose.position.y, msg_p.pose.position.z]
        mat = np.eye(4)
        mat[:3, :3] = R.from_quat(curr_q).as_matrix()
        mat[:3, 3] = curr_t
        
        # Step A: Ingest (Transient Tracking)
        self.tracker.ingest_frame(img_l, img_r, mat)
        
        # Step B: Keyframe Logic (Mapping)
        if self.is_significant_move(msg_p.pose):
            self.get_logger().info("KEYFRAME TRIGGERED: New viewpoints added.")
            self.last_kf_pose = msg_p.pose
            
            # Extract high-confidence points visible from this keyframe
            points_3d, points_2d, ages = self.tracker.get_confident_points()
            
            # Call the new debug function
            self.save_debug_keyframe(img_l, points_3d, points_2d, ages)
            
            # Proceed with adding to keyframes list
            self.keyframes.append({
                'image': img_l,
                'pose': mat,
                'points': points_3d
            })
            
            # TODO: Here is where we will call the C++ Carving node via a ROS Service/Action
            self.get_logger().info(f"Keyframe saved with {len(points_3d)} confident points.")

def main():
    rclpy.init()
    rclpy.spin(SpatialReconstructionNode())