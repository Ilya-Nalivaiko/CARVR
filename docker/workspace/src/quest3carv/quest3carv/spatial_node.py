import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation as R

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
            points = self.tracker.get_confident_points()
            self.keyframes.append({
                'image': img_l,
                'pose': mat,
                'points': points
            })
            
            # TODO: Here is where we will call the C++ Carving node via a ROS Service/Action
            self.get_logger().info(f"Keyframe saved with {len(points)} confident points.")

def main():
    rclpy.init()
    rclpy.spin(SpatialReconstructionNode())