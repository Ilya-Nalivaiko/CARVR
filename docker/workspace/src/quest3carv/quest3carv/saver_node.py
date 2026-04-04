import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from quest3carv_interfaces.msg import KeyframeData
from cv_bridge import CvBridge
import cv2
import json
import os
import numpy as np

class DatasetExportNode(Node):
    def __init__(self):
        super().__init__('dataset_export_node')

        # --- FOLDER SETUP ---
        self.export_dir = "/workspace/export_dataset"
        self.kf_dir = os.path.join(self.export_dir, "keyframes")
        
        os.makedirs(self.kf_dir, exist_ok=True)
        self.get_logger().info(f"Saving dataset to: {self.export_dir}")

        # --- STATE ---
        self.bridge = CvBridge()
        self.kf_count = 0
        self.latest_mesh_points = []

        # --- SUBSCRIBERS ---
        # Adjust these topic names to match your actual graph!
        self.kf_sub = self.create_subscription(
            KeyframeData,
            'quest3carv/keyframe', 
            self.keyframe_callback,
            10
        )
        
        self.mesh_sub = self.create_subscription(
            Marker,
            'quest3carv/carved_mesh', 
            self.mesh_callback,
            10
        )

    def keyframe_callback(self, msg):
        """Saves the image as a PNG and the math as a JSON file."""
        try:
            # 1. Save Image
            img = self.bridge.imgmsg_to_cv2(msg.image, "bgr8")
            img_path = os.path.join(self.kf_dir, f"kf_{self.kf_count:04d}.png")
            cv2.imwrite(img_path, img)

            # 2. Extract Pose (Position and Quaternion)
            pose_dict = {
                "position": {"x": msg.camera_pose.position.x, "y": msg.camera_pose.position.y, "z": msg.camera_pose.position.z},
                "orientation": {"x": msg.camera_pose.orientation.x, "y": msg.camera_pose.orientation.y, "z": msg.camera_pose.orientation.z, "w": msg.camera_pose.orientation.w}
            }

            # 3. Extract 3D Points explicitly seen in this frame
            points_list = [{"id": pid, "x": pt.x, "y": pt.y, "z": pt.z} 
                           for pid, pt in zip(msg.point_ids, msg.points)]

            # 4. Save JSON Metadata
            meta = {
                "frame_id": self.kf_count,
                "pose": pose_dict,
                "visible_points": points_list
            }
            
            json_path = os.path.join(self.kf_dir, f"kf_{self.kf_count:04d}.json")
            with open(json_path, "w") as f:
                json.dump(meta, f, indent=4)

            self.get_logger().info(f"Saved Keyframe {self.kf_count:04d}")
            self.kf_count += 1

        except Exception as e:
            self.get_logger().error(f"Failed to save keyframe: {e}")

    def mesh_callback(self, msg):
        """Silently caches the latest mesh vertices."""
        # Assuming the C++ node sends a TRIANGLE_LIST where every 3 points make a face
        if msg.type == Marker.TRIANGLE_LIST:
            self.latest_mesh_points = msg.points

    def save_obj_on_shutdown(self):
        """Writes the cached ROS Marker to a standard 3D .obj file upon exit."""
        if not self.latest_mesh_points:
            # Use standard print so it survives the ROS launch teardown!
            print("[MESH SAVER] WARNING: No mesh received! Skipping OBJ export.")
            return

        obj_path = os.path.join(self.export_dir, "final_mesh.obj")
        print(f"[MESH SAVER] Writing final mesh with {len(self.latest_mesh_points)} vertices to {obj_path}...")

        with open(obj_path, "w") as f:
            f.write("# Quest 3 SLAM Export\n")
            f.write(f"# Vertices: {len(self.latest_mesh_points)}\n")
            f.write(f"# Triangles: {len(self.latest_mesh_points) // 3}\n\n")

            # 1. Write all vertices
            for pt in self.latest_mesh_points:
                f.write(f"v {pt.x:.4f} {pt.y:.4f} {pt.z:.4f}\n")

            f.write("\n")

            # 2. Write all faces
            for i in range(1, len(self.latest_mesh_points), 3):
                f.write(f"f {i} {i+1} {i+2}\n")
                
        print("[MESH SAVER] OBJ Export Complete!")

def main(args=None):
    rclpy.init(args=args)
    node = DatasetExportNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[MESH SAVER] Shutdown signal received.")
    finally:
        # Guarantee the mesh saves when you kill the node!
        node.save_obj_on_shutdown()
        node.destroy_node()
        # Prevent the double-shutdown crash
        if rclpy.ok(): 
            rclpy.shutdown()

if __name__ == '__main__':
    main()