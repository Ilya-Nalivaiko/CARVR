import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped

import cv2
import av
import socket
import struct
import threading
import numpy as np

class StereoReceiverNode(Node):
    def __init__(self):
        super().__init__('quest3_stereo_receiver')

        # 1. Setup ROS Publishers
        self.left_pub = self.create_publisher(Image, '/quest3/camera/left/image_raw', 10)
        self.right_pub = self.create_publisher(Image, '/quest3/camera/right/image_raw', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/quest3/tracking/pose', 10)

        # 2. Shared Memory for the Decoder Threads
        self.lock = threading.Lock()
        self.latest_left = None
        self.latest_right = None
        self.latest_pose = None

        self.get_logger().info("Stereo ROS Node Active. Listening on UDP 5000 & 5001...")

        # 3. Spin up two independent UDP decoding threads
        self.left_thread = threading.Thread(target=self.udp_listen_loop, args=(5000, 'left'), daemon=True)
        self.right_thread = threading.Thread(target=self.udp_listen_loop, args=(5001, 'right'), daemon=True)
        
        self.left_thread.start()
        self.right_thread.start()

        # 4. Setup the Main Thread Display & Publish Loop (60Hz)
        self.timer = self.create_timer(1.0 / 60.0, self.sync_and_publish)

    def udp_listen_loop(self, port, eye):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", port))
        codec = av.CodecContext.create('h264', 'r')

        UUID = b"CMPUT428_POSE_ID"
        POSE_STRUCT_FMT = "<q7f" 
        
        # --- NEW: Prove the port is alive ---
        self.get_logger().info(f"[{eye.upper()}] Thread listening on {port}...")

        while rclpy.ok():
            try:
                data, addr = sock.recvfrom(65535)
                
                # --- NEW: Print every 100th packet just to prove data is flowing ---
                if np.random.rand() < 0.01:
                    self.get_logger().info(f"[{eye.upper()}] Receiving data... (Packet size: {len(data)})")

                # --- INTERCEPT THE SEI NAL UNIT ---
                if len(data) == 60 and data[4] == 0x06 and data[5] == 0x05 and data[7:23] == UUID:
                    pose_bytes = data[23:59]
                    timestamp, px, py, pz, qx, qy, qz, qw = struct.unpack(POSE_STRUCT_FMT, pose_bytes)
                    
                    pose_msg = PoseStamped()
                    pose_msg.header.frame_id = "quest3_world"
                    pose_msg.pose.position.x = px
                    pose_msg.pose.position.y = py
                    pose_msg.pose.position.z = pz
                    pose_msg.pose.orientation.x = qx
                    pose_msg.pose.orientation.y = qy
                    pose_msg.pose.orientation.z = qz
                    pose_msg.pose.orientation.w = qw
                    
                    with self.lock:
                        self.latest_pose = pose_msg
                    continue 
                    
                # --- DECODE THE VIDEO ---
                packets = codec.parse(data)
                for packet in packets:
                    try:
                        frames = codec.decode(packet)
                    except av.error.InvalidDataError:
                        continue 
                        
                    for frame in frames:
                        img = frame.to_ndarray(format='bgr24')
                        h, w, _ = img.shape
                        img = cv2.resize(img, (w // 2, h // 2))
                        
                        with self.lock:
                            if eye == 'left':
                                self.latest_left = img
                            else:
                                self.latest_right = img

            except Exception as e:
                self.get_logger().error(f"Stream error on port {port}: {e}")

    def create_ros_image(self, img, frame_id, timestamp):
        img_msg = Image()
        img_msg.header.stamp = timestamp
        img_msg.header.frame_id = frame_id
        img_msg.height = img.shape[0]
        img_msg.width = img.shape[1]
        img_msg.encoding = 'bgr8'
        img_msg.is_bigendian = False
        img_msg.step = img.shape[1] * 3
        img_msg.data = img.tobytes()
        return img_msg

    def sync_and_publish(self):
        with self.lock:
            left_img = self.latest_left
            right_img = self.latest_right
            pose_msg = self.latest_pose

        now = self.get_clock().now().to_msg()

        # --- NEW: Create placeholder black images if an eye is missing ---
        if left_img is None:
            left_img = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.putText(left_img, "WAITING FOR LEFT EYE", (150, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
        if right_img is None:
            right_img = np.zeros((640, 640, 3), dtype=np.uint8)
            cv2.putText(right_img, "WAITING FOR RIGHT EYE", (150, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 1. Publish whatever we have to ROS
        if self.latest_left is not None:
            self.left_pub.publish(self.create_ros_image(left_img, "quest3_camera_left", now))
        if self.latest_right is not None:
            self.right_pub.publish(self.create_ros_image(right_img, "quest3_camera_right", now))
        
        if pose_msg is not None:
            pose_msg.header.stamp = now
            self.pose_pub.publish(pose_msg)

        # 2. Render the Window unconditionally
        sbs_image = cv2.hconcat([left_img, right_img])
        
        cv2.putText(sbs_image, "LEFT EYE (Port 5000)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(sbs_image, "RIGHT EYE (Port 5001)", (left_img.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        status_color = (0, 255, 0) if pose_msg else (0, 0, 255)
        status_text = "Tracking: LOCKED" if pose_msg else "Tracking: SEARCHING..."
        cv2.putText(sbs_image, status_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

        cv2.imshow("CMPUT428: Stereoscopic Passthrough", sbs_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = StereoReceiverNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()