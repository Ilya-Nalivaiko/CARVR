import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import cv2
import av
import socket
import struct
import numpy as np
import multiprocessing as mp
from cv_bridge import CvBridge

# Constants matching your secondary script
WIDTH, HEIGHT = 640, 640
UUID = b"CMPUT428_POSE_ID"
POSE_STRUCT_FMT = "<q7f"

def eye_worker(port, eye_side, frame_queue, pose_queue):
    """
    Isolated process for network ingestion and H.264 decoding.
    Using a separate process avoids the Python GIL and improves ROS2 performance.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Increase buffer to prevent packet loss at high bitrates
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2*1024*1024)
    sock.bind(("0.0.0.0", port))
    
    codec = av.CodecContext.create('h264', 'r')
    codec.thread_type = 'FRAME'
    codec.thread_count = 4

    while True:
        try:
            data, _ = sock.recvfrom(65535)
            
            # 1. Handle Pose SEI (60 bytes total)
            if len(data) == 60 and data[4] == 0x06 and data[7:23] == UUID:
                pose_data = struct.unpack(POSE_STRUCT_FMT, data[23:59])
                # Only keep latest pose
                if pose_queue.full():
                    try: pose_queue.get_nowait()
                    except: pass
                pose_queue.put(pose_data)
                continue

            # 2. Decode Video
            packets = codec.parse(data)
            for packet in packets:
                try:
                    frames = codec.decode(packet)
                    for frame in frames:
                        img = frame.to_ndarray(format='bgr24')
                        
                        # Maintain latest frame only to prevent lag
                        if frame_queue.full():
                            try: frame_queue.get_nowait()
                            except: pass
                        frame_queue.put(img)
                except av.error.InvalidDataError:
                    continue
        except Exception as e:
            print(f"Error in {eye_side} worker: {e}")

class Quest3ReceiverNode(Node):
    def __init__(self):
        super().__init__('quest3_receiver')
        self.bridge = CvBridge()
        
        # ROS Publishers
        self.left_pub = self.create_publisher(Image, 'quest3/left/raw', 10)
        self.right_pub = self.create_publisher(Image, 'quest3/right/raw', 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'quest3/pose', 10)

        # Multiprocessing Communication
        self.left_q = mp.Queue(maxsize=1)
        self.right_q = mp.Queue(maxsize=1)
        self.pose_q = mp.Queue(maxsize=2) # Store poses from both eyes

        # Start Workers
        self.p_left = mp.Process(target=eye_worker, args=(5000, 'left', self.left_q, self.pose_q), daemon=True)
        self.p_right = mp.Process(target=eye_worker, args=(5001, 'right', self.right_q, self.pose_q), daemon=True)
        
        self.p_left.start()
        self.p_right.start()

        # Timer to poll queues and publish to ROS (30Hz)
        self.create_timer(1/30.0, self.poll_and_publish)
        self.get_logger().info("Quest 3 Stereo Receiver Node Started (Parallel Mode)")

    def poll_and_publish(self):
        # Check if we have new frames
        l_img = self.left_q.get() if not self.left_q.empty() else None
        r_img = self.right_q.get() if not self.right_q.empty() else None
        
        # Common timestamp for sync
        now = self.get_clock().now().to_msg()

        # Publish Images if available
        if l_img is not None:
            msg = self.bridge.cv2_to_imgmsg(l_img, "bgr8")
            msg.header.stamp = now
            msg.header.frame_id = "quest3_left"
            self.left_pub.publish(msg)

        if r_img is not None:
            msg = self.bridge.cv2_to_imgmsg(r_img, "bgr8")
            msg.header.stamp = now
            msg.header.frame_id = "quest3_right"
            self.right_pub.publish(msg)

        # Publish Latest Pose if available
        while not self.pose_q.empty():
            lp = self.pose_q.get()
            # lp format: (ts, px, py, pz, qx, qy, qz, qw)
            p_msg = PoseStamped()
            p_msg.header.stamp = now
            p_msg.header.frame_id = "world"
            p_msg.pose.position.x, p_msg.pose.position.y, p_msg.pose.position.z = lp[1], lp[2], lp[3]
            p_msg.pose.orientation.x, p_msg.pose.orientation.y, p_msg.pose.orientation.z, p_msg.pose.orientation.w = lp[4], lp[5], lp[6], lp[7]
            self.pose_pub.publish(p_msg)

def main():
    rclpy.init()
    node = Quest3ReceiverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.p_left.terminate()
        node.p_right.terminate()
        rclpy.shutdown()

if __name__ == '__main__':
    main()