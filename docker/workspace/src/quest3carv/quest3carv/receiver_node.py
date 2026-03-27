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
from cv_bridge import CvBridge

class Quest3ReceiverNode(Node):
    def __init__(self):
        super().__init__('quest3_receiver')
        self.bridge = CvBridge()
        
        # Publishers
        self.left_pub = self.create_publisher(Image, 'quest3/left/raw', 10)
        self.right_pub = self.create_publisher(Image, 'quest3/right/raw', 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'quest3/pose', 10)

        self.lock = threading.Lock()
        self.latest_left = None
        self.latest_right = None
        self.latest_pose = None

        # UDP Threads
        threading.Thread(target=self.udp_listener, args=(5000, 'left'), daemon=True).start()
        threading.Thread(target=self.udp_listener, args=(5001, 'right'), daemon=True).start()

        # Sync and Publish at 30Hz
        self.create_timer(1/30.0, self.publish_sync_data)

    def udp_listener(self, port, eye):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", port))
        codec = av.CodecContext.create('h264', 'r')
        codec.thread_count = 4 

        UUID = b"CMPUT428_POSE_ID"
        while rclpy.ok():
            data, _ = sock.recvfrom(65535)
            # Pose Extraction
            if len(data) == 60 and data[4] == 0x06 and data[7:23] == UUID:
                _, px, py, pz, qx, qy, qz, qw = struct.unpack("<q7f", data[23:59])
                p = PoseStamped()
                p.pose.position.x, p.pose.position.y, p.pose.position.z = px, py, pz
                p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w = qx, qy, qz, qw
                with self.lock: self.latest_pose = p
                continue

            # Video Decoding
            packets = codec.parse(data)
            for packet in packets:
                frames = codec.decode(packet)
                for frame in frames:
                    img = frame.to_ndarray(format='bgr24')
                    with self.lock:
                        if eye == 'left': self.latest_left = img
                        else: self.latest_right = img

    def publish_sync_data(self):
        with self.lock:
            if self.latest_left is None or self.latest_right is None or self.latest_pose is None:
                return
            l_img, r_img, pose = self.latest_left, self.latest_right, self.latest_pose

        stamp = self.get_clock().now().to_msg()
        
        # Image Left
        msg_l = self.bridge.cv2_to_imgmsg(l_img, "bgr8")
        msg_l.header.stamp = stamp
        
        # Image Right
        msg_r = self.bridge.cv2_to_imgmsg(r_img, "bgr8")
        msg_r.header.stamp = stamp
        
        # Pose
        pose.header.stamp = stamp
        pose.header.frame_id = "world"

        self.left_pub.publish(msg_l)
        self.right_pub.publish(msg_r)
        self.pose_pub.publish(pose)

def main():
    rclpy.init()
    rclpy.spin(Quest3ReceiverNode())