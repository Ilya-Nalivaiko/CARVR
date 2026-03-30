import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
import socket
import struct
import numpy as np

class MeshStreamerNode(Node):
    def __init__(self):
        super().__init__('mesh_streamer')
        
        # --- CONFIG ---
        self.quest_ip = '192.168.1.XX' # <--- CHANGE THIS TO YOUR QUEST'S IP
        self.tcp_port = 5002
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        
        self.sub = self.create_subscription(
            Marker,
            'quest3carv/carved_mesh',
            self.mesh_callback,
            10
        )
        self.get_logger().info(f"Waiting to connect to Quest on {self.quest_ip}:{self.tcp_port}...")

    def connect_to_quest(self):
        try:
            self.sock.connect((self.quest_ip, self.tcp_port))
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) # Disable Nagle's algorithm for low latency
            self.connected = True
            self.get_logger().info("CONNECTED TO QUEST 3 TCP SERVER!")
        except Exception as e:
            pass # Keep trying silently

    def mesh_callback(self, msg):
        if not self.connected:
            self.connect_to_quest()
            return
            
        if not msg.points:
            return

        # The Marker is a TRIANGLE_LIST (A,B,C, A,B,C...)
        # We need to convert it to a LINE_LIST (A,B, B,C, C,A...) so OpenGL ES can draw it as a wireframe
        lines = []
        for i in range(0, len(msg.points), 3):
            p1, p2, p3 = msg.points[i], msg.points[i+1], msg.points[i+2]
            
            # Line 1: p1 -> p2
            lines.extend([p1.x, p1.y, p1.z, p2.x, p2.y, p2.z])
            # Line 2: p2 -> p3
            lines.extend([p2.x, p2.y, p2.z, p3.x, p3.y, p3.z])
            # Line 3: p3 -> p1
            lines.extend([p3.x, p3.y, p3.z, p1.x, p1.y, p1.z])

        # Pack into binary: [Header: Num_Floats] + [Float Array]
        num_floats = len(lines)
        header = struct.pack('<I', num_floats) # Unsigned 32-bit int
        payload = struct.pack(f'<{num_floats}f', *lines) # Little-endian floats
        
        try:
            self.sock.sendall(header + payload)
            self.get_logger().info(f"Sent wireframe overlay: {num_floats//6} lines.")
        except Exception as e:
            self.get_logger().error("Connection lost. Reconnecting...")
            self.connected = False
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def main():
    rclpy.init()
    rclpy.spin(MeshStreamerNode())

if __name__ == '__main__':
    main()