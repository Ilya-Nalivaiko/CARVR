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
        self.quest_ip = '192.168.1.147' # 143 at home 147 in lab
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

        # --- THE FIX: Coordinate Frame Swap ---
        def xr_convert(p):
            # Points are now natively born in the correct OpenXR frame
            return p.x, p.y, p.z

        lines = []
        for i in range(0, len(msg.points), 3):
            # Grab the 3 points of the triangle
            p1 = msg.points[i]
            p2 = msg.points[i+1]
            p3 = msg.points[i+2]
            
            # Convert them to OpenXR Space
            c1 = xr_convert(p1)
            c2 = xr_convert(p2)
            c3 = xr_convert(p3)

            # Line 1: p1 -> p2
            lines.extend([c1[0], c1[1], c1[2], c2[0], c2[1], c2[2]])
            # Line 2: p2 -> p3
            lines.extend([c2[0], c2[1], c2[2], c3[0], c3[1], c3[2]])
            # Line 3: p3 -> p1
            lines.extend([c3[0], c3[1], c3[2], c1[0], c1[1], c1[2]])

        # Pack into binary: [Header: Num_Floats] + [Float Array]
        num_floats = len(lines)
        header = struct.pack('<I', num_floats) 
        payload = struct.pack(f'<{num_floats}f', *lines) 
        
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