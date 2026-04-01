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

        def xr_convert(p):
            return p.x, p.y, p.z

        tri_floats = []
        line_floats = []
        for i in range(0, len(msg.points), 3):
            c1 = xr_convert(msg.points[i])
            c2 = xr_convert(msg.points[i+1])
            c3 = xr_convert(msg.points[i+2])

            # 1. Solid Triangles (The Invisible Depth Shield)
            tri_floats.extend([*c1, *c2, *c3])
            # 2. Lines (The Visible Hologram)
            line_floats.extend([*c1, *c2, *c2, *c3, *c3, *c1])

        # Header: [UInt32 Num_Tri_Floats] [UInt32 Num_Line_Floats]
        header = struct.pack('<II', len(tri_floats), len(line_floats)) 
        payload = struct.pack(f'<{len(tri_floats)}f', *tri_floats) + struct.pack(f'<{len(line_floats)}f', *line_floats)
        
        try:
            self.sock.sendall(header + payload)
            self.get_logger().info(f"Sent wireframe overlay: {len(line_floats)//6} lines, {len(tri_floats)//3} triangles.")
        except Exception as e:
            self.get_logger().error("Connection lost. Reconnecting...")
            self.connected = False
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def main():
    rclpy.init()
    rclpy.spin(MeshStreamerNode())

if __name__ == '__main__':
    main()