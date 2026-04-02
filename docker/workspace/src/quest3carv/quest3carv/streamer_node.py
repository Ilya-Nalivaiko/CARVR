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
        
        # State memory for the async callbacks
        self.latest_tri_floats = []
        self.latest_line_floats = []
        self.latest_point_floats = []
        
        self.sub_mesh = self.create_subscription(Marker, 'quest3carv/carved_mesh', self.mesh_callback, 10)
        self.sub_points = self.create_subscription(Marker, 'quest3carv/carved_points', self.points_callback, 10)
        
        self.get_logger().info(f"Waiting to connect to Quest on {self.quest_ip}:{self.tcp_port}...")

    def connect_to_quest(self):
        try:
            self.sock.connect((self.quest_ip, self.tcp_port))
            self.connected = True
            self.get_logger().info("Connected to Quest!")
        except:
            pass

    def points_callback(self, msg):
        pts = []
        for p in msg.points:
            pts.extend([p.x, p.y, p.z])
        self.latest_point_floats = pts
        
        # Trigger an immediate send so points show up even before the mesh is ready
        self.send_payload()

    def mesh_callback(self, msg):
        tri_floats = []
        line_floats = []
        
        # Only process if there are actually triangles to draw
        if msg.points:
            for i in range(0, len(msg.points), 3):
                c1 = (msg.points[i].x, msg.points[i].y, msg.points[i].z)
                c2 = (msg.points[i+1].x, msg.points[i+1].y, msg.points[i+1].z)
                c3 = (msg.points[i+2].x, msg.points[i+2].y, msg.points[i+2].z)

                tri_floats.extend([*c1, *c2, *c3])
                line_floats.extend([*c1, *c2, *c2, *c3, *c3, *c1])

        self.latest_tri_floats = tri_floats
        self.latest_line_floats = line_floats
        
        # Trigger an updated send
        self.send_payload()

    def send_payload(self):
        if not self.connected:
            self.connect_to_quest()
            return

        tris = self.latest_tri_floats
        lines = self.latest_line_floats
        pts = self.latest_point_floats

        # If both are entirely empty, don't spam the network
        if not tris and not pts:
            return 

        # 12-Byte Header
        header = struct.pack('<III', len(tris), len(lines), len(pts)) 
        
        # Binary Payload
        payload = (struct.pack(f'<{len(tris)}f', *tris) + 
                   struct.pack(f'<{len(lines)}f', *lines) +
                   struct.pack(f'<{len(pts)}f', *pts))
        
        try:
            self.sock.sendall(header + payload)
            self.get_logger().info(f"Streamed: {len(tris)//9} tris, {len(pts)//3} points.")
        except Exception as e:
            self.get_logger().error("Connection to Quest lost. Reconnecting...")
            self.connected = False
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def main():
    rclpy.init()
    node = MeshStreamerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()