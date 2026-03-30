from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Node 1: Video Receiver & Pose Ingestion (Python)
        Node(
            package='quest3carv',
            executable='receiver',
            name='video_receiver',
            output='screen',
            emulate_tty=True
        ),
        
        # Node 2: Spatial Reconstruction / The Tracker (Python)
        Node(
            package='quest3carv',
            executable='spatial_recon',
            name='spatial_recon',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'max_points': 500,
                'dist_kf_threshold': 0.15,
                'rot_kf_threshold': 20.0
            }]
        ),

        # Node 3: Incremental Free-Space Carver (C++)
        Node(
            package='quest3carv_cpp',
            executable='carving_node',
            name='carving_node',
            output='screen',
            emulate_tty=True
        ),

        # Node 4: Send Mesh to Quest
        Node(
            package='quest3carv',
            executable='sender',
            name='mesh_sender',
            output='screen',
            emulate_tty=True
        ),
    ])