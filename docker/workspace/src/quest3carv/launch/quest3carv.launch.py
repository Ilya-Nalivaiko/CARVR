from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Node 1: Video Receiver
        Node(
            package='quest3carv',
            executable='receiver',
            name='video_receiver',
            output='screen',
            emulate_tty=True
        ),
        # Node 2: Spatial Reconstruction (The Tracker)
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
        )
    ])