rm -rf debug/
rm -rf export_dataset/

source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch quest3carv quest3carv.launch.py
