# Start with ROS 2 Humble (Ubuntu 22.04)
FROM ros:humble-ros-base

# Install system dependencies for OpenCV and FFmpeg
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install the Python libraries for decoding and displaying the stream
RUN pip3 install numpy opencv-python av

# Setup the workspace directory
WORKDIR /workspace

# Source ROS 2 automatically when opening a bash shell in the container
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc