rm -rf build/ install/ log/

# Build the message package first so the C++ and Python nodes can find the header/module
colcon build --packages-select quest3carv_interfaces

# Source the new message overlay
source install/setup.bash

# Build the rest (It will fail on compiling carving_node because we haven't written it yet, but it will validate the CMake structure!)
colcon build
