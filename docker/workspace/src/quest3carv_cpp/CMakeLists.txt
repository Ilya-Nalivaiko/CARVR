cmake_minimum_required(VERSION 3.8)
project(quest3carv_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Find ROS 2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(quest3carv_interfaces REQUIRED)

# Find Math/Geometry Libraries
find_package(Eigen3 REQUIRED)
find_package(CGAL REQUIRED)
find_package(Boost REQUIRED COMPONENTS graph)

# Create the executable (We will create carving_node.cpp next)
add_executable(carving_node 
  src/carving_node.cpp 
  src/FreespaceDelaunayAlgorithm.cc
  src/GraphWrapper_Boost.cc
)

# Include directories
target_include_directories(carving_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

# Link ROS 2 and CGAL dependencies
ament_target_dependencies(carving_node
  rclcpp
  geometry_msgs
  visualization_msgs
  quest3carv_interfaces
)
target_link_libraries(carving_node CGAL::CGAL Boost::graph)

# Install rules
install(TARGETS carving_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
