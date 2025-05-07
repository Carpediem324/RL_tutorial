cmake_minimum_required(VERSION 3.8)
project(jethexa_description)

find_package(ament_cmake REQUIRED)
find_package(xacro REQUIRED)
find_package(urdf REQUIRED)


install(DIRECTORY urdf meshes gazebo
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
