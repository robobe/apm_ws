cmake_minimum_required(VERSION 3.8)
project(mavros_ex)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies

find_package(ament_cmake REQUIRED)

# find mavros dependencies
find_package(rclcpp REQUIRED)
find_package(pluginlib REQUIRED)
find_package(message_filters REQUIRED)

find_package(mavlink REQUIRED)
find_package(mavros REQUIRED)
find_package(tf2_ros REQUIRED)

find_package(diagnostic_msgs REQUIRED)
find_package(diagnostic_updater REQUIRED)

find_package(mavros_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)


add_library(${PROJECT_NAME}
  src/plugins/dummy.cpp
  )

ament_target_dependencies(
  ${PROJECT_NAME}
  mavros
  pluginlib
  mavros_msgs
  rclcpp
  rclcpp_components
  rcpputils
  diagnostic_updater
  tf2_ros
  sensor_msgs
)

pluginlib_export_plugin_description_file(mavros plugins.xml)


install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

add_dependencies(${PROJECT_NAME}
${PROJECT_NAME}
)
ament_export_targets(
  export_${PROJECT_NAME}
)
ament_package()
