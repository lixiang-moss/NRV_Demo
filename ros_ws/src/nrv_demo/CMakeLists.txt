cmake_minimum_required(VERSION 3.0.2)
project(nrv_demo)
add_compile_options(-std=c++14 -O3 -Wall -Wextra)
find_package(nlohmann_json REQUIRED)
find_package(Threads REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  camera_info_manager dvs_msgs event_camera_codecs event_camera_msgs
  nodelet pluginlib roscpp sensor_msgs std_msgs message_generation)
add_message_files(FILES E2faiResult.msg)
generate_messages(DEPENDENCIES sensor_msgs std_msgs)
catkin_package(LIBRARIES nrv_demo_nodelets CATKIN_DEPENDS
  camera_info_manager dvs_msgs event_camera_codecs event_camera_msgs
  nodelet pluginlib roscpp sensor_msgs std_msgs message_runtime)
include_directories(include ${catkin_INCLUDE_DIRS})
add_library(nrv_demo_nodelets src/dvs_adapter_nodelet.cpp)
target_compile_definitions(nrv_demo_nodelets PRIVATE USING_ROS_1)
target_link_libraries(nrv_demo_nodelets ${catkin_LIBRARIES})
add_dependencies(nrv_demo_nodelets ${catkin_EXPORTED_TARGETS})
add_executable(e2fai_bridge src/e2fai_bridge.cpp)
target_link_libraries(e2fai_bridge ${catkin_LIBRARIES} nlohmann_json::nlohmann_json Threads::Threads)
add_dependencies(e2fai_bridge ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
catkin_install_python(PROGRAMS scripts/capture_monitor.py scripts/gui.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(TARGETS nrv_demo_nodelets e2fai_bridge
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(FILES nodelet_plugins.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

foreach(module e2fai_protocol performance)
  configure_file(scripts/${module}.py ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}/${module}.py COPYONLY)
  install(FILES scripts/${module}.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
endforeach()
