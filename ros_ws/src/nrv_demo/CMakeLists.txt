cmake_minimum_required(VERSION 3.0.2)
project(nrv_demo)
add_compile_options(-std=c++14 -O3 -Wall -Wextra)
find_package(catkin REQUIRED COMPONENTS
  camera_info_manager dvs_msgs event_camera_codecs event_camera_msgs
  nodelet pluginlib roscpp sensor_msgs)
find_package(Threads REQUIRED)
catkin_package(LIBRARIES nrv_demo_nodelets CATKIN_DEPENDS
  camera_info_manager dvs_msgs event_camera_codecs event_camera_msgs
  nodelet pluginlib roscpp sensor_msgs)
include_directories(${catkin_INCLUDE_DIRS})
add_library(nrv_demo_nodelets src/dvs_adapter_nodelet.cpp)
target_compile_definitions(nrv_demo_nodelets PRIVATE USING_ROS_1)
target_link_libraries(nrv_demo_nodelets ${catkin_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
add_dependencies(nrv_demo_nodelets ${catkin_EXPORTED_TARGETS})
catkin_install_python(PROGRAMS scripts/demo.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(TARGETS nrv_demo_nodelets LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
install(FILES nodelet_plugins.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
