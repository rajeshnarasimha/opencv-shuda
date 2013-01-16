project( FarnebackOpticalFlowGpu )
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
add_executable( FarnebackOpticalFlowGpu FarnebackOpticalFlowGpu.cpp )
target_link_libraries( FarnebackOpticalFlowGpu ${OpenCV_LIBS} )
#install( TARGETS FarnebackOpticalFlowGpu DESTINATION ${PROJECT_SOURCE_DIR} )


