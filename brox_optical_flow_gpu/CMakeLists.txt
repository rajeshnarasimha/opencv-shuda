project( BroxOpticalFlowGpu )
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
add_executable( BroxOpticalFlowGpu BroxOpticalFlowGpu.cpp )
target_link_libraries( BroxOpticalFlowGpu ${OpenCV_LIBS} )
install( TARGETS BroxOpticalFlowGpu DESTINATION ${PROJECT_SOURCE_DIR} )


