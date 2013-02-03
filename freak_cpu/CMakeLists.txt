project( FreakCPU )
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
add_executable( FreakCPU FreakCPU.cpp )
target_link_libraries( FreakCPU ${OpenCV_LIBS} )
#install( TARGETS FreakKeypointMatcher DESTINATION ${PROJECT_SOURCE_DIR} )


