project( BriskCPU )
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
add_executable( BriskCPU BriskCPU.cpp )
target_link_libraries( BriskCPU ${OpenCV_LIBS} )
#install( TARGETS FreakKeypointMatcher DESTINATION ${PROJECT_SOURCE_DIR} )


