project( CVGPU )
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
add_executable( CVGPU SurfKeypointMatcher.cpp )
target_link_libraries( CVGPU ${OpenCV_LIBS} )
install( TARGETS CVGPU DESTINATION ${PROJECT_SOURCE_DIR} )


