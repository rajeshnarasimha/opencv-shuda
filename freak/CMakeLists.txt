project( FreakKeypointMatcher )
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
add_executable( FreakKeypointMatcher FreakKeypointMatcher.cpp )
target_link_libraries( FreakKeypointMatcher ${OpenCV_LIBS} )
install( TARGETS FreakKeypointMatcher DESTINATION ${PROJECT_SOURCE_DIR} )


