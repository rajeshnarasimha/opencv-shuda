project( OrbKeypointMatcher )
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
add_executable( OrbKeypointMatcher OrbKeypointMatcher.cpp )
target_link_libraries( OrbKeypointMatcher ${OpenCV_LIBS} )
install( TARGETS OrbKeypointMatcher DESTINATION ${PROJECT_SOURCE_DIR} )


