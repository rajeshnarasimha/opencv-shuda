project( FastGpuKeypointExtractionAndMatcher)
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
add_executable( FastGpuKeypointExtractionAndMatcher FastGpuKeypointExtractionAndMatcher.cpp )
target_link_libraries( FastGpuKeypointExtractionAndMatcher ${OpenCV_LIBS} )
install( TARGETS FastGpuKeypointExtractionAndMatcher DESTINATION ${PROJECT_SOURCE_DIR} )


