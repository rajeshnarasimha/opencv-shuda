project( SurfGpu )
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
include(FindCUDA)
cuda_add_executable( SurfGpu SurfKeypointMatcher.cpp Surf.cu Surf.cpp Surf.h)
target_link_libraries( SurfGpu ${OpenCV_LIBS} )
#install( TARGETS SurfGpu DESTINATION ${PROJECT_SOURCE_DIR} )


