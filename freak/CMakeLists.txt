project( FreakKeypointMatcher )
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
find_package( CUDA )
cuda_add_executable( FreakKeypointMatcher FreakKeypointMatcher.cpp Freak.cpp Freak.h Freak.cu Freak.cuh TestFreak.cpp TestFreak.h
Surf.h Surf.cu Surf.cpp 
)
target_link_libraries( FreakKeypointMatcher ${OpenCV_LIBS} )
install( TARGETS FreakKeypointMatcher DESTINATION ${PROJECT_SOURCE_DIR} )


