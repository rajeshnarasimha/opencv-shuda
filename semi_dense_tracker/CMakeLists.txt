project( SemiDenseTrackter )
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
include(FindCUDA)
cuda_add_executable( SemiDenseTrackter Main.cpp SemiDenseTracker.cu )
target_link_libraries( SemiDenseTrackter ${OpenCV_LIBS} )

