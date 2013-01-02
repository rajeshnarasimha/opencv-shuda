project( SemiDenseTrackter )
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
include(FindCUDA)
cuda_add_executable( SemiDenseTrackter Main.cpp SemiDenseTracker.cpp SemiDenseTracker.h SemiDenseTracker.cu Orb.cpp
Orb.h Orb.cu Fast.cpp Fast.cu Fast.h SemiDenseTrackerOrb.cpp SemiDenseTrackerOrb.h SemiDenseTrackerOrb.cu TestCudaOrb.cpp TestCudaFast.cpp CudaHelper.hpp Helper.hpp Helper.cpp)
target_link_libraries( SemiDenseTrackter ${OpenCV_LIBS} )

add_executable( TestSemiDense TestSemiDense.cpp )
target_link_libraries( TestSemiDense ${OpenCV_LIBS} )


