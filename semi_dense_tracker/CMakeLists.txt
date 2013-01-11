project( SemiDenseTracker )
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
include(FindCUDA)
FIND_PACKAGE(Boost)
IF (Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
    ADD_DEFINITIONS( "-DHAS_BOOST" )
ENDIF()
cuda_add_executable( SemiDenseTracker Main.cpp SemiDenseTracker.cpp SemiDenseTracker.h SemiDenseTracker.cu Orb.cpp
Orb.h Orb.cu Fast.cpp Fast.cu Fast.h SemiDenseTrackerOrb.cpp SemiDenseTrackerOrb.h SemiDenseTrackerOrb.cu TestCudaOrb.cpp TestCudaFast.cpp CudaHelper.hpp Helper.hpp Helper.cpp)
target_link_libraries( SemiDenseTracker ${OpenCV_LIBS} )




