project( ShudaOrb )
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
include(FindCUDA)
#set( CUDA_NVCC_FLAGS "-arch=sm_21" ) #allow atomicAdd operators
cuda_add_executable( ShudaOrb Main.cpp Orb.cpp Orb.h Orb.cu Fast.cpp Fast.h Fast.cu )

target_link_libraries( ShudaOrb ${OpenCV_LIBS}  )
install( TARGETS ShudaOrb DESTINATION ${PROJECT_SOURCE_DIR} )


