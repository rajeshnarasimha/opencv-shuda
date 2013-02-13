project( ShudaOrb )
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
find_package( CUDA )
include(FindCUDA)
if( WIN32 )
    include_directories ( "C:/csxsl/src/opencv-shuda/btl_descriptor/" )
    link_directories ( "C:/csxsl/src/opencv-shuda/btl_descriptor/lib/" )
    if(MSVC)
        set(BTLDESCRIPTORLIB optimized BtlDescriptor debug BtlDescriptord)
    endif()
endif()
#set( CUDA_NVCC_FLAGS "-arch=sm_21" ) #allow atomicAdd operators
cuda_add_executable( ShudaOrb Main.cpp)

target_link_libraries( ShudaOrb ${OpenCV_LIBS}  ${BTLDESCRIPTORLIB}  )
#install( TARGETS ShudaOrb DESTINATION ${PROJECT_SOURCE_DIR} )


