project( FreakKeypointMatcher )
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
cuda_add_executable( FreakKeypointMatcher FreakKeypointMatcher.cpp)
target_link_libraries( FreakKeypointMatcher ${OpenCV_LIBS} ${BTLDESCRIPTORLIB} )

install( TARGETS FreakKeypointMatcher DESTINATION ${PROJECT_SOURCE_DIR} )


