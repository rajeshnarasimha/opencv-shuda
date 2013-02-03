cmake_minimum_required (VERSION 2.6)

project (BtlRgbd)
find_package( OpenCV REQUIRED )

find_package( CUDA )
include(FindCUDA)

#find_package(Boost 1.51.0 REQUIRED COMPONENTS system filesystem serialization date_time)

if( UNIX )
    message(STATUS "platform: UNIX")
    include_directories ( ${EIGEN_INCLUDE_DIRS} )

	include_directories ( "/usr/local/include/eigen3" )
	include_directories ( "/usr/include/ni" )
    set( FLANN_LIBRARY "/usr/local/lib64/libflann.so" )
elseif( WIN32 )
    message(STATUS "platform: Windows")

	include_directories ( $ENV{EIGEN_INCLUDE_DIR} )
	include_directories ( $ENV{OPEN_NI_INCLUDE} )
    include_directories ( $ENV{BOOST_INCLUDE_DIR} ) 
    include_directories ( $ENV{FREEGLUT_INCLUDE_DIR} )
    #include_directories ( $ENV{GLFW_INC} )
    include_directories ( $ENV{YAML-CPP_INCLUDE_DIR} )
    include_directories ( $ENV{GLEW_INCLUDE_DIR} )
	include_directories ( $ENV{CUDA_INC_PATH} )
	include_directories ( $ENV{CUDALIB} )
	include_directories ( $ENV{PCL_INC} )

	link_directories ( $ENV{FREEGLUT_LIB_DIR} )
endif()

find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
#find all files

file(GLOB HEADERS "[a-zA-Z]*.h" "[a-zA-Z]*.hpp"   )
message(STATUS "header names: ${HEADERS}")
file(GLOB SOURCES "[a-zA-Z]*.cpp")
message(STATUS "source names: ${SOURCES}")
file(GLOB TESTERS "test/[a-zA-Z]*.cpp")
message(STATUS "tester names: ${TESTERS}")
file(GLOB CUDASOURCES "cuda/[a-zA-Z]*.cu")
message(STATUS "cuda sources names: ${CUDASOURCES}")
file(GLOB CUDAHEADERS "cuda/[a-zA-Z]*.h"  "cuda/cv/[a-zA-Z]*.hpp" "cuda/pcl/[a-zA-Z]*.hpp" "cuda/pcl/[a-zA-Z]*.h"
"cuda/[a-zA-Z]*.cuh")
message(STATUS "cuda sources names: ${CUDAHEADERS}")

set( CUDA_NVCC_FLAGS "-arch=sm_21" ) #allow atomicAdd operators
cuda_add_library( CudaLib ${CUDASOURCES} ${CUDAHEADERS} )
add_library( BtlRgbd ${SOURCES} ${HEADERS} )
 
if(MSVC)
	if(NOT BUILD_SHARED_LIBS)
		# correct library names
		set_target_properties(BtlRgbd PROPERTIES
			DEBUG_POSTFIX "${LIB_TARGET_SUFFIX}d"
			RELEASE_POSTFIX "${LIB_TARGET_SUFFIX}"
			MINSIZEREL_POSTFIX "${LIB_TARGET_SUFFIX}"
			RELWITHDEBINFO_POSTFIX "${LIB_TARGET_SUFFIX}"
		)
		# correct library names
        set_target_properties(CudaLib PROPERTIES
			DEBUG_POSTFIX "${LIB_TARGET_SUFFIX}d"
			RELEASE_POSTFIX "${LIB_TARGET_SUFFIX}"
			MINSIZEREL_POSTFIX "${LIB_TARGET_SUFFIX}"
			RELWITHDEBINFO_POSTFIX "${LIB_TARGET_SUFFIX}"
		)
	endif()
endif()

#install 
install( TARGETS BtlRgbd  DESTINATION "${PROJECT_SOURCE_DIR}/lib" )
install( TARGETS CudaLib   DESTINATION "${PROJECT_SOURCE_DIR}/lib" )
#install( FILES ${HEADERS} ${CUDAHEADERS} DESTINATION "${PROJECT_SOURCE_DIR}/include")

add_executable( TestBtlRgbd ${TESTERS} )
target_link_libraries( TestBtlRgbd ${OpenCV_LIBS} freeglut BtlRgbd CudaLib ) #${Boost_DATE_TIME_LIBRARY})#

include("kinect_fusion_qglviewer/CMakeLists.txt")
include("kinect_liveview/CMakeLists.txt")
include("kinect_fusion/CMakeLists.txt")
include("render_normal/CMakeLists.txt")
include("render_qglviewer/CMakeLists.txt")
include("classifier/CMakeLists.txt")
include( "kinect_fusion_simulator/CMakeLists.txt" )
