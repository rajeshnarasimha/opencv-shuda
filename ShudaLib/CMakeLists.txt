cmake_minimum_required (VERSION 2.6)

project (ShudaLib)
find_package( OpenCV REQUIRED )

find_package( CUDA )
include(FindCUDA)

if( UNIX )
    message(STATUS "platform: UNIX")
    include_directories ( ${EIGEN_INCLUDE_DIRS} )

	include_directories ( "/usr/local/include/eigen3" )
	include_directories ( "/usr/include/ni" )
    set( FLANN_LIBRARY "/usr/local/lib64/libflann.so" )
elseif( WIN32 )
    message(STATUS "platform: Windows")

	set( EIGEN_INCLUDE_DIRS "C:/csxsl/install/eigen/include/eigen3" )
    include_directories ( ${EIGEN_INCLUDE_DIRS} )
    include_directories ( "C:/Program Files/OpenNI/Include" )
    include_directories ( "C:/csxsl/install/boost_1_48_0/include" ) 
    include_directories ( "C:/csxsl/install/freeglut_x64/include" )
    include_directories ( "C:/csxsl/install/yaml-cpp/include" )

	link_directories ( "C:/csxsl/install/freeglut_x64/lib/x64" )
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
file(GLOB CUDAHEADERS "cuda/[a-zA-Z]*.h")
message(STATUS "cuda sources names: ${CUDAHEADERS}")

cuda_add_library( CudaLib ${CUDASOURCES} ${CUDAHEADERS} )
add_library( ShudaLib ${SOURCES} ${HEADERS} )
 
if(MSVC)
	if(NOT BUILD_SHARED_LIBS)
		# correct library names
		set_target_properties(ShudaLib PROPERTIES
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
install( TARGETS ShudaLib  DESTINATION "${PROJECT_SOURCE_DIR}/lib" )
install( TARGETS CudaLib   DESTINATION "${PROJECT_SOURCE_DIR}/lib" )
install( FILES ${HEADERS} ${CUDAHEADERS} DESTINATION "${PROJECT_SOURCE_DIR}/include")



add_executable( TestShudaLib ${TESTERS} )
target_link_libraries( TestShudaLib ${OpenCV_LIBS} ShudaLib CudaLib freeglut)
