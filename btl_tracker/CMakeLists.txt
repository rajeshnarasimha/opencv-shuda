project( BtlTracker )
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
include(FindCUDA)
find_package(Boost)

IF (Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
    ADD_DEFINITIONS( "-DHAS_BOOST" )
ENDIF()
if( WIN32 )
	include_directories ( $ENV{EIGEN_INCLUDE_DIR} )
endif()
cuda_add_library( BtlTracker SemiDenseTracker.cpp SemiDenseTracker.h  
SemiDenseTracker.cu SemiDenseTracker.cuh 
SemiDenseTrackerOrb.cu SemiDenseTrackerOrb.cuh
SemiDenseTrackerOrb.cpp SemiDenseTrackerOrb.h
SemiDenseTrackerFHessFreak.cu SemiDenseTrackerFHessFreak.cuh
SemiDenseTrackerFHessFreak.cpp SemiDenseTrackerFHessFreak.h
TestCudaOrb.cpp TestCudaOrb.h
TestCudaFast.cpp TestCudaFast.h
CudaHelper.hpp Helper.hpp Helper.cpp
TrackerSimpleFreak.h TrackerSimpleFreak.cpp
FullFrameAlignment.cu FullFrameAlignment.cuh
)

if(MSVC)
	if(NOT BUILD_SHARED_LIBS)
		# correct library names
		set_target_properties(BtlTracker PROPERTIES
			DEBUG_POSTFIX "${LIB_TARGET_SUFFIX}d"
			RELEASE_POSTFIX "${LIB_TARGET_SUFFIX}"
			MINSIZEREL_POSTFIX "${LIB_TARGET_SUFFIX}m"
			RELWITHDEBINFO_POSTFIX "${LIB_TARGET_SUFFIX}r"
		)
	endif()
endif()

install( TARGETS BtlTracker  DESTINATION "${PROJECT_SOURCE_DIR}/lib" )

