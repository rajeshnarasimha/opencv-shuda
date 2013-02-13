project( BtlDescriptor )
cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )
find_package( CUDA )
include(FindCUDA)
cuda_add_library( BtlDescriptor 
Freak.cpp Freak.h Freak.cu Freak.cuh TestFreak.cpp TestFreak.h
Surf.h Surf.cu Surf.cpp Surf.cuh 
Orb.cpp Orb.cu Orb.h
Fast.h Fast.cpp Fast.cu
)
#target_link_libraries( BtlDescriptor ${OpenCV_LIBS} )

if(MSVC)
	if(NOT BUILD_SHARED_LIBS)
		# correct library names
		set_target_properties(BtlDescriptor PROPERTIES
			DEBUG_POSTFIX "${LIB_TARGET_SUFFIX}d"
			RELEASE_POSTFIX "${LIB_TARGET_SUFFIX}"
			MINSIZEREL_POSTFIX "${LIB_TARGET_SUFFIX}m"
			RELWITHDEBINFO_POSTFIX "${LIB_TARGET_SUFFIX}r"
		)
	endif()
endif()

install( TARGETS BtlDescriptor  DESTINATION "${PROJECT_SOURCE_DIR}/lib" )

