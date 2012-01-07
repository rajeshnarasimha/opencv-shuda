cmake_minimum_required (VERSION 2.6)

project (ShudaLib)
FIND_PACKAGE( OpenCV REQUIRED )

if( UNIX )
    message(STATUS "platform: UNIX")

	include_directories ( "/usr/local/include/eigen3" )
	include_directories ( "/usr/include/ni" )
elseif( WIN32 )
    message(STATUS "platform: Windows")

	set( EIGEN_INCLUDE_DIRS "C:/csxsl/install/eigen/include/eigen3" )
    include_directories ( ${EIGEN_INCLUDE_DIRS} )
	include_directories ( "C:/Program Files/OpenNI/Include" )
    include_directories ( "C:/csxsl/install/boost_1_48_0/include" ) 
    include_directories ( "C:/csxsl/install/freeglut_x64/include" )
    include_directories ( "C:/csxsl/install/yaml-cpp-0.2.7/include" )
endif()
find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
#find all files
file(GLOB HEADERS "[a-zA-Z]*.h" "[a-zA-Z]*.hpp" )
message(STATUS "header names: ${HEADERS}")
file(GLOB SOURCES "[a-zA-Z]*.cpp")
message(STATUS "source names: ${SOURCES}")
file(GLOB TESTERS "test/[a-zA-Z]*.cpp")
message(STATUS "tester names: ${TESTERS}")

add_library( ShudaLib ${SOURCES} ${HEADERS} )

add_executable( TestShudaLib ${TESTERS} )
target_link_libraries( TestShudaLib ${OpenCV_LIBS} )

if(MSVC)
	if(NOT BUILD_SHARED_LIBS)
		# correct library names
		set_target_properties(ShudaLib PROPERTIES
			DEBUG_POSTFIX "${LIB_TARGET_SUFFIX}d"
			RELEASE_POSTFIX "${LIB_TARGET_SUFFIX}"
			MINSIZEREL_POSTFIX "${LIB_TARGET_SUFFIX}"
			RELWITHDEBINFO_POSTFIX "${LIB_TARGET_SUFFIX}"
		)
        # correct executable names
		set_target_properties(TestShudaLib PROPERTIES
			DEBUG_POSTFIX "${LIB_TARGET_SUFFIX}d"
			RELEASE_POSTFIX "${LIB_TARGET_SUFFIX}"
			MINSIZEREL_POSTFIX "${LIB_TARGET_SUFFIX}"
			RELWITHDEBINFO_POSTFIX "${LIB_TARGET_SUFFIX}"
		)
	endif()
endif()

#install 
install( TARGETS ShudaLib  DESTINATION "${PROJECT_SOURCE_DIR}/lib" )
install( TARGETS TestShudaLib DESTINATION "${PROJECT_SOURCE_DIR}/bin" )
install( FILES ${HEADERS} DESTINATION "${PROJECT_SOURCE_DIR}/include")


