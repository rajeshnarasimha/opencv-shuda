cmake_minimum_required (VERSION 2.6)

project (KinectCapturerOpenNI)
find_package( OpenCV REQUIRED )

if( UNIX )
    message(STATUS "platform: UNIX")
    include_directories ( ${EIGEN_INCLUDE_DIRS} )

	include_directories ( "/usr/local/include/eigen3" )
	include_directories ( "/usr/include/ni" )
elseif( WIN32 )
    message(STATUS "platform: Windows")

	include_directories ( $ENV{OPEN_NI_INCLUDE} )
    include_directories ( $ENV{BOOST_INCLUDE_DIR} ) 
    include_directories ( $ENV{EIGEN_INCLUDE_DIR} )
    include_directories ( $ENV{OPEN_NI_INCLUDE} )
    link_directories ( $ENV{OPEN_NI_LIB_DIR}  )
    include_directories ( "C:/csxsl/src/opencv-shuda/btl_rgbd/include" )
endif()

add_executable(KinectCapturerOpenNI KinectCapturerOpenNI.cpp KinectCapturerOpenNI.h)
# link with the 3d party libraries
target_link_libraries( KinectCapturerOpenNI OpenNI64 ${OpenCV_LIBS} )
