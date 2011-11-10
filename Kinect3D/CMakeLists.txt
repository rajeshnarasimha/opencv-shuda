cmake_minimum_required (VERSION 2.6)
project (Kinect3D)

include_directories ( "/space/csxsl/src/opencv-shuda/ShudaLib" )
link_directories ( "/space/csxsl/src/opencv-shuda/ShudaLib" )

include_directories ( "/usr/local/include/eigen3" )
include_directories ( "/usr/include/ni" )

# set collection of libraries
set ( EXTRA_LIBS ${EXTRA_LIBS} ShudaLib OpenNI boost_system boost_filesystem boost_serialization yaml-cpp glut GLU opencv_core
opencv_highgui opencv_calib3d opencv_features2d opencv_video opencv_imgproc )

add_executable(Kinect3D main.cpp)
# link with the 3d party libraries
target_link_libraries( Kinect3D ${EXTRA_LIBS} )
