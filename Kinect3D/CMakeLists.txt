cmake_minimum_required (VERSION 2.6)
project (Kinect3D)

include_directories ( "../ShudaLib" )
link_directories ( "../ShudaLib" )

include_directories ( "/usr/local/include/eigen3" )
include_directories ( "/usr/include/ni" )


add_executable(Kinect3D main.cpp)
target_link_libraries( Kinect3D ShudaLib OpenNI boost_system boost_filesystem boost_serialization yaml-cpp glut GLU opencv_core
opencv_highgui opencv_calib3d opencv_features2d opencv_video opencv_imgproc )
