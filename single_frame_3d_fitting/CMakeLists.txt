cmake_minimum_required (VERSION 2.6)
project (SingleFrame3DFitting)

include_directories ( "/space/csxsl/src/opencv-shuda/ShudaLib" )
link_directories ( "/space/csxsl/src/opencv-shuda/ShudaLib" )

include_directories ( "/usr/local/include/eigen3" )
include_directories ( "/usr/include/ni" )

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# set collection of libraries
set ( EXTRA_LIBS ${EXTRA_LIBS} ShudaLib OpenNI boost_system boost_filesystem boost_serialization yaml-cpp glut GLU opencv_core
opencv_highgui opencv_calib3d opencv_features2d opencv_video opencv_imgproc )

add_executable(SingleFrame3DFitting SingleFrame3DFitting.cpp)
# link with the 3d party libraries
target_link_libraries( SingleFrame3DFitting ${PCL_LIBRARIES} ${EXTRA_LIBS} )
