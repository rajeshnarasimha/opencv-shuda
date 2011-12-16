cmake_minimum_required (VERSION 2.6)
project (ImageViewer)

#include_directories ( "d:/MinGW/msys/1.0/home/sdli/install/opencv/build/include" )
#link_directories ( "d:/MinGW/msys/1.0/home/sdli/install/opencv/build/x86/mingw/lib" )
include_directories ( "/space/csxsl/src/opencv-shuda/ShudaLib" )
link_directories ( "/space/csxsl/src/opencv-shuda/ShudaLib" )
link_directories ( "/usr/lib" )
include_directories ( "/usr/local/include/eigen3" )

# set collection of libraries
#set ( EXTRA_LIBS ${EXTRA_LIBS} opencv_highgui231.dll opencv_core231.dll)
set ( EXTRA_LIBS ${EXTRA_LIBS} ShudaLib opencv_core opencv_highgui opencv_calib3d opencv_features2d opencv_video opencv_imgproc )

add_executable(ImageViewer main.cpp)
# link with the 3d party libraries
target_link_libraries( ImageViewer ${EXTRA_LIBS} )



