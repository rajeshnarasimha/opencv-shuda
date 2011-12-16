cmake_minimum_required (VERSION 2.6)
project (CameraCalib)


# set collection of libraries
set ( EXTRA_LIBS ${EXTRA_LIBS} opencv_core
opencv_highgui opencv_calib3d opencv_features2d opencv_video opencv_imgproc )

add_executable(CameraCalib CameraCalib.cpp)
target_link_libraries( CameraCalib ${EXTRA_LIBS} )# link with the 3d party libraries

# copy the compiled lib file to destination folder 
install( TARGETS CameraCalib DESTINATION "${PROJECT_SOURCE_DIR}" )
