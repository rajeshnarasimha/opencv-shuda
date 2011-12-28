PROJECT( example )
cmake_minimum_required(VERSION 2.8)
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE( example main.cpp )
TARGET_LINK_LIBRARIES( example ${OpenCV_LIBS} )
install( TARGETS example DESTINATION "${PROJECT_SOURCE_DIR}" )



#cmake_minimum_required (VERSION 2.6)
#project (example)

#find_package (OpenCV COMPONENTS opencv_highgui232)
#if ( NOT OpenCV_FOUND )
#  message(FATAL_ERROR "Package OpenCV and component opencv_highgui required, but not found!")
#endif( NOT OpenCV_FOUND )


#if (OpenCV_FOUND)
#	include_directories(${OpenCV_INCLUDE_DIRS})
#	add_executable(AVIPlayer main.cpp)
#	target_link_libraries (AVIPlayer opencv_highgui232)
#endif (OpenCV_FOUND)

#include_directories ( "C:/csxsl/install/opencv_231/build/install/install/include" )
#link_directories ( "C:/csxsl/install/opencv_231/build/install/lib/Debug" "C:/csxsl/install/opencv_231/build/lib/Release" )




# set collection of libraries
#set ( EXTRA_LIBS ${EXTRA_LIBS} opencv_highgui231.dll)

#add_executable(AVIPlayer main.cpp)
# link with the 3d party libraries
#target_link_libraries( AVIPlayer ${EXTRA_LIBS} )
