PROJECT( VideoConverter )
cmake_minimum_required(VERSION 2.8)
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE( VideoConverter VideoConverter.cpp )
TARGET_LINK_LIBRARIES( VideoConverter ${OpenCV_LIBS} )
#install( TARGETS WebCam DESTINATION "${PROJECT_SOURCE_DIR}" )

