cmake_minimum_required (VERSION 2.6)

project (ShudaLib)
include_directories ( "/usr/local/include/eigen3" )
include_directories ( "/usr/include/ni" )
#add executable
add_library( ShudaLib  calibratekinect.cpp optim.cpp VideoSourceKinect.cpp OsgSceneMarkers.cpp)

include_directories ( "/usr/local/include" )# for osg
add_library( OsgRender OsgSceneMarkers.cpp )

install( TARGETS ShudaLib OsgRender DESTINATION "${PROJECT_SOURCE_DIR}" )

