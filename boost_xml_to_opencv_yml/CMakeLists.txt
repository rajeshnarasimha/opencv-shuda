PROJECT( BoostXML2CVYML )
cmake_minimum_required(VERSION 2.8)
FIND_PACKAGE( OpenCV REQUIRED )

if( UNIX )
# set collection of libraries
    set ( EXTRA_LIBS ${EXTRA_LIBS} ShudaLib OpenNI boost_system boost_filesystem boost_serialization yaml-cpp glut GLU opencv_core
opencv_highgui opencv_calib3d opencv_features2d opencv_video opencv_imgproc )
elseif( WIN32 )
    include_directories ( "C:/csxsl/install/boost_1_48_0/include" ) 
    include_directories ( "C:/csxsl/src/opencv-shuda/ShudaLib" )
    include_directories ( "C:/csxsl/install/eigen/include/eigen3" )
    
    link_directories ( "C:/csxsl/install/boost_1_48_0/lib" )
    link_directories ( "C:/csxsl/src/opencv-shuda/ShudaLib" )
    
    set ( EXTRA_LIBS ${EXTRA_LIBS}     boost_serialization-vc100-mt-1_48 ShudaLib )
endif()		

ADD_EXECUTABLE( BoostXML2CVYML BoostXML2CVYML.cpp )
TARGET_LINK_LIBRARIES( BoostXML2CVYML ${OpenCV_LIBS} ${EXTRA_LIBS} )
install( TARGETS BoostXML2CVYML DESTINATION "${PROJECT_SOURCE_DIR}" )



