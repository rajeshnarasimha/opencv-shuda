cmake_minimum_required (VERSION 2.6)
project (SingleFrame3DFitting)

find_package(OpenCV REQUIRED)

if( UNIX )
    include_directories ( "/space/csxsl/src/opencv-shuda/ShudaLib/include" )
    link_directories ( "/space/csxsl/src/opencv-shuda/ShudaLib/lib" )
    link_directories ( "/usr/lib" )
    include_directories ( "/usr/local/include/eigen3" )
    include_directories ( "/usr/include/ni" )

    # set collection of libraries
    set ( EXTRA_LIBS ${EXTRA_LIBS} ShudaLib OpenNI boost_system boost_filesystem boost_serialization yaml-cpp glut GLU opencv_core
    opencv_highgui opencv_calib3d opencv_features2d opencv_video opencv_imgproc GLEW)

elseif( WIN32 )
    set( EIGEN_INCLUDE_DIRS "C:/csxsl/install/eigen/include/eigen3" )
    include_directories ( ${EIGEN_INCLUDE_DIRS} )
	include_directories ( "C:/Program Files/OpenNI/Include" )
    include_directories ( "C:/csxsl/install/boost_1_48_0/include" ) 
    include_directories ( "C:/csxsl/install/freeglut_x64/include" )
    include_directories ( "C:/csxsl/install/yaml-cpp-0.2.7/include" )
    include_directories ( "C:/csxsl/src/opencv-shuda/ShudaLib/include" )
    include_directories ( "C:/csxsl/install/glew_170_win64/include" )
    
    link_directories ( "C:/Program Files/OpenNI/Lib64" )
    link_directories ( "C:/csxsl/src/opencv-shuda/ShudaLib/lib" )
    link_directories ( "C:/csxsl/install/boost_1_48_0/lib" )
    link_directories ( "C:/csxsl/install/freeglut_x64/lib/x64" )
    link_directories ( "C:/csxsl/install/yaml-cpp-0.2.7/lib" )
    link_directories ( "C:/csxsl/install/glew_170_win64/lib" )
    if(MSVC)
        set(SHUDALIB optimized ShudaLib debug ShudaLibd)
    	set(YAML-CPP optimized yaml-cpp debug yaml-cppd)
        set ( EXTRA_LIBS ${EXTRA_LIBS} ${SHUDALIB} OpenNI64 boost_system-vc100-mt-1_48 boost_filesystem-vc100-mt-1_48
        boost_serialization-vc100-mt-1_48 freeglut GLU32 ${OpenCV_LIBS} ${YAML-CPP} glew32)
    endif()
endif()

find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(SingleFrame3DFitting SingleFrame3DFitting.cpp)
# link with the 3d party libraries
target_link_libraries( SingleFrame3DFitting ${PCL_LIBRARIES} ${EXTRA_LIBS} )
