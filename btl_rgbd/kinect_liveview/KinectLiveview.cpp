//display kinect depth in real-time
#define INFO

#include <GL/glew.h>
#include <gl/freeglut.h>
//#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <string>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>

#include "Converters.hpp"
#include <opencv2/gpu/gpu.hpp>
#include <XnCppWrapper.h>

#include "Kinect.h"
#include <gl/freeglut.h>
#include "Camera.h"

#include "EigenUtil.hpp"
#include "GLUtil.h"
#include "PlaneObj.h"
#include "Histogram.h"
#include "SemiDenseTracker.h"
#include "SemiDenseTrackerOrb.h"
#include "KeyFrame.h"
#include "CyclicBuffer.h"
#include "VideoSourceKinect.hpp"
#include "CubicGrids.h"
#include "GLUtil.h"
//camera calibration from a sequence of images
#include "Camera.h"

btl::kinect::VideoSourceKinect::tp_shared_ptr _pKinect;
btl::gl_util::CGLUtil::tp_shared_ptr _pGL;

unsigned short _nWidth, _nHeight;
int _nDensity = 2;

ushort _uResolution = 0;
ushort _uPyrHeight = 3;
Eigen::Vector3f _eivCw(1.5f,1.5f,-0.3f);
bool _bUseNIRegistration = true;
ushort _uCubicGridResolution = 512;
float _fVolumeSize = 3.f;
int _nMode = 3;//btl::kinect::VideoSourceKinect::PLAYING_BACK
std::string _oniFileName("x.oni"); // the openni file 
bool _bRepeat = false;// repeatedly play the sequence 
int _nRecordingTimeInSecond = 30;
float _fTimeLeft = _nRecordingTimeInSecond;
int _nStatus = 01;//1 restart; 2 //recording continue 3://pause 4://dump
bool _bDisplayImage = false;
bool _bLightOn = false;
bool _bRenderReference = false;


void loadFromYml(){

#if __linux__
	cv::FileStorage cFSRead( "/space/csxsl/src/opencv-shuda/Data/kinect_intrinsics.yml", cv::FileStorage::READ );
#else if _WIN32 || _WIN64
	cv::FileStorage cFSRead ( "C:\\csxsl\\src\\opencv-shuda\\btl_rgbd\\kinect_liveview\\KinectLiveview.yml", cv::FileStorage::READ );
#endif
	cFSRead["uResolution"] >> _uResolution;
	cFSRead["uPyrHeight"] >> _uPyrHeight;
	cFSRead["bUseNIRegistration"] >> _bUseNIRegistration;
	cFSRead["uCubicGridResolution"] >> _uCubicGridResolution;
	cFSRead["fVolumeSize"] >> _fVolumeSize;
	//rendering
	cFSRead["bDisplayImage"] >> _bDisplayImage;
	cFSRead["bLightOn"] >> _bLightOn;
	cFSRead["bRenderReference"] >> _bRenderReference;
	cFSRead["nMode"] >> _nMode;//1 kinect; 2 recorder; 3 player
	cFSRead["oniFile"] >> _oniFileName;
	cFSRead["bRepeat"] >> _bRepeat;
	cFSRead["nRecordingTimeInSecond"] >> _nRecordingTimeInSecond;
	cFSRead["nStatus"] >> _nStatus;

	cFSRead.release();
}
void saveToYml(){

#if __linux__
	cv::FileStorage cFSWrite( "/space/csxsl/src/opencv-shuda/Data/kinect_intrinsics.yml", cv::FileStorage::WRITE );
#else if _WIN32 || _WIN64
	cv::FileStorage cFSWrite ( "C:\\csxsl\\src\\opencv-shuda\\btl_rgbd\\kinect_liveview\\KinectLiveview.yml", cv::FileStorage::WRITE );
#endif

	cFSWrite << "uResolution" << _uResolution;
	cFSWrite << "uPyrHeight" << _uPyrHeight;

	cFSWrite << "bUseNIRegistration" << _bUseNIRegistration;
	cFSWrite << "uCubicGridResolution" << _uCubicGridResolution;
	cFSWrite << "fVolumeSize" << _fVolumeSize;
	//rendering
	cFSWrite << "bDisplayImage" << _pGL->_bDisplayCamera;
	cFSWrite << "bLightOn"  << _pGL->_bEnableLighting;
	cFSWrite << "bRenderReference" << _pGL->_bRenderReference;
	cFSWrite << "nMode" <<  _nMode;//1 kinect; 2 recorder; 3 player
	cFSWrite << "oniFile" << _oniFileName;
	cFSWrite << "bRepeat" << _bRepeat;
	cFSWrite << "nRecordingTimeInSecond" << _nRecordingTimeInSecond;
	cFSWrite << "nStatus" << _nStatus;

	cFSWrite.release();
}
void init ( ){
	//load parameters
	loadFromYml();
	//initialize rendering environment
	_pGL.reset( new btl::gl_util::CGLUtil(_uResolution,_uPyrHeight,btl::utility::BTL_GL) );
	_pGL->_bDisplayCamera = _bDisplayImage;
	_pGL->_bEnableLighting = _bLightOn;
	_pGL->_bRenderReference = _bRenderReference;
	_pGL->clearColorDepth();
	_pGL->init();
	_pGL->constructVBOsPBOs();
	//set opengl flags
	glDepthFunc  ( GL_LESS );
	glEnable     ( GL_DEPTH_TEST );
	glEnable 	 ( GL_SCISSOR_TEST );
	glEnable     ( GL_CULL_FACE );
	glShadeModel ( GL_FLAT );
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	//initialize rgbd sensor
	_pKinect.reset(new btl::kinect::VideoSourceKinect(_uResolution,_uPyrHeight,_bUseNIRegistration,_eivCw));
	switch(_nMode)
	{
	case btl::kinect::VideoSourceKinect::SIMPLE_CAPTURING: //the simple capturing mode of the rgbd camera
		_pKinect->initKinect();
		break;
	case btl::kinect::VideoSourceKinect::RECORDING: //record the captured sequence from the camera
		_pKinect->setDumpFileName(_oniFileName);
		_pKinect->initRecorder(_oniFileName,_nRecordingTimeInSecond);
		break;
	case btl::kinect::VideoSourceKinect::PLAYING_BACK: //replay from files
		_pKinect->initPlayer(_oniFileName,_bRepeat);
		break;
	default://only simply capturing and playing back mode are allowed for efficiency requirements
		_nMode = btl::kinect::VideoSourceKinect::SIMPLE_CAPTURING;
		_pKinect->initKinect();
		break;
	}
	//capture the first frame
	_pKinect->getNextFrame(btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV,&_nStatus);
	_pKinect->_pFrame->gpuTransformToWorldCVCV();
	_pKinect->_pFrame->setView(&_pGL->_eimModelViewGL);

	return;
}//init()

void specialKeys( int key, int x, int y ){
	_pGL->specialKeys( key, x, y );
}

void normalKeys ( unsigned char key, int x, int y )
{
	switch( key )
	{
	case 27:
		saveToYml();
		exit ( 0 );
		break;
	case 'q':
		_nDensity++;
		glutPostRedisplay();
		PRINT( _nDensity );
		break;
	case ';':
		_nDensity--;
		_nDensity = _nDensity > 0 ? _nDensity : 1;
		glutPostRedisplay();
		PRINT( _nDensity );
		break;
	case '0':
		_pKinect->_pFrame->setView(&_pGL->_eimModelViewGL);
		_pGL->setInitialPos();
		break;
	case 's'://dump recording in RECORDING MODE
		_nStatus = (_nStatus&(~btl::kinect::VideoSourceKinect::MASK_RECORDER))|btl::kinect::VideoSourceKinect::DUMP_RECORDING;
		glutPostRedisplay();
		break;
	case 'p'://pause/continue switcher for all 3 modes
		if ((_nStatus&btl::kinect::VideoSourceKinect::MASK1) == btl::kinect::VideoSourceKinect::PAUSE){
			_nStatus = (_nStatus&(~btl::kinect::VideoSourceKinect::MASK1))|btl::kinect::VideoSourceKinect::CONTINUE;
		}else if ((_nStatus&btl::kinect::VideoSourceKinect::MASK1) == btl::kinect::VideoSourceKinect::CONTINUE){
			_nStatus = (_nStatus&(~btl::kinect::VideoSourceKinect::MASK1))|btl::kinect::VideoSourceKinect::PAUSE;
		}
		glutPostRedisplay();
		break;
	case 'c'://start recording in RECORDING mode 
		_nStatus = (_nStatus&(~btl::kinect::VideoSourceKinect::MASK_RECORDER))|btl::kinect::VideoSourceKinect::START_RECORDING;
		glutPostRedisplay();
		break;
	case 'r':
		if (_nMode == btl::kinect::VideoSourceKinect::PLAYING_BACK)	{
			_pKinect->initPlayer(_oniFileName,_bRepeat);
			_nStatus = (_nStatus&(~btl::kinect::VideoSourceKinect::MASK1))|btl::kinect::VideoSourceKinect::CONTINUE;
		}
		glutPostRedisplay();
		break;
	case 'R':
		//replay
		init();
		break;
	case ']':
		_pKinect->_fSigmaSpace += 1;
		PRINT( _pKinect->_fSigmaSpace );
		break;
	case '[':
		_pKinect->_fSigmaSpace -= 1;
		PRINT( _pKinect->_fSigmaSpace );
		break;
    }
	_pGL->normalKeys( key, x, y);
    return;
}
void mouseClick ( int nButton_, int nState_, int nX_, int nY_ )
{
	_pGL->mouseClick( nButton_, nState_ ,nX_,nY_ );
	return;
}
void mouseMotion ( int nX_, int nY_ )
{
	_pGL->mouseMotion( nX_,nY_ );
	return;
}

void display ( void )
{
	//load data from video source and model
	_pKinect->getNextFrame(btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV,&_nStatus);
	_pKinect->_pFrame->gpuTransformToWorldCVCV();
	//set viewport
    glMatrixMode ( GL_MODELVIEW );
	glViewport (0, 0, _nWidth/2, _nHeight);
	glScissor  (0, 0, _nWidth/2, _nHeight);
	// after set the intrinsics and extrinsics
    // load the matrix to set camera pose
	_pGL->viewerGL();	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// render objects
    _pGL->renderAxisGL();
	_pGL->renderPatternGL(.1f,20,20);
	_pGL->renderPatternGL(1.f,10,10);
	_pGL->renderVoxelGL(_fVolumeSize);
	//_pGL->timerStart();
	_pKinect->_pFrame->gpuRenderPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel);
	//show text
	float aColor[4] = {0.f,1.f,0.f,1.f};
	switch(_nMode){ 
	case btl::kinect::VideoSourceKinect::RECORDING:
		_pGL->drawString("Recorder", 5, _nHeight-20, aColor, GLUT_BITMAP_8_BY_13);
		if ( (_nStatus&btl::kinect::VideoSourceKinect::MASK_RECORDER) == btl::kinect::VideoSourceKinect::CONTINUE_RECORDING ){
			float aColor[4] = {1.f,0.f,0.f,1.f};
			_pGL->drawString("Recording...", 5, _nHeight-40, aColor, GLUT_BITMAP_8_BY_13);
		}
		break;
	case btl::kinect::VideoSourceKinect::PLAYING_BACK:
		_pGL->drawString("Player", 5, _nHeight-20, aColor, GLUT_BITMAP_8_BY_13);
		break;
	case btl::kinect::VideoSourceKinect::SIMPLE_CAPTURING:
		_pGL->drawString("Simple", 5, _nHeight-20, aColor, GLUT_BITMAP_8_BY_13);
		break;
	}
	//set viewport 2
	glViewport (_nWidth/2, 0, _nWidth/2, _nHeight);
	glScissor  (_nWidth/2, 0, _nWidth/2, _nHeight);
	glLoadIdentity();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    _pGL->renderAxisGL();
	_pKinect->_pRGBCamera->LoadTexture(*_pKinect->_pFrame->_acvmShrPtrPyrRGBs[_pGL->_usLevel],&_pGL->_auTexture[_pGL->_usLevel]);
	_pKinect->_pRGBCamera->renderCameraInGLLocal(_pGL->_auTexture[_pGL->_usLevel] );
	
	
	glutSwapBuffers();
	glutPostRedisplay();
	return;
}
void reshape ( int nWidth_, int nHeight_ ){
	//cout << "reshape() " << endl;
    _pKinect->_pRGBCamera->setGLProjectionMatrix( 1, 0.01, 100 );

    // setup blending 
    glBlendFunc ( GL_SRC_ALPHA, GL_ONE );			// Set The Blending Function For Translucency
    glColor4f ( 1.0f, 1.0f, 1.0f, 0.5 );

	unsigned short nTemp = nWidth_/8;//make sure that _nWidth is divisible to 4
	_nWidth = nTemp*8;
	_nHeight = nTemp*3;
	glutReshapeWindow( int ( _nWidth ), int ( _nHeight ) );
    return;
}
int main ( int argc, char** argv ){
    try {
		glutInit ( &argc, argv );
        glutInitDisplayMode ( GLUT_DOUBLE | GLUT_RGB );
        glutInitWindowSize ( 1280, 480 );
        glutCreateWindow ( "CameraPose" );
		GLenum eError = glewInit();
		if (GLEW_OK != eError){
			PRINTSTR("glewInit() error.");
			PRINT( glewGetErrorString(eError) );
		}
        glutKeyboardFunc( normalKeys );
		glutSpecialFunc ( specialKeys );
        glutMouseFunc   ( mouseClick );
        glutMotionFunc  ( mouseMotion );

		glutReshapeFunc ( reshape );
        glutDisplayFunc ( display );

		btl::gl_util::CGLUtil::initCuda();
		btl::gl_util::CGLUtil::setCudaDeviceForGLInteroperation();
		
		init();
		glutMainLoop();
		_pGL->destroyVBOsPBOs();
	}
    catch ( btl::utility::CError& e )  {
        if ( std::string const* mi = boost::get_error_info< btl::utility::CErrorInfo > ( e ) ) {
            std::cerr << "Error Info: " << *mi << std::endl;
        }
    }
	catch ( std::runtime_error& e ){
		PRINTSTR( e.what() );
	}

    return 0;
}
