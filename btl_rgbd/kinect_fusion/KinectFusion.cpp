//display kinect depth in real-time
#define INFO
#include <GL/glew.h>
#include <gl/freeglut.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
//stl
#include <iostream>
#include <string>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
//boost
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
//openncv
#include <opencv2/gpu/gpumat.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <utility>
#include <boost/lexical_cast.hpp>
#include <gl/freeglut.h>
#include <XnCppWrapper.h>
#include "Converters.hpp"
#include "GLUtil.h"
#include "EigenUtil.hpp"
#include "Camera.h"
#include "GLUtil.h"
#include "PlaneObj.h"
#include "Histogram.h"
#include "KeyFrame.h"
#include "CyclicBuffer.h"
#include "VideoSourceKinect.hpp"
#include "CubicGrids.h"
#include "KinfuTracker.h"
#define _nReserved 5

btl::kinect::VideoSourceKinect::tp_shared_ptr _pKinect;
btl::gl_util::CGLUtil::tp_shared_ptr _pGL;
btl::geometry::CKinFuTracker::tp_shared_ptr _pTracker;
btl::geometry::CKinFuTracker::tp_shared_ptr _pTrackerICP;
btl::geometry::CCubicGrids::tp_shared_ptr _pCubicGrids;
btl::geometry::CCubicGrids::tp_shared_ptr _pCubicGridsICP;
btl::kinect::CKeyFrame::tp_shared_ptr _pVirtualFrameWorld;
btl::kinect::CKeyFrame::tp_shared_ptr _pVirtualFrameWorldICP;
btl::kinect::CKeyFrame::tp_shared_ptr _pForDisplay;
btl::kinect::CKeyFrame::tp_shared_ptr _pKFrame;
unsigned short _nWidth, _nHeight;

bool _bContinuous = true;
bool _bCapture = false;

std::string _strPath("");
std::string _strPathName;
std::string _strFileName;
int _nN = 1;

std::vector<Eigen::Matrix4f> _veimPoses;


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
int _nStatus = 3;//1 restart; 2 //recording continue 3://pause 4://dump
bool _bDisplayImage = false;
bool _bLightOn = false;
bool _bRenderReference = false;
bool _bViewLocked = true;
std::string _strTrackingMethod("ICP");

void loadFromYml(){
#if __linux__
	cv::FileStorage cFSRead( "/space/csxsl/src/opencv-shuda/Data/kinect_intrinsics.yml", cv::FileStorage::READ );
#else if _WIN32 || _WIN64
	cv::FileStorage cFSRead ( "C:\\csxsl\\src\\opencv-shuda\\btl_rgbd\\kinect_fusion\\KinectFusion.yml", cv::FileStorage::READ );
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
	cFSRead["Tracking_Method"] >> _strTrackingMethod;
	float fCamearCenterInWorldX,fCamearCenterInWorldY,fCamearCenterInWorldZ;
	cFSRead["fCamearCenterInWorldX"] >> fCamearCenterInWorldX;
	cFSRead["fCamearCenterInWorldY"] >> fCamearCenterInWorldY;
	cFSRead["fCamearCenterInWorldZ"] >> fCamearCenterInWorldZ;
	_eivCw(0) = fCamearCenterInWorldX; _eivCw(1) = fCamearCenterInWorldY; _eivCw(2) = fCamearCenterInWorldZ;
	cFSRead.release();
}

void saveToYml(){
#if __linux__
	cv::FileStorage cFSWrite( "/space/csxsl/src/opencv-shuda/Data/KinectFusion.yml", cv::FileStorage::WRITE );
#else if _WIN32 || _WIN64
	cv::FileStorage cFSWrite ( "C:\\csxsl\\src\\opencv-shuda\\btl_rgbd\\kinect_fusion\\KinectFusion.yml", cv::FileStorage::WRITE );
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
	cFSWrite << "Tracking_Method" << _strTrackingMethod;
	float fCamearCenterInWorldX = _eivCw(0);
	float fCamearCenterInWorldY = _eivCw(1);
	float fCamearCenterInWorldZ = _eivCw(2);
	cFSWrite << "fCamearCenterInWorldX" << fCamearCenterInWorldX;
	cFSWrite << "fCamearCenterInWorldY" << fCamearCenterInWorldY;
	cFSWrite << "fCamearCenterInWorldZ" << fCamearCenterInWorldZ;

	cFSWrite.release();
}


void printVolume(){
/*
	std::string strPath("C:\\csxsl\\src\\opencv-shuda\\output\\");
	for (int i = 0; i< 256; i++ ){
		_pCubicGrids->gpuExportVolume(strPath,_nRFIdx,i,btl::geometry::CCubicGrids::_X);
		_pCubicGrids->gpuExportVolume(strPath,_nRFIdx,i,btl::geometry::CCubicGrids::_Y);
		_pCubicGrids->gpuExportVolume(strPath,_nRFIdx,i,btl::geometry::CCubicGrids::_Z);
	}
	return;*/
}
void init ( ){
	//load parameters from script
	loadFromYml();
	//initialize rendering environment
	_pGL.reset( new btl::gl_util::CGLUtil(_uResolution,_uPyrHeight,btl::utility::BTL_GL) );
	_pGL->constructVBOsPBOs();
	_pGL->_bDisplayCamera = _bDisplayImage;
	_pGL->_bEnableLighting = _bLightOn;
	_pGL->_bRenderReference = _bRenderReference;
	_pGL->init();
	_pGL->clearColorDepth();
	//setup opengl flags
	glDepthFunc  ( GL_LESS );
	glEnable     ( GL_DEPTH_TEST );
	glEnable 	 ( GL_SCISSOR_TEST );
	//glEnable     ( GL_BLEND );
	//glBlendFunc  ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	glShadeModel ( GL_SMOOTH );
	glEnable ( GL_LINE_SMOOTH );
	glEnable ( GL_POINT_SMOOTH );
	glPixelStorei ( GL_UNPACK_ALIGNMENT, 1 );
	//initialize rgbd camera
	_pKinect.reset(new btl::kinect::VideoSourceKinect(_uResolution,_uPyrHeight,_bUseNIRegistration,_eivCw));
	switch(_nMode)
	{
	case btl::kinect::VideoSourceKinect::SIMPLE_CAPTURING: //the simple capturing mode of the rgbd camera
		_pKinect->initKinect();
		break;
	case btl::kinect::VideoSourceKinect::PLAYING_BACK: //replay from files
		_pKinect->initPlayer(_oniFileName,_bRepeat);
		break;
	case btl::kinect::VideoSourceKinect::RECORDING: //record the captured sequence from the camera
		_pKinect->setDumpFileName(_oniFileName);
		_pKinect->initRecorder(_oniFileName,_nRecordingTimeInSecond);
		break;
	default://only simply capturing and playing back mode are allowed for efficiency requirements
		_nMode = btl::kinect::VideoSourceKinect::SIMPLE_CAPTURING;
		_pKinect->initKinect();
		break;
	}
	// store a frame and detect feature points for tracking.
	_pKinect->getNextFrame(btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV,&_nStatus);
	_pVirtualFrameWorld.reset(new btl::kinect::CKeyFrame(_pKinect->_pRGBCamera.get(),_uResolution,_uPyrHeight,_eivCw));	
	_pForDisplay.reset(new btl::kinect::CKeyFrame(_pKinect->_pFrame.get()));
	_pKFrame.reset(new btl::kinect::CKeyFrame(_pKinect->_pFrame.get()));
	//initialize the cubic grids
	_pCubicGrids.reset( new btl::geometry::CCubicGrids(_uCubicGridResolution,_fVolumeSize) );
	//initialize the tracker
	_pTracker.reset( new btl::geometry::CKinFuTracker(_pKinect->_pFrame.get(),_pCubicGrids));
	if (!_strTrackingMethod.compare("ICP")){
		_pTracker->setMethod(btl::geometry::CKinFuTracker::ICP);
	}
	else if(!_strTrackingMethod.compare("ORBICP")){
		_pTracker->setMethod(btl::geometry::CKinFuTracker::ORBICP);
	}
	else if(!_strTrackingMethod.compare("SURF")){
		_pTracker->setMethod(btl::geometry::CKinFuTracker::SURF);
	}
	else if(!_strTrackingMethod.compare("ORB")){
		_pTracker->setMethod(btl::geometry::CKinFuTracker::ORB);
	}
	else if(!_strTrackingMethod.compare("ORBICP")){
		_pTracker->setMethod(btl::geometry::CKinFuTracker::ORBICP);
	}
	_pTracker->init(_pKinect->_pFrame.get());
	_pTracker->setNextView(&_pGL->_eimModelViewGL);//printVolume();
	//initialize the tracker ICP
	_pVirtualFrameWorldICP.reset(new btl::kinect::CKeyFrame(_pKinect->_pRGBCamera.get(),_uResolution,_uPyrHeight,_eivCw));	
	_pCubicGridsICP.reset( new btl::geometry::CCubicGrids(_uCubicGridResolution,_fVolumeSize) );
	_pTrackerICP.reset( new btl::geometry::CKinFuTracker(_pKinect->_pFrame.get(),_pCubicGridsICP));
	_pTrackerICP->setMethod(btl::geometry::CKinFuTracker::ICP);
	_pTrackerICP->init(_pKinect->_pFrame.get());
	_pTrackerICP->setNextView(&_pGL->_eimModelViewGL);//printVolume();
	return;
}
void specialKeys( int key, int x, int y ){
	_pGL->specialKeys( key, x, y );
	switch ( key ) {
	case GLUT_KEY_F6: //display camera
		glutPostRedisplay();
		break;
	case GLUT_KEY_F3:
		glutPostRedisplay();
		break;
	}
}
void normalKeys ( unsigned char key, int x, int y ){
    switch ( key ) {
	case 27:
		saveToYml();
		exit ( 0 );
		break;
	case 'R':
		init();
		glutPostRedisplay();
		break;
    case 'r':
		/*if (_nMode == btl::kinect::VideoSourceKinect::PLAYING_BACK)	{
			_pKinect->initPlayer(_oniFileName,_bRepeat);
			_nStatus = btl::kinect::VideoSourceKinect::CONTINUE;
		}*/
        //reset
		_pKinect->getNextFrame(btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV,&_nStatus);
		_pVirtualFrameWorld.reset(new btl::kinect::CKeyFrame(_pKinect->_pRGBCamera.get(),_uResolution,_uPyrHeight,_eivCw));	
		_pVirtualFrameWorldICP.reset(new btl::kinect::CKeyFrame(_pKinect->_pRGBCamera.get(),_uResolution,_uPyrHeight,_eivCw));	
		//initialize the tracker
		_pTracker->init(_pKinect->_pFrame.get());
		_pTracker->setNextView(&_pGL->_eimModelViewGL);//printVolume();
		//
		_pTrackerICP->init(_pKinect->_pFrame.get());
		_pTrackerICP->setNextView(&_pGL->_eimModelViewGL);

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
    case 's':
		_nStatus = (_nStatus&(~btl::kinect::VideoSourceKinect::MASK_RECORDER))|btl::kinect::VideoSourceKinect::DUMP_RECORDING;
		glutPostRedisplay();
        break;
    case 'c': 
		//capture current view as a key frame
        _bCapture = !_bCapture;
		if (_bCapture){
			_nStatus = (_nStatus&(~btl::kinect::VideoSourceKinect::MASK_RECORDER))|btl::kinect::VideoSourceKinect::START_RECORDING;
		}
		else{
			_nStatus = (_nStatus&(~btl::kinect::VideoSourceKinect::MASK_RECORDER))|btl::kinect::VideoSourceKinect::STOP_RECORDING;
		}
		glutPostRedisplay();
        break;
	case 'd':
		//remove last key frame
		glutPostRedisplay();
		break;
	case '1':
		_bContinuous = !_bContinuous;
		glutPostRedisplay();
		break;
	case '2':
		//export volume debug images
		//printVolume();
		_bViewLocked = !_bViewLocked;
		glutPostRedisplay();
		break;
	case '3':
		//export RayCast debug image
		_strPath = "C:\\csxsl\\src\\opencv-shuda\\output\\" + boost::lexical_cast<std::string> ( _nN ) + ".bmp";
		_nN ++;
		break;
	case '4':
		_pVirtualFrameWorld->exportPCL(_strPathName,_strFileName);
		break;
	case '8':
		glutPostRedisplay();
		break;
	case '0':
		_pTracker->setNextView(&_pGL->_eimModelViewGL);
		//
		_pTrackerICP->setNextView(&_pGL->_eimModelViewGL);
		_pGL->setInitialPos();
		glutPostRedisplay();
		break;
    }
	_pGL->normalKeys(key,x,y);
    return;
}

void mouseClick ( int nButton_, int nState_, int nX_, int nY_ ){
    _pGL->mouseClick(nButton_,nState_,nX_,nY_);
}
void mouseMotion ( int nX_, int nY_ ){
    _pGL->mouseMotion(nX_,nY_);
}



#define TIMER
//timer
boost::posix_time::ptime _cT0, _cT1;
boost::posix_time::time_duration _cTDAll;
float _fFPS;//frame per second
void display ( void ) {
	_pGL->timerStart();

// update frame
    _pKinect->getNextFrame(btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV,&_nStatus);//the current frame must be in camera coordinate
	_pKinect->_pFrame->copyTo(&*_pKFrame);
	_pKinect->_pFrame->copyTo(&*_pForDisplay);
	//PRINTSTR("Contruct pyramid.");
	//_pGL->timerStop();

	if ( _bCapture ){
		_pTracker->track(&*_pKinect->_pFrame);
		_pTrackerICP->track(&*_pKFrame);
		//PRINTSTR("trackICP done.");
		//_pGL->timerStop();
	}//if( _bCapture )
	else{
		_nStatus = (_nStatus&(~btl::kinect::VideoSourceKinect::MASK_RECORDER))|btl::kinect::VideoSourceKinect::STOP_RECORDING;
	}

	
	
////////////////////////////////////////////////////////////////////

// render 1st viewport
    glMatrixMode ( GL_MODELVIEW );
    glViewport ( 0, _nHeight/2, _nWidth/2, _nHeight/2 ); //lower left is the origin (0,0) and x and y are pointing toward right and up.
    glScissor  ( 0, _nHeight/2, _nWidth/2, _nHeight/2 );
    // after set the intrinsics and extrinsics
    //_pGL->viewerGL();
	glLoadIdentity();
	glRotatef(180.f,1.f,0.f,0.f);
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	
	//_pKinect->_pFrame->renderCameraInWorldCVCV(_pGL.get(),_pGL->_bDisplayCamera,.05f,_pGL->_usLevel);
	//_pKinect->_pFrame->render3DPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel,0,false);
	//_pForDisplay->gpuTransformToWorldCVCV();
	_pForDisplay->gpuRenderPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel);
	
	// render objects
	{
		_pGL->renderAxisGL();
		_pGL->renderPatternGL(.1f,20,20);
		_pGL->renderPatternGL(1.f,10,10);
		_pGL->renderVoxelGL(3.f);
	}

////////////////////////////////////////////////////////////////////
// render 2nd viewport
    glViewport ( _nWidth/2, _nHeight/2, _nWidth/2, _nHeight/2 );
    glScissor  ( _nWidth/2, _nHeight/2, _nWidth/2, _nHeight/2 );

	_pKinect->_pRGBCamera->setGLProjectionMatrix(1,0.2f,30.f);

	glMatrixMode ( GL_MODELVIEW );
    glLoadIdentity();

	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

#if 1
	_pGL->gpuMapRgb2PixelBufferObj(*_pKinect->_pFrame->_acvgmShrPtrPyrRGBs[_pGL->_usLevel],_pGL->_usLevel);
#else
	_pKinect->_pRGBCamera->LoadTexture(*_pKinect->_pFrame->_acvmShrPtrPyrRGBs[_pGL->_usLevel],&_pGL->_auTexture[_pGL->_usLevel]);
#endif
	_pKinect->_pRGBCamera->renderCameraInGLLocal(_pGL->_auTexture[_pGL->_usLevel], 0.2f );
/*	
///////////////////////////////////////////////////////////////////
// render 3rd viewport
	
	// render 3nd viewport
	glViewport ( 0, 0, _nWidth/2, _nHeight/2 );
	glScissor  ( 0, 0, _nWidth/2, _nHeight/2 );
	_pKinect->_pRGBCamera->setGLProjectionMatrix(1,0.1f,100.f);
	glMatrixMode ( GL_MODELVIEW );
	//_pGL->viewerGL();
	glLoadIdentity();
	//glClearColor(0, 0, 1, 0);
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
#if USE_PBO
	_pGL->gpuMapRgb2PixelBufferObj(*_pTracker->prevFrame()->_acvgmShrPtrPyrRGBs[_pGL->_usLevel],_pGL->_usLevel);
#else
	_pKinect->_pRGBCamera->LoadTexture(*pPrevKF->_acvmShrPtrPyrRGBs[_pGL->_usLevel],&_pGL->_auTexture[_pGL->_usLevel]);
#endif
	_pKinect->_pRGBCamera->renderCameraInGLLocal(_pGL->_auTexture[_pGL->_usLevel], 0.2f );

	float aColor[4] = {0.f,1.f,0.f,1.f};
	switch(_nMode){ 
	case btl::kinect::VideoSourceKinect::RECORDING:
		_pGL->drawString("Recorder", 5, _nHeight/2-20, aColor, GLUT_BITMAP_8_BY_13);
		if ( (_nStatus&btl::kinect::VideoSourceKinect::MASK_RECORDER) == btl::kinect::VideoSourceKinect::CONTINUE_RECORDING ){
			float aColor[4] = {1.f,0.f,0.f,1.f};
			_pGL->drawString("Recording...", 5, _nHeight/2-40, aColor, GLUT_BITMAP_8_BY_13);
		}
		break;
	case btl::kinect::VideoSourceKinect::PLAYING_BACK:
		_pGL->drawString("Player", 5, _nHeight/2-20, aColor, GLUT_BITMAP_8_BY_13);
		break;
	case btl::kinect::VideoSourceKinect::SIMPLE_CAPTURING:
		_pGL->drawString("Simple", 5, _nHeight/2-20, aColor, GLUT_BITMAP_8_BY_13);
		break;
	}
	*/
////////////////////////////////////////////////////////////////////
// render 4th viewport
	glViewport ( 0, 0, _nWidth/2, _nHeight/2 );
	glScissor  ( 0, 0, _nWidth/2, _nHeight/2 );
	//glViewport ( _nWidth/2, 0, _nWidth/2, _nHeight/2 );
	//glScissor  ( _nWidth/2, 0, _nWidth/2, _nHeight/2 );
	_pKinect->_pRGBCamera->setGLProjectionMatrix(1,0.1f,100.f);
	if (_bViewLocked){
		_pTracker->setPrevView(&_pGL->_eimModelViewGL);
		_pGL->setInitialPos();
	}
	_pGL->viewerGL();
	//glClearColor(1, 0, 0, 0);
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	_pVirtualFrameWorld->assignRTfromGL();
	_pCubicGrids->gpuRaycast(&*_pVirtualFrameWorld); //get virtual frame
	//std::string strPath("C:\\csxsl\\src\\opencv-shuda\\Data\\");
	//std::string strFileName =  boost::lexical_cast<std::string> ( _nRFIdx ) + "1.yml";
	//_pPrevFrameWorld->exportYML(strPath,strFileName);
	//_pPrevFrameWorld->render3DPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel,0,false);
	_pVirtualFrameWorld->gpuRenderPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel);
	//_pPrevFrameWorld->setView(&_pGL->_eimModelViewGL);
	{
		_pGL->renderAxisGL();
		_pGL->renderPatternGL(.1f,20,20);
		_pGL->renderPatternGL(1.f,10,10);
		_pGL->renderVoxelGL(_fVolumeSize);
		//_pGL->renderOctTree(0.f,0.f,0.f,3.f,1); this is very slow when the level of octree is deep.
	}
	float aColor[4] = {0.f,1.f,0.f,1.f};
	_pGL->drawString("Proposed Approach", 5, _nHeight/2-20, aColor, GLUT_BITMAP_8_BY_13);

	if (!_strTrackingMethod.compare("ICP")){
		float aColor[4] = {0.f,1.f,0.f,1.f};
		_pGL->drawString("ICP", 5, 10, aColor, GLUT_BITMAP_8_BY_13);
	}
	else if(!_strTrackingMethod.compare("ORBICP")){
		float aColor[4] = {0.f,1.f,0.f,1.f};
		_pGL->drawString("ORBICP", 5, 10, aColor, GLUT_BITMAP_8_BY_13);
	}
	else if(!_strTrackingMethod.compare("SURF")){
		float aColor[4] = {0.f,1.f,0.f,1.f};
		_pGL->drawString("SURF", 5, 10, aColor, GLUT_BITMAP_8_BY_13);
	}
	else if(!_strTrackingMethod.compare("ORB")){
		float aColor[4] = {0.f,1.f,0.f,1.f};
		_pGL->drawString("ORB", 5, 10, aColor, GLUT_BITMAP_8_BY_13);
	}
	else if(!_strTrackingMethod.compare("ORBICP")){
		float aColor[4] = {0.f,1.f,0.f,1.f};
		_pGL->drawString("ORBICP", 5, 10, aColor, GLUT_BITMAP_8_BY_13);
	}
////////////////////////////////////////////////////////////////////
// render 5th viewport
	//glViewport ( _nWidth/2, 0, _nWidth/2, _nHeight );
	//glScissor  ( _nWidth/2, 0, _nWidth/2, _nHeight );
	glViewport ( _nWidth/2, 0, _nWidth/2, _nHeight/2 );
	glScissor  ( _nWidth/2, 0, _nWidth/2, _nHeight/2 );
	_pKinect->_pRGBCamera->setGLProjectionMatrix(1,0.1f,100.f);
	if (_bViewLocked){
		_pTrackerICP->setPrevView(&_pGL->_eimModelViewGL);
		_pGL->setInitialPos();
	}
	_pGL->viewerGL();
	//glClearColor(1, 0, 0, 0);
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	_pVirtualFrameWorldICP->assignRTfromGL();
	_pCubicGridsICP->gpuRaycast(&*_pVirtualFrameWorldICP); //get virtual frame
	//std::string strPath("C:\\csxsl\\src\\opencv-shuda\\Data\\");
	//std::string strFileName =  boost::lexical_cast<std::string> ( _nRFIdx ) + "1.yml";
	//_pPrevFrameWorld->exportYML(strPath,strFileName);
	//_pPrevFrameWorld->render3DPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel,0,false);
	_pVirtualFrameWorldICP->gpuRenderPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel);
	//_pPrevFrameWorld->setView(&_pGL->_eimModelViewGL);
	{
		_pGL->renderAxisGL();
		_pGL->renderPatternGL(.1f,20,20);
		_pGL->renderPatternGL(1.f,10,10);
		_pGL->renderVoxelGL(_fVolumeSize);
		//_pGL->renderOctTree(0.f,0.f,0.f,3.f,1); this is very slow when the level of octree is deep.
	}
	_pGL->drawString("KinectFusion", 5, _nHeight/2-20, aColor, GLUT_BITMAP_8_BY_13);
	_pGL->drawString("ICP", 5, 10, aColor, GLUT_BITMAP_8_BY_13);

/*
	stringstream ss;
	ss << "FBO: ";
	ss << "Render-To-Texture Time: " << renderToTextureTime << " ms" << ends;
	ss << std::resetiosflags(std::ios_base::fixed | std::ios_base::floatfield);
*/
	
	glutSwapBuffers();
    if ( _bContinuous ) {
        glutPostRedisplay();
		//_bContinuous = false;
    }
}

void reshape ( int nWidth_, int nHeight_ ) {
	_nHeight = nHeight_;
	_nWidth = nWidth_;
    return;
}

int main ( int argc, char** argv ) {
    try {
        glutInit ( &argc, argv );
        glutInitDisplayMode ( GLUT_DOUBLE | GLUT_RGB );
        glutInitWindowSize (1000,750 );//1280, 480 ); //480
        glutCreateWindow ( "CameraPose" );
		GLenum eError = glewInit();
		if (GLEW_OK != eError){
			PRINTSTR("glewInit() error.");
			PRINT( glewGetErrorString(eError) );
		}
        glutKeyboardFunc ( normalKeys );
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
	catch ( btl::utility::CError& e )	{
		if ( std::string const* mi = boost::get_error_info< btl::utility::CErrorInfo > ( e ) )	{
			std::cerr << "Error Info: " << *mi << std::endl;
		}
	}
	catch ( std::runtime_error& e )	{
		PRINTSTR( e.what() );
	}

    return 0;
}


