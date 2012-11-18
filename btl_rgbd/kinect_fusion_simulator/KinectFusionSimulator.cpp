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
#include "VideoSourceKinect.hpp"
#include "VideoSourceKinectSimulator.hpp"
#include "CubicGrids.h"
#define _nReserved 5

btl::kinect::VideoSourceKinectSimulator::tp_shared_ptr _pKinectSimulator;
btl::gl_util::CGLUtil::tp_shared_ptr _pGL;
btl::geometry::CCubicGrids::tp_shared_ptr _pCubicGrids;
btl::kinect::CKeyFrame::tp_shared_ptr _pPrevFrameWorld, _pVirtualFrame2;
unsigned short _nWidth, _nHeight;

btl::kinect::CKeyFrame::tp_shared_ptr _aShrPtrKFs[_nReserved];

std::vector< btl::kinect::CKeyFrame::tp_shared_ptr* > _vShrPtrsKF;
int _nRFIdx = 0; //reference frame counter

bool _bContinuous = true;
bool _bCapture = false;
bool _bCapture2 = false;
int _nView = 0;
ushort _usViewNO = 0;
ushort _uResolution = 0;
ushort _uPyrHeight = 3;
std::string _strPath("");
std::string _strPathName;
std::string _strFileName;
int _nN = 1;

void printVolume(){
	std::string strPath("C:\\csxsl\\src\\opencv-shuda\\output\\");
	for (int i = 0; i< 256; i++ ){
		_pCubicGrids->gpuExportVolume(strPath,_nRFIdx,i,btl::geometry::CCubicGrids::_X);
		_pCubicGrids->gpuExportVolume(strPath,_nRFIdx,i,btl::geometry::CCubicGrids::_Y);
		_pCubicGrids->gpuExportVolume(strPath,_nRFIdx,i,btl::geometry::CCubicGrids::_Z);
	}
	return;
}
void init ( ){
	for(int i=0; i <_nReserved; i++){ 
		_aShrPtrKFs[i].reset(new btl::kinect::CKeyFrame(_pKinectSimulator->_pRGBCamera.get(),_uResolution,_uPyrHeight,1.5f,1.5f,-0.3f));	
	}
	_pPrevFrameWorld.reset(new btl::kinect::CKeyFrame(_pKinectSimulator->_pRGBCamera.get(),_uResolution,_uPyrHeight,1.5f,1.5f,-0.3f));	
	_pVirtualFrame2.reset(new btl::kinect::CKeyFrame(_pKinectSimulator->_pRGBCamera.get(),_uResolution,_uPyrHeight,1.5f,1.5f,-0.3f));	
	_pGL->clearColorDepth();
	glDepthFunc  ( GL_LESS );
	glEnable     ( GL_DEPTH_TEST );
	glEnable 	 ( GL_SCISSOR_TEST );
	glEnable     ( GL_BLEND );
	glBlendFunc  ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	glShadeModel ( GL_FLAT );
	glEnable ( GL_LINE_SMOOTH );
	glEnable ( GL_POINT_SMOOTH );

	glPixelStorei ( GL_UNPACK_ALIGNMENT, 1 );

	_pGL->init();

	_pKinectSimulator->_pFrame->setView(&_pGL->_eimModelViewGL);
	_pGL->setInitialPos();
	// store a frame and detect feature points for tracking.
	btl::kinect::CKeyFrame::tp_shared_ptr& p1stKF = _aShrPtrKFs[_nRFIdx];
	//_pKinectSimulator->getNextFrame(btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV);
	//_pKinectSimulator->_pFrame->copyTo(&*p1stKF);
	//p1stKF->extractSurfFeatures();
	//p1stKF->extractOrbFeatures();
	//p1stKF->gpuTransformToWorldCVCV();

	//std::string strPath("C:\\csxsl\\src\\opencv-shuda\\output\\");
	//std::string strFileName =  boost::lexical_cast<std::string> ( _nRFIdx ) + ".yml";
	//p1stKF->exportYML(strPath,strFileName);
	//p1stKF->importYML(strPath,strFileName);
	//_pCubicGrids->gpuIntegrateFrameIntoVolumeCVCV(*p1stKF);
	//printVolume();
	// assign the rgb and depth to the current frame.
	//p1stKF->setView(&_pGL->_eimModelViewGL);
	//_vShrPtrsKF.push_back( &p1stKF );
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
    case 'r':
        //reset
		_nRFIdx = 0;
		_vShrPtrsKF.clear();
		_pCubicGrids->reset();
        init();
        glutPostRedisplay();
        break;
    case 'n':
        //next step
		PRINTSTR("CubicGrid import started.")
		_pCubicGrids->importYML(std::string("volume1.yml"));
		PRINTSTR("CubicGrid import done.")
        glutPostRedisplay();
        break;
    case 's':
		//save volume
		_pCubicGrids->exportYML(std::string(""),1);
        //single step
        //_bContinuous = !_bContinuous;
        break;
    case 'c': 
		//capture current view as a key frame
        _bCapture = !_bCapture;
        break;
	case 'w':
		_bCapture2 = !_bCapture2;
		break;
	case 'd':
		//remove last key frame
		if(_nRFIdx >0 ) {
			_nRFIdx--;
			_vShrPtrsKF.pop_back();
		}
		glutPostRedisplay();
		break;
	case '1':
		glutPostRedisplay();
		break;
	case '2':
		//export volume debug images
		printVolume();
		glutPostRedisplay();
		break;
	case '3':
		//export RayCast debug image
		_strPath = "C:\\csxsl\\src\\opencv-shuda\\output\\" + boost::lexical_cast<std::string> ( _nN ) + ".bmp";
		_nN ++;
		break;
	case '4':
		_pPrevFrameWorld->exportPCL(_strPathName,_strFileName);
		break;
	case '8':
			glutPostRedisplay();
			break;
	case '0':
		//_usViewNO = ++_usViewNO % _vShrPtrsKF.size(); 
		//(*_vShrPtrsKF[ _usViewNO ])->setView(&_pGL->_eimModelViewGL);
		_pKinectSimulator->_pFrame->setView(&_pGL->_eimModelViewGL);
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
	
////////////////////////////////////////////////////////////////////
// render 1st viewport
    glViewport ( 0, _nHeight/2, _nWidth/2, _nHeight/2 ); //lower left is the origin (0,0) and x and y are pointing toward right and up.
    glScissor  ( 0, _nHeight/2, _nWidth/2, _nHeight/2 );

	_pKinectSimulator->setSensorDepthRange();
	_pGL->viewerGL();
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	// render objects
	_pGL->renderTeapot();
	//_pGL->renderTestPlane();
	_pKinectSimulator->getNextFrame(btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV);//the current frame must be in camera coordinate
	_pKinectSimulator->_pFrame->copyTo(&*_pPrevFrameWorld);
/*
	if (_bCapture){
		_pPrevFrameWorld->exportPCL(std::string(""),std::string(""));
		_pKinectSimulator->exportRawDepth();
	}
	PRINT(_pPrevFrameWorld->_eimRw);
	PRINT(_pPrevFrameWorld->_eivTw);
	PRINT(-_pPrevFrameWorld->_eimRw.transpose()*_pPrevFrameWorld->_eivTw);
*/
	_pPrevFrameWorld->assignRTfromGL();
	_pPrevFrameWorld->gpuTransformToWorldCVCV();
	if (_bCapture)
	{
		_pCubicGrids->gpuIntegrateFrameIntoVolumeCVCV(*_pPrevFrameWorld);
	}
	if(_pGL->_bRenderReference) {
		_pGL->renderAxisGL();
		_pGL->renderPatternGL(.1f,20 ,20 );
		_pGL->renderPatternGL(1.f,10 ,10 );
		_pGL->renderVoxelGL(3.f);
		//_pGL->renderOctTree(0.f,0.f,0.f,3.f,1); this is very slow when the level of octree is deep.
	}
  
////////////////////////////////////////////////////////////////////
// render 2nd viewport
    glViewport ( _nWidth/2, _nHeight/2, _nWidth/2, _nHeight/2 );
    glScissor  ( _nWidth/2, _nHeight/2, _nWidth/2, _nHeight/2 );
	
	//_pKinectSimulator->_pRGBCamera->setGLProjectionMatrix(1,0.1f,100.f);
	_pKinectSimulator->setSensorDepthRange();
	_pGL->viewerGL();
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	
	//_pPrevFrameWorld->gpuRenderPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel);
	_pPrevFrameWorld->render3DPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel,0,false);
	//_pPrevFrameWorld->exportPCL(std::string(""),std::string(""));
	// render objects
	if(_pGL->_bRenderReference) {
		_pGL->renderAxisGL();
		_pGL->renderPatternGL(.1f,20 ,20 );
		_pGL->renderPatternGL(1.f,10 ,10 );
		_pGL->renderVoxelGL(3.f);
		//_pGL->renderOctTree(0.f,0.f,0.f,3.f,1); this is very slow when the level of octree is deep.
	}

///////////////////////////////////////////////////////////////////
// render 3rd viewport

	// render 3nd viewport
	glViewport ( 0, 0, _nWidth/2, _nHeight/2 );
	glScissor  ( 0, 0, _nWidth/2, _nHeight/2 );

	// after set the intrinsics and extrinsics
	//_pKinectSimulator->_pRGBCamera->setGLProjectionMatrix(1,0.1f,100.f);
	_pKinectSimulator->setSensorDepthRange();
	_pGL->viewerGL();
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	_pGL->renderTeapot();
	//_pGL->renderTestPlane();
	//_pKinectSimulator->_pFrame->render3DPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel,0,false);

	// render objects
	if(_pGL->_bRenderReference) {
		_pGL->renderAxisGL();
		_pGL->renderPatternGL(.1f,20 ,20 );
		_pGL->renderPatternGL(1.f,10 ,10 );
		_pGL->renderVoxelGL(3.f);
		//_pGL->renderOctTree(0.f,0.f,0.f,3.f,1); this is very slow when the level of octree is deep.
	}
	
////////////////////////////////////////////////////////////////////
// render 4th viewport
	glViewport ( _nWidth/2, 0, _nWidth/2, _nHeight/2 );
	glScissor  ( _nWidth/2, 0, _nWidth/2, _nHeight/2 );

	//_pKinectSimulator->_pRGBCamera->setGLProjectionMatrix(1,0.1f,100.f);
	_pKinectSimulator->setSensorDepthRange();
	_pGL->viewerGL();
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	_pVirtualFrame2->assignRTfromGL();
	_pCubicGrids->gpuRaycast( &*_pVirtualFrame2 );
	_pCubicGrids->gpuGetOccupiedVoxels();
	_pVirtualFrame2->render3DPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel,0,false);
	if(_pGL->_bRenderReference) {
		_pGL->renderAxisGL();
		_pGL->renderPatternGL(.1f,20 ,20 );
		_pGL->renderPatternGL(1.f,10 ,10 );
		_pGL->renderVoxelGL(3.f);
		//_pGL->renderOctTree(0.f,0.f,0.f,3.f,1); this is very slow when the level of octree is deep.
	}
	if(_bCapture2 ){
		_pKinectSimulator->captureScreen();
		_bCapture2  = false;
	}
/*

	//_pGL->setOrthogonal();
	_pKinectSimulator->_pRGBCamera->setGLProjectionMatrix(1,0.1f,100.f);
	// switch to modelview matrix in order to set scene
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
/ *
	glRasterPos2i(0, 0);
	glDrawPixels(_nWidth/2, _nHeight/2,  GL_LUMINANCE, GL_UNSIGNED_BYTE, _pKinectSimulator->_pFrame->_acvmShrPtrPyrBWs[0]->data);* /
#if USE_PBO
	_pGL->gpuMapRgb2PixelBufferObj(*_pKinectSimulator->_pFrame->_acvgmShrPtrPyrRGBs[_pGL->_usLevel],_pGL->_usLevel);
#else
	_pKinectSimulator->_pRGBCamera->LoadTexture(*_pKinectSimulator->_pFrame->_acvmShrPtrPyrRGBs[_pGL->_usLevel],&_pGL->_auTexture[_pGL->_usLevel]);
#endif
	_pKinectSimulator->_pRGBCamera->renderCameraInGLLocal(_pGL->_auTexture[_pGL->_usLevel], *_pKinectSimulator->_pFrame->_acvmShrPtrPyrRGBs[_pGL->_usLevel],0.2 );

*/

	/*_pGL->setOrthogonal();

	// switch to modelview matrix in order to set scene
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glRasterPos2i(0, 0);
	glDrawPixels(_nWidth/2, _nHeight/2,  GL_BGR, GL_UNSIGNED_BYTE, _pKinectSimulator->_pFrame->_acvmShrPtrPyrRGBs[0]->data);*/

	
	glutSwapBuffers();
    if ( _bContinuous ) {
        glutPostRedisplay();
    }
}

void reshape ( int nWidth_, int nHeight_ ) {
    //cout << "reshape() " << endl;
    //_pKinectSimulator->_pRGBCamera->setGLProjectionMatrix ( 1, 0.01f, 10.f );

    // setup blending
    //glBlendFunc ( GL_SRC_ALPHA, GL_ONE );			// Set The Blending Function For Translucency

    //unsigned short nTemp = nWidth_ / 8; //make sure that _nWidth is divisible to 4
    _nWidth = nWidth_;
    _nHeight = nHeight_; //3
    glutReshapeWindow ( int ( _nWidth ), int ( _nHeight ) );
    return;
}

int main ( int argc, char** argv ) {
    try {
        glutInit ( &argc, argv );
        glutInitDisplayMode ( GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH );
        glutInitWindowSize ( 1280, 960 ); //480
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

		_pGL.reset( new btl::gl_util::CGLUtil(_uResolution,_uPyrHeight,btl::utility::BTL_CV,Eigen::Vector3f(1.5,1.5,1.5)) );
		_pGL->setCudaDeviceForGLInteroperation();
		//_pKinectSimulator.reset(new btl::kinect::VideoSourceKinect(_uResolution,_uPyrHeight,true,1.5,1.5,-0.3));
		//_pKinectSimulator->initKinect();

		_pKinectSimulator.reset(new btl::kinect::VideoSourceKinectSimulator(_uResolution,_uPyrHeight,true,1.5f,1.5f,-0.3f));
		_pCubicGrids.reset( new btl::geometry::CCubicGrids(64,3) );
		init();
		_pGL->constructVBOsPBOs();
		//_pCubicGrids->gpuCreateVBO(_pGL.get());
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


