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
btl::geometry::CCubicGrids::tp_shared_ptr _pCubicGrids;
btl::kinect::CKeyFrame::tp_shared_ptr _pVirtualFrameWorld;
unsigned short _nWidth, _nHeight;

bool _bContinuous = true;
bool _bCapture = false;
ushort _uResolution = 0;
ushort _uPyrHeight = 3;

std::string _strPath("");
std::string _strPathName;
std::string _strFileName;
int _nN = 1;

std::vector<Eigen::Matrix4f> _veimPoses;

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
	
	_pVirtualFrameWorld.reset(new btl::kinect::CKeyFrame(_pKinect->_pRGBCamera.get(),_uResolution,_uPyrHeight,1.5f,1.5f,-0.3f));	

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

	// store a frame and detect feature points for tracking.
	_pKinect->getNextFrame(btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV);
	_pTracker->init(_pKinect->_pFrame.get());

	//printVolume();
	_pTracker->setNextView(&_pGL->_eimModelViewGL);
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
		_pKinect->initPlayer(std::string("20121121-153156.oni"),false);
		_pCubicGrids->reset();
        init();
        glutPostRedisplay();
        break;
    case 'n':
        //next step
        glutPostRedisplay();
        break;
    case 's':
        //single step
		_pKinect->record();
        //_bContinuous = !_bContinuous;
        break;
    case 'c': 
		//capture current view as a key frame
        _bCapture = !_bCapture;
        break;
	case 'd':
		//remove last key frame
		
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
		_pVirtualFrameWorld->exportPCL(_strPathName,_strFileName);
		break;
	case '8':
		glutPostRedisplay();
		break;
	case '0':
		_pTracker->setNextView(&_pGL->_eimModelViewGL);
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
    _pKinect->getNextFrame(btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV);//the current frame must be in camera coordinate
	PRINTSTR("Contruct pyramid.");
	_pGL->timerStop();

// ( second frame )
	if ( _bCapture && !_pKinect->isPlayStop() ){
		_pTracker->track(&*_pKinect->_pFrame);
		PRINTSTR("trackICP done.");
		_pGL->timerStop();
	}//if( _bCapture )
	
////////////////////////////////////////////////////////////////////
// render 1st viewport
   /* glMatrixMode ( GL_MODELVIEW );
    glViewport ( 0, _nHeight/2, _nWidth/2, _nHeight/2 ); //lower left is the origin (0,0) and x and y are pointing toward right and up.
    glScissor  ( 0, _nHeight/2, _nWidth/2, _nHeight/2 );
    // after set the intrinsics and extrinsics
    _pGL->viewerGL();
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	
	//_pKinect->_pFrame->renderCameraInWorldCVCV(_pGL.get(),_pGL->_bDisplayCamera,.05f,_pGL->_usLevel);
	_pKinect->_pFrame->render3DPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel,0,false);
	
	// render objects
	ushort usViewIdxTmp = 0;
	for( std::vector< btl::kinect::CKeyFrame::tp_shared_ptr* >::iterator cit = _vShrPtrsKF.begin(); cit!= _vShrPtrsKF.end(); cit++,usViewIdxTmp++ ) {
		if (usViewIdxTmp == _usViewNO)
			(**cit)->renderCameraInWorldCVCV(_pGL.get(),_pGL->_bDisplayCamera,.1f,_pGL->_usLevel);
		else
			(**cit)->renderCameraInWorldCVCV(_pGL.get(),false,.05f,_pGL->_usLevel);
		(**cit)->render3DPtsInWorldCVCV(_pGL.get(), _pGL->_usLevel, _usColorIdx, false );
	}
	//_pKinect->_pFrame->renderCameraInWorldCVGL2( _pGL.get(), _pGL->_bDisplayCamera, true, .1f,_pGL->_usPyrLevel );

	if(_pGL->_bRenderReference) {
		_pGL->renderAxisGL();
		_pGL->renderPatternGL(.1f,20.f,20.f);
		_pGL->renderPatternGL(1.f,10.f,10.f);
		_pGL->renderVoxelGL(3.f);
		//_pGL->renderOctTree(0.f,0.f,0.f,3.f,1); this is very slow when the level of octree is deep.
	}
	*/
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
	_pKinect->_pRGBCamera->renderCameraInGLLocal(_pGL->_auTexture[_pGL->_usLevel], *_pKinect->_pFrame->_acvmShrPtrPyrRGBs[_pGL->_usLevel],0.2f );
	
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
	_pKinect->_pRGBCamera->renderCameraInGLLocal(_pGL->_auTexture[_pGL->_usLevel], *_pTracker->prevFrame()->_acvmShrPtrPyrRGBs[_pGL->_usLevel],0.2f );

/*
	glViewport ( 0, 0, _nWidth/2, _nHeight/2 );
	glScissor  ( 0, 0, _nWidth/2, _nHeight/2 );
	_pGL->viewerGL();
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	_pCubicGrids->gpuRenderVoxelInWorldCVGL();*/

////////////////////////////////////////////////////////////////////
// render 4th viewport
	glViewport ( _nWidth/2, 0, _nWidth/2, _nHeight/2 );
	glScissor  ( _nWidth/2, 0, _nWidth/2, _nHeight/2 );
	_pKinect->_pRGBCamera->setGLProjectionMatrix(1,0.1f,100.f);
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
	if(_pGL->_bRenderReference) {
		_pGL->renderAxisGL();
		_pGL->renderPatternGL(.1f,20.f,20.f);
		_pGL->renderPatternGL(1.f,10.f,10.f);
		_pGL->renderVoxelGL(3.f);
		//_pGL->renderOctTree(0.f,0.f,0.f,3.f,1); this is very slow when the level of octree is deep.
	}
	float aColor[4] = {1.f,0.f,0.f,1.f};

	_pGL->drawString("Test", 1, 1, aColor, GLUT_BITMAP_8_BY_13);
/*
	stringstream ss;
	ss << "FBO: ";
	if(fboUsed)
		ss << "on" << ends;
	else
		ss << "off" << ends;

	drawString(ss.str().c_str(), 1, screenHeight-TEXT_HEIGHT, color, font);
	ss.str(""); // clear buffer

	ss << std::fixed << std::setprecision(3);
	ss << "Render-To-Texture Time: " << renderToTextureTime << " ms" << ends;
	drawString(ss.str().c_str(), 1, screenHeight-(2*TEXT_HEIGHT), color, font);
	ss.str("");
	ss << std::resetiosflags(std::ios_base::fixed | std::ios_base::floatfield);
*/
	
	glutSwapBuffers();
    if ( _bContinuous ) {
        glutPostRedisplay();
    }
}

void reshape ( int nWidth_, int nHeight_ ) {
    //cout << "reshape() " << endl;
    //_pKinect->_pRGBCamera->setGLProjectionMatrix ( 1, 0.01, 100 );

    //// setup blending
    // glBlendFunc ( GL_SRC_ALPHA, GL_ONE );			// Set The Blending Function For Translucency
    //glColor4f ( 1.0f, 1.0f, 1.0f, 0.5 );

    unsigned short nTemp = nWidth_ / 8; //make sure that _nWidth is divisible to 4
    _nWidth = nTemp * 8;
    _nHeight = nTemp * 6; //3
    glutReshapeWindow ( int ( _nWidth ), int ( _nHeight ) );
    return;
}

int main ( int argc, char** argv ) {
    try {
        glutInit ( &argc, argv );
        glutInitDisplayMode ( GLUT_DOUBLE | GLUT_RGB );
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

		_pGL.reset( new btl::gl_util::CGLUtil(_uResolution,_uPyrHeight,btl::utility::BTL_CV) );
		_pGL->initCuda();
		_pGL->setCudaDeviceForGLInteroperation();
		_pKinect.reset(new btl::kinect::VideoSourceKinect(_uResolution,_uPyrHeight,true,1.5f,1.5f,-0.3f));
		//_pKinect->initKinect();
		//_pKinect->initRecorder(std::string("."),30);
		_pKinect->initPlayer(std::string("20121121-153156.oni"),false);
		_pCubicGrids.reset( new btl::geometry::CCubicGrids(512,3) );
		_pTracker.reset( new btl::geometry::CKinFuTracker(_pKinect->_pFrame.get(),_pCubicGrids));
		_pTracker->setMethod(btl::geometry::CKinFuTracker::ICP);
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


