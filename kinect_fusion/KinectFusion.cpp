//display kinect depth in real-time
#define INFO

#include <iostream>
#include <string>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <Converters.hpp>
#include <opencv2/gpu/gpumat.hpp>
#include <utility>
#include <boost/lexical_cast.hpp>
#include <gl/freeglut.h>
#include <XnCppWrapper.h>
#include "GLUtil.h"
#include "EigenUtil.hpp"
#include "Camera.h"
#include "GLUtil.h"
#include "Histogram.h"
#include "KeyFrame.h"
#include <VideoSourceKinect.hpp>

#define _nReserved 60

btl::kinect::VideoSourceKinect::tp_shared_ptr _pKinect;
btl::gl_util::CGLUtil::tp_shared_ptr _pGL;

unsigned short _nWidth, _nHeight;

btl::kinect::CKeyFrame::tp_shared_ptr _aShrPtrKFs[_nReserved];
std::vector< btl::kinect::CKeyFrame::tp_shared_ptr* > _vShrPtrsKF;
std::vector< int > _vRFIdx;
int _nKFCounter = 1; //key frame counter
int _nRFIdx = 0; //reference frame counter

bool _bContinuous = true;
bool _bPrevStatus = true;
bool _bRenderReference = true;
bool _bCapture = false;
bool _bRenderPlane = false;
int _nN = 1;
int _nView = 0;
void init();
void specialKeys( int key, int x, int y ){
	_pGL->specialKeys( key, x, y );
}
void normalKeys ( unsigned char key, int x, int y ){
    switch ( key ) {
    case 'r':
        //reset
		_nKFCounter=1;
		_nRFIdx=0;
		_vShrPtrsKF.clear();
        init();
        glutPostRedisplay();
        break;
    case 'n':
        //next step
        glutPostRedisplay();
        break;
    case 's':
        //single step
        _bContinuous = !_bContinuous;
        break;
    case 'c': 
		//capture current view as a key frame
        _bCapture = !_bCapture;
		
        break;
	case 'd':
		//remove last key frame
		if(_nKFCounter >0 ) {
			if( _nRFIdx ==  _nKFCounter ) _nRFIdx--;
			_nKFCounter--;
			_vShrPtrsKF.pop_back();
		}
		glutPostRedisplay();
		break;
	case 'v':
		//use current keyframe as a reference
		if( _nRFIdx <_nKFCounter )
		{
			_nRFIdx = _nKFCounter-1;
			_vRFIdx.push_back( _nRFIdx );
			_aShrPtrKFs[_nRFIdx]->_bIsReferenceFrame = true;
			glutPostRedisplay();
		}
		break;
	case 'p':
		//use current keyframe as a reference
		_bRenderPlane =! _bRenderPlane;
		for(unsigned int i=0; i < _nKFCounter; i++)	{
			_aShrPtrKFs[i]->_bRenderPlaneSeparately=_bRenderPlane;
			//if(_bRenderPlane) {_aShrPtrKFs[i]->gpuDetectPlane(3);}
		}
		glutPostRedisplay();
		break;
	case '0':
		(*_vShrPtrsKF[ _nView ])->setView(&_pGL->_eimModelViewGL);
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

void init ( ){
	for(int i=0; i <_nReserved; i++){ 
		_aShrPtrKFs[i].reset(new btl::kinect::CKeyFrame(_pKinect->_pRGBCamera.get()));	
		_aShrPtrKFs[i]->_pGL = _pGL.get();
	}
    
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
    _pKinect->getNextPyramid(4,btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV);
	btl::kinect::CKeyFrame::tp_shared_ptr& p1stKF = _aShrPtrKFs[0];
	_vRFIdx.push_back(0);
    // assign the rgb and depth to the current frame.
	_pKinect->_pFrame->copyTo(&*p1stKF);
	p1stKF->_bIsReferenceFrame = true;
	p1stKF->setView(&_pGL->_eimModelViewGL);
	p1stKF->gpuDetectPlane(2);
	_vShrPtrsKF.push_back( &p1stKF );
    return;
}
#define TIMER
//timer
boost::posix_time::ptime _cT0, _cT1;
boost::posix_time::time_duration _cTDAll;
float _fFPS;//frame per second
void display ( void ) {
// update frame
    _pKinect->getNextPyramid(4,btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV);
// ( second frame )
	//_pGL->timerStart();
	unsigned short uInliers;
    if ( false && _nKFCounter < _nReserved ) {
		// assign the rgb and depth to the current frame.
		btl::kinect::CKeyFrame::tp_shared_ptr& pCurrentKF = _aShrPtrKFs[_nKFCounter];
		_pKinect->_pFrame->copyTo(&*pCurrentKF);
		btl::kinect::CKeyFrame::tp_shared_ptr& pReferenceKF = _aShrPtrKFs[_nRFIdx];
        // track camera motion
		pCurrentKF->detectConnectionFromCurrentToReference(*pReferenceKF,0);
        pCurrentKF->calcRT ( *pReferenceKF,0,&uInliers );
 		pCurrentKF->applyRelativePose( *pReferenceKF );
		//detect planes
		//pCurrentKF->detectPlane(_pGL->_uLevel);
		_vShrPtrsKF.push_back( &pCurrentKF );
		_nKFCounter++;
		std::cout << "new key frame added" << std::flush;

		//use current keyframe as a reference
		if( _nRFIdx <_nKFCounter && _nKFCounter < _nReserved )
		{
			_nRFIdx = _nKFCounter-1;
			_vRFIdx.push_back( _nRFIdx );
			_aShrPtrKFs[_nRFIdx]->_bIsReferenceFrame = true;
		}
		_bCapture = false;
    }
	else if( _nKFCounter > 49 )	{
		std::cout << "two many key frames to hold" << std::flush;  
	}
	else if (_bCapture && _nKFCounter < _nReserved){
		// assign the rgb and depth to the current frame.
		btl::kinect::CKeyFrame::tp_shared_ptr& pReferenceKF = _aShrPtrKFs[_nRFIdx];
		// track camera motion
		_pKinect->_pFrame->detectConnectionFromCurrentToReference(*pReferenceKF,0);
		double dE = _pKinect->_pFrame->calcRT ( *pReferenceKF,0, &uInliers);
		Eigen::AngleAxis<double> eiAA(_pKinect->_pFrame->_eimR);
		double dAngle = eiAA.angle();
		double dNorm = _pKinect->_pFrame->_eivT.norm(); 
		_pKinect->_pFrame->applyRelativePose( *pReferenceKF );
		if( dE < 0.05 && uInliers> 40 && ( dNorm > 0.05 || dAngle > M_PI_4/4.) ){
			PRINT(dAngle);
			PRINT(dNorm)
			btl::kinect::CKeyFrame::tp_shared_ptr& pCurrentKF = _aShrPtrKFs[_nKFCounter];
			_pKinect->_pFrame->copyTo(&*pCurrentKF);
			_vShrPtrsKF.push_back( &pCurrentKF );
			_nKFCounter++;
			//use current keyframe as a reference
			if( _nRFIdx <_nKFCounter ){
				_nRFIdx = _nKFCounter-1;
				_vRFIdx.push_back( _nRFIdx );
				_aShrPtrKFs[_nRFIdx]->_bIsReferenceFrame = true;
			}
			pCurrentKF->gpuDetectPlane(2);
			std::cout << "new key frame added" << std::flush;
		}
	}
	//_pGL->timerStop();
// render first viewport
    glMatrixMode ( GL_MODELVIEW );
    glViewport ( 0, 0, _nWidth / 2, _nHeight );
    glScissor  ( 0, 0, _nWidth / 2, _nHeight );
    // after set the intrinsics and extrinsics
    //glLoadIdentity();
	_pGL->viewerGL();

    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	    // render objects
	for( std::vector< btl::kinect::CKeyFrame::tp_shared_ptr* >::iterator cit = _vShrPtrsKF.begin(); cit!= _vShrPtrsKF.end(); cit++ ) {
		(**cit)->renderCameraInGLWorld( _pGL->_bDisplayCamera,true,true, .05f,_pGL->_uLevel );
	}
	_pKinect->_pFrame->renderCameraInGLWorld( false, true, false, .1f,_pGL->_uLevel );

	if(_pGL->_bRenderReference) {
		_pGL->renderAxisGL();
		_pGL->renderPatternGL(.1f,20.f,20.f);
		_pGL->renderPatternGL(1.f,10.f,10.f);
		//_pGL->renderVoxelGL(2.f);
		//_pGL->renderOctTree(0.f,0.f,0.f,3.f,1); this is very slow when the level of octree is deep.
	}

// render second viewport
    glViewport ( _nWidth/2, 0, _nWidth/2, _nHeight );
    glScissor  ( _nWidth/2, 0, _nWidth/2, _nHeight );
    glLoadIdentity();
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	_pKinect->_pRGBCamera->LoadTexture(*_pKinect->_pFrame->_acvmShrPtrPyrRGBs[_pGL->_uLevel],&_pKinect->_pFrame->_uTexture);
    _pKinect->_pRGBCamera->renderCameraInGLLocal(_pKinect->_pFrame->_uTexture, *_pKinect->_pFrame->_acvmShrPtrPyrRGBs[_pGL->_uLevel], .2f );

    glutSwapBuffers();

    if ( _bContinuous ) {
        glutPostRedisplay();
    }
}

void reshape ( int nWidth_, int nHeight_ ) {
    //cout << "reshape() " << endl;
    _pKinect->_pRGBCamera->setIntrinsics ( 1, 0.01, 100 );

    // setup blending
    glBlendFunc ( GL_SRC_ALPHA, GL_ONE );			// Set The Blending Function For Translucency
    glColor4f ( 1.0f, 1.0f, 1.0f, 0.5 );

    unsigned short nTemp = nWidth_ / 8; //make sure that _nWidth is divisible to 4
    _nWidth = nTemp * 8;
    _nHeight = nTemp * 3;
    glutReshapeWindow ( int ( _nWidth ), int ( _nHeight ) );
    return;
}

int main ( int argc, char** argv ) {
    try {
		_pKinect.reset(new btl::kinect::VideoSourceKinect);
		_pGL.reset(new btl::gl_util::CGLUtil(btl::utility::BTL_CV));
		_pKinect->_pFrame->_pGL = _pGL.get();
		_pKinect->_pFrame->_bGPURender = true;
		//_pRGBCamera=_pKinect->_pRGBCamera.get();
		
        glutInit ( &argc, argv );
        glutInitDisplayMode ( GLUT_DOUBLE | GLUT_RGB );
        glutInitWindowSize ( 1280, 480 );
        glutCreateWindow ( "CameraPose" );
        init();
        glutKeyboardFunc ( normalKeys );
		glutSpecialFunc ( specialKeys );
        glutMouseFunc   ( mouseClick );
        glutMotionFunc  ( mouseMotion );

        glutReshapeFunc ( reshape );
        glutDisplayFunc ( display );
        glutMainLoop();
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


