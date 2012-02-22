//display kinect depth in real-time
#define INFO

#include <iostream>
#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <Converters.hpp>
#include <opencv2/gpu/gpumat.hpp>
#include <utility>
#include <boost/lexical_cast.hpp>
#include <gl/freeglut.h>
#include <XnCppWrapper.h>
#include "GLUtil.h"
#include "EigenUtil.hpp"
#include "Camera.h"
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include "KeyFrame.h"
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <VideoSourceKinect.hpp>

//camera calibration from a sequence of images

using namespace cv;

//class CKinectView;
//class KeyPoint;

btl::kinect::VideoSourceKinect::tp_shared_ptr _pKinect;
btl::gl_util::CGLUtil::tp_shared_ptr _pGL;
btl::kinect::SCamera::tp_ptr _pRGBCamera;

double _dNear = 0.01;
double _dFar  = 10.;

unsigned short _nWidth, _nHeight;

btl::kinect::CKeyFrame::tp_shared_ptr _aShrPtrKFs[10];
int _nKFCounter = 1; //key frame counter
int _nRFCounter = 0; //reference frame counter
std::vector< btl::kinect::CKeyFrame::tp_shared_ptr* > _vShrPtrsKF;
std::vector< int > _vRFIdx;

bool _bContinuous = true;
bool _bPrevStatus = true;
bool _bDisplayCamera = true;
bool _bRenderReference = true;
bool _bCapture = false;

int _nN = 1;
int _nView = 0;

void init();

void specialKeys( int key, int x, int y ){
	switch ( key ) {
	case GLUT_KEY_F2: //display camera
		_bDisplayCamera = !_bDisplayCamera;
		glutPostRedisplay();
		break;
	case GLUT_KEY_F3:
		_bRenderReference = !_bRenderReference;
		glutPostRedisplay();
		break;
	}
}

void normalKeys ( unsigned char key, int x, int y ){
    switch ( key ) {
    case 'r':
        //reset
		_nKFCounter=1;
		_nRFCounter=0;
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
        _bCapture = true;
        break;
	case 'd':
		//remove last key frame
		if(_nKFCounter >0 ) {
			if( _nRFCounter ==  _nKFCounter ) _nRFCounter--;
			_nKFCounter--;
			_vShrPtrsKF.pop_back();
		}
		glutPostRedisplay();
		break;
	/*
	case 'v':
			//use current keyframe as a reference
			if( _nRFCounter <_nKFCounter )
			{
				_nRFCounter = _nKFCounter-1;
				_vRFIdx.push_back( _nRFCounter );
				SKeyFrame<float>& s1stKF = _aShrPtrKFs[_nRFCounter];
				s1stKF._bIsReferenceFrame = true;
				glutPostRedisplay();
			}
			break;*/
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
	for(int i=0; i <10; i++){ _aShrPtrKFs[i].reset(new btl::kinect::CKeyFrame(_pRGBCamera));	}
    
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
    // load as texture
    _pRGBCamera->LoadTexture ( *_pKinect->_pFrame->_acvmShrPtrPyrRGBs[0] );
	btl::kinect::CKeyFrame::tp_shared_ptr& p1stKF = _aShrPtrKFs[0];
	_vRFIdx.push_back(0);
    // assign the rgb and depth to the current frame.
    p1stKF->assign ( *_pKinect->_pFrame->_acvmShrPtrPyrRGBs[0], (const float*)_pKinect->_pFrame->_acvmShrPtrPyrPts[0]->data );
    // corner detection and ranking ( first frame )
    p1stKF->detectCorners();
	p1stKF->_bIsReferenceFrame = true;
	p1stKF->setView(&_pGL->_eimModelViewGL);
	_vShrPtrsKF.push_back( &p1stKF );
    return;
}

void display ( void ) {
// update frame
    _pKinect->getNextFrame(btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV);
// ( second frame )
    // assign the rgb and depth to the current frame.
	btl::kinect::CKeyFrame::tp_shared_ptr& pCurrentKF = _aShrPtrKFs[_nKFCounter];
    pCurrentKF->assign ( *_pKinect->_pFrame->_acvmShrPtrPyrRGBs[0],  (const float*)_pKinect->_pFrame->_acvmShrPtrPyrPts[0]->data );

    if ( _bCapture && _nKFCounter < 10 ) {
		btl::kinect::CKeyFrame::tp_shared_ptr& p1stKF = _aShrPtrKFs[_nRFCounter];
        _bCapture = false;
        // detect corners
        pCurrentKF->detectCorners();
        pCurrentKF->detectCorrespondences ( *p1stKF );
        pCurrentKF->calcRT ( *p1stKF );
 		pCurrentKF->applyRelativePose( *p1stKF );
		_vShrPtrsKF.push_back( &pCurrentKF );
		_nKFCounter++;
		std::cout << "new key frame added" << std::flush;
    }
	else if( _nKFCounter > 49 )	{
		std::cout << "two many key frames to hold" << std::flush;  
	}

// render first viewport
    glMatrixMode ( GL_MODELVIEW );
    glViewport ( 0, 0, _nWidth / 2, _nHeight );
    glScissor  ( 0, 0, _nWidth / 2, _nHeight );
    // after set the intrinsics and extrinsics
    //glLoadIdentity();
	_pGL->viewerGL();

    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // render objects
	for( vector< btl::kinect::CKeyFrame::tp_shared_ptr* >::iterator cit = _vShrPtrsKF.begin(); cit!= _vShrPtrsKF.end(); cit++ ) {
		(**cit)->renderCamera( _bDisplayCamera );
	}

	if(_bRenderReference) {
		_pGL->renderAxisGL();
		_pGL->renderPatternGL(.1f,20.f,20.f);
		_pGL->renderPatternGL(1.f,10.f,10.f);
		_pGL->renderVoxelGL(2.f);
		_pGL->renderOctTree(0.f,0.f,0.f,2.f,2);
	}

// render second viewport
    glViewport ( _nWidth/2, 0, _nWidth/2, _nHeight );
    glScissor  ( _nWidth/2, 0, _nWidth/2, _nHeight );
    glLoadIdentity();
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	_pRGBCamera->LoadTexture(*_pKinect->_pFrame->_acvmShrPtrPyrRGBs[0]);
    _pRGBCamera->renderCamera( *_pKinect->_pFrame->_acvmShrPtrPyrRGBs[0], .2 );

// rendering
    /*
    //corners at the first frame
    glPointSize ( 3 );
    glColor3d ( 1, 0, 0 );
    glBegin ( GL_POINTS );
    for ( vector< Point2f >::const_iterator cit = _sPreviousKF._vCorners.begin(); cit != _sPreviousKF._vCorners.end(); cit++ )
    {
        _cView.renderOnImage ( cit->x, cit->y );
    }
    glEnd();
	*/
    glutSwapBuffers();

    if ( _bContinuous ) {
        glutPostRedisplay();
    }
}

void reshape ( int nWidth_, int nHeight_ ) {
    //cout << "reshape() " << endl;
    _pRGBCamera->setIntrinsics ( 1, 0.01, 100 );

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
		_pRGBCamera=_pKinect->_pRGBCamera.get();
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
		if ( string const* mi = boost::get_error_info< btl::utility::CErrorInfo > ( e ) )	{
			std::cerr << "Error Info: " << *mi << std::endl;
		}
	}
	catch ( std::runtime_error& e )	{
		PRINTSTR( e.what() );
	}

    return 0;
}


