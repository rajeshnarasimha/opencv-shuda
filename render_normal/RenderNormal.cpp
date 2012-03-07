//display kinect depth in real-time
#define INFO
#define TIMER
#include <GL/glew.h>
#include <gl/freeglut.h>
//#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <string>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "Utility.hpp"

//camera calibration from a sequence of images
#include <opencv2/gpu/gpu.hpp>
#include <XnCppWrapper.h>
#include <gl/freeglut.h>
#include "Kinect.h"
#include "Camera.h"
#include "EigenUtil.hpp"
#include "GLUtil.h"
#include "PlaneObj.h"
#include "Histogram.h"
#include "KeyFrame.h"
#include "VideoSourceKinect.hpp"
#include "Model.h"
#include "GLUtil.h"

using namespace Eigen;

class CKinectView;

btl::kinect::VideoSourceKinect::tp_shared_ptr _pKinect;
btl::gl_util::CGLUtil::tp_shared_ptr _pGL;
btl::geometry::CModel::tp_shared_ptr _pModel;

Matrix4d _mGLMatrix;
double _dNear = 0.01;
double _dFar  = 10.;

unsigned short _nWidth, _nHeight;

bool _bCaptureCurrentFrame = false;
bool _bRenderNormal = false;
bool _bEnableLighting = false;
double _dDepthFilterThreshold = 0.01;
int _nDensity = 2;
float _fSize = 0.2f; // range from 0.05 to 1 by step 0.05
unsigned int _uPyrHeight = 4;
int _nColorIdx = 0;
bool _bRenderPlane = true;
bool _bGpuPlane = true;
bool _bGPURender = true;

btl::kinect::CKeyFrame::tp_cluster _enumType = btl::kinect::CKeyFrame::NORMAL_CLUSTER;

void normalKeys ( unsigned char key, int x, int y )
{
    switch( key )
    {
	case 'p':
		//use current keyframe as a reference
		_bRenderPlane =! _bRenderPlane;
		glutPostRedisplay();
    case 'c':
        //capture current frame the depth map and color
        _bCaptureCurrentFrame = true;
        break;
    case '>':
        _dDepthFilterThreshold += 0.01;
        PRINT( _dDepthFilterThreshold );
        glutPostRedisplay();
        break;
    case '<':
        _dDepthFilterThreshold -= 0.011;
        _dDepthFilterThreshold = _dDepthFilterThreshold > 0? _dDepthFilterThreshold : 1;
        PRINT( _dDepthFilterThreshold );
        glutPostRedisplay();
        break;
    case 'n':
        _bRenderNormal = !_bRenderNormal;
        glutPostRedisplay();
		PRINT( _bRenderNormal );
        break;
    case 'l':
        _bEnableLighting = !_bEnableLighting;
        glutPostRedisplay();
		PRINT( _bEnableLighting );
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
    case 'k':
        _fSize += 0.05f;// range from 0.05 to 1 by step 0.05
        _fSize = _fSize < 1.f ? _fSize: 1.f;
        glutPostRedisplay();
        PRINT( _fSize );
        break;
    case 'j':
        _fSize -= 0.05f;
        _fSize = _fSize > 0.05f? _fSize : 0.05f;
        glutPostRedisplay();
        PRINT( _fSize );
        break;
	case '0':
		//_pKinect->_pFrame->setView2(_pGL->_adModelViewGL);
		_pKinect->_pFrame->setView(&_pGL->_eimModelViewGL);
		break;
	case '7':
		_bGPURender = !_bGPURender;
		glutPostRedisplay();
		break;
	case '8':
		_bGpuPlane = !_bGpuPlane;
		glutPostRedisplay();
	case ']':
		_pKinect->_fSigmaSpace += 1;
		PRINT( _pKinect->_fSigmaSpace );
		break;
	case '[':
		_pKinect->_fSigmaSpace -= 1;
		PRINT( _pKinect->_fSigmaSpace );
		break;
    }
	_pGL->normalKeys( key, x, y );
    return;
}
void specialKeys(int nKey_,int x, int y)
{
	_pGL->specialKeys(nKey_,x,y);
	switch( nKey_ )
	{
	case GLUT_KEY_F4: //display camera
		_nColorIdx++;
		PRINT(_nColorIdx);
		break;
	case GLUT_KEY_F5:
		_enumType = btl::kinect::CKeyFrame::NORMAL_CLUSTER == _enumType? btl::kinect::CKeyFrame::DISTANCE_CLUSTER : btl::kinect::CKeyFrame::NORMAL_CLUSTER;
		if(btl::kinect::CKeyFrame::NORMAL_CLUSTER == _enumType) {
			PRINTSTR( "NORMAL_CLUSTER" );
		}
		else{
			PRINTSTR( "DISTANCE_CLUSTER" );
		}
		break;
	}
}

void mouseClick ( int nButton_, int nState_, int nX_, int nY_ ){
	_pGL->mouseClick(nButton_,nState_,nX_,nY_);
}

void mouseMotion ( int nX_, int nY_ ){
	_pGL->mouseMotion(nX_,nY_);
}

void display ( void )
{
	//if(_bCaptureCurrentFrame) 
	{
		_pGL->timerStart();
		_pKinect->getNextPyramid(4,btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV);
		PRINTSTR("Pyramid")
		_pGL->timerStop();
		_pGL->timerStart();
		_pKinect->_pFrame->_bGPURender = _bGPURender;
		_pKinect->_pFrame->_bRenderPlane = false;
		_pKinect->_pFrame->gpuDetectPlane(_pGL->_uLevel);
		PRINTSTR("Plane")
		_pGL->timerStop();
		_bCaptureCurrentFrame = false;
	}

    glMatrixMode ( GL_MODELVIEW );
    glViewport (0, 0, _nWidth/2, _nHeight);
    glScissor  (0, 0, _nWidth/2, _nHeight);
    // after set the intrinsics and extrinsics
    // load the matrix to set camera pose
    glLoadIdentity();
    //glLoadMatrixd( _mGLMatrix.data() );
    _pGL->viewerGL();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // render objects
    _pGL->renderAxisGL();
	_pGL->renderVoxelGL(2.f);
	_pModel->gpuRenderVoxelInWorldCVGL();
    //render3DPts();
	_pKinect->_pFrame->_bRenderPlane = _bRenderPlane;
	_pKinect->_pFrame->_eClusterType = _enumType;
	_pGL->timerStart();
	_pKinect->_pFrame->renderCameraInGLWorld(_pGL->_bDisplayCamera,true,true,.05f,_pGL->_uLevel);
	PRINTSTR("renderCameraInGLWorld()")
	_pGL->timerStop();
    glViewport (_nWidth/2, 0, _nWidth/2, _nHeight);
    glScissor  (_nWidth/2, 0, _nWidth/2, _nHeight);
    //gluLookAt ( _eivCamera(0), _eivCamera(1), _eivCamera(2),  _eivCentroid(0), _eivCentroid(1), _eivCentroid(2), _eivUp(0), _eivUp(1), _eivUp(2) );
    //glScaled( _dZoom, _dZoom, _dZoom );
    //glRotated ( _dYAngle, 0, 1 ,0 );
    //glRotated ( _dXAngle, 1, 0 ,0 );
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    // render objects
	_pGL->timerStart();
	_pKinect->_pRGBCamera->LoadTexture( *_pKinect->_pFrame->_acvmShrPtrPyrBWs[_pGL->_uLevel],&(_pKinect->_pFrame->_uTexture) );
	_pKinect->_pRGBCamera->renderCameraInGLLocal( _pKinect->_pFrame->_uTexture,*_pKinect->_pFrame->_acvmShrPtrPyrBWs[_pGL->_uLevel] );
	PRINTSTR("Camera");
	_pGL->timerStop();


    glutSwapBuffers();
    glutPostRedisplay();
}//display()

void reshape ( int nWidth_, int nHeight_ ){
    _pKinect->_pRGBCamera->setIntrinsics( 1, 0.01, 100 );
    // setup blending
    //glBlendFunc ( GL_SRC_ALPHA, GL_ONE );			// Set The Blending Function For Translucency
    //glColor4f ( 1.0f, 1.0f, 1.0f, 1.0f );
    unsigned short nTemp = nWidth_/8;//make sure that _nWidth is divisible to 4
    _nWidth = nTemp*8;
    _nHeight = nTemp*3;
    glutReshapeWindow( int ( _nWidth ), int ( _nHeight ) );
    return;
}

void init ( ){
	_pGL->clearColorDepth();
    glDepthFunc  ( GL_LESS );
    glEnable     ( GL_DEPTH_TEST );
    glEnable 	 ( GL_SCISSOR_TEST );
    glEnable     ( GL_CULL_FACE );
    glShadeModel ( GL_FLAT );
    glPixelStorei( GL_UNPACK_ALIGNMENT, 1);
	_pGL->_uLevel=0;
	_pKinect->getNextPyramid(4,btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV);
 	_pKinect->_pFrame->gpuDetectPlane(_pGL->_uLevel);

	_pGL->init();
}

int main ( int argc, char** argv ){
    try{


        glutInit ( &argc, argv );
        glutInitDisplayMode ( GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH );
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
		
		//btl::gl_util::CGLUtil::glBindBuffer    = (PFNGLBINDBUFFERARBPROC)   wglGetProcAddress("glBindBuffer");
		//btl::gl_util::CGLUtil::glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)wglGetProcAddress("glDeleteBuffers");
		//btl::gl_util::CGLUtil::glGenBuffers    = (PFNGLGENBUFFERSARBPROC)   wglGetProcAddress("glGenBuffers");
		//btl::gl_util::CGLUtil::glBufferData    = (PFNGLBUFFERDATAARBPROC)   wglGetProcAddress("glBufferData");

		_pGL.reset( new btl::gl_util::CGLUtil(btl::utility::BTL_CV) );
		_pGL->initVBO();
		_pKinect.reset( new btl::kinect::VideoSourceKinect() );
		_pKinect->_pFrame->_pGL=_pGL.get();
		_pModel.reset( new btl::geometry::CModel() );
		_pModel->_pGL=_pGL.get();
		init();
        _pModel->gpuCreateVBO();
		glutMainLoop();
    }
	/*
    catch ( CError& e ) {
        if ( std::string const* mi = boost::get_error_info< CErrorInfo > ( e ) ){
            std::cerr << "Error Info: " << *mi << std::endl;
        }
    }*/
	catch ( std::runtime_error& e )	{
		PRINTSTR( e.what() );
	}
    return 0;
}//main()


