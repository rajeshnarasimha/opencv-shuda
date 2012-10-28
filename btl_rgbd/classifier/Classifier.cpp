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
#include "KeyFrame.h"
#include "VideoSourceKinect.hpp"
#include "KinfuTracker.h"
#include "GLUtil.h"
//camera calibration from a sequence of images
#include "Camera.h"

btl::kinect::VideoSourceKinect::tp_shared_ptr _pKinect;
btl::gl_util::CGLUtil::tp_shared_ptr _pGL;

unsigned short _nWidth, _nHeight;
double _dDepthFilterThreshold = 10;
int _nDensity = 2;

bool _bCaptureCurrentFrame = true;
float _fThreshold = 0.02;

void specialKeys( int key, int x, int y ){
	_pGL->specialKeys( key, x, y );
}

void normalKeys ( unsigned char key, int x, int y )
{
	switch( key )
	{
	case 27:
		exit ( 0 );
		break;
	case '>':
		_dDepthFilterThreshold += 10.0;
		PRINT( _dDepthFilterThreshold );
		glutPostRedisplay();
		break;
	case '<':
		_dDepthFilterThreshold -= 11.0;
		_dDepthFilterThreshold = _dDepthFilterThreshold > 0? _dDepthFilterThreshold : 1;
		PRINT( _dDepthFilterThreshold );
		glutPostRedisplay();
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
		//_pKinect->_pFrame->setView2(_pGL->_adModelViewGL);
		_pKinect->_pFrame->setView(&_pGL->_eimModelViewGL);
		break;
	case 't':
		_fThreshold += 0.01;
		glutPostRedisplay();
		break;
	case 'n':
		_fThreshold -= 0.01;
		glutPostRedisplay();
		break;
	case '[':
		_pKinect->_fSigmaSpace -= 1;
		PRINT( _pKinect->_fSigmaSpace );
		break;
	case 'c':
		_bCaptureCurrentFrame = !_bCaptureCurrentFrame;
		glutPostRedisplay();
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
	if( _bCaptureCurrentFrame )	
	{
		_pKinect->getNextFrame(btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV);
		_pKinect->_pFrame->gpuTransformToWorldCVCV(_pGL->_usLevel);
		_pKinect->_pFrame->gpuBoundaryDetector(_fThreshold,_pGL->_usLevel);
	}
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
	_pGL->renderPatternGL(.1f,20.f,20.f);
	_pGL->renderPatternGL(1.f,10.f,10.f);
	_pGL->renderVoxelGL(4.f);
	//_pGL->timerStart();
	_pKinect->_pFrame->renderCameraInWorldCVCV(_pGL.get(),_pGL->_bDisplayCamera,_pGL->_fSize,_pGL->_usLevel);
	_pKinect->_pFrame->gpuRenderPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel);
	//_pKinect->_pFrame->render3DPtsInWorldCVCV(_pGL.get(),_pGL->_usPyrLevel,0,false);

	//PRINTSTR("renderCameraInGLWorld");
	//_pGL->timerStop();

	//set viewport 2
	glViewport (_nWidth/2, 0, _nWidth/2, _nHeight);
	glScissor  (_nWidth/2, 0, _nWidth/2, _nHeight);
	glLoadIdentity();
	//gluLookAt ( _eivCamera(0), _eivCamera(1), _eivCamera(2),  _eivCenter(0), _eivCenter(1), _eivCenter(2), _eivUp(0), _eivUp(1), _eivUp(2) );
    //glScaled( _dZoom, _dZoom, _dZoom );    
    //glRotated ( _dYAngle, 0, 1 ,0 );
    //glR	//if(_bRenderVolume) _pModel->gpuRenderVoxelInWorldCVGL();
	//_pKinect->_pFrame->renderCameraInWorldCVGL(_pGL.get(),0,_pGL->_bDisplayCamera,true,true,_pGL->_fSize,_pGL->_usPyrLevel);otated ( _dXAngle, 1, 0 ,0 );
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// render objects
    _pGL->renderAxisGL();
	//render3DPts();
	_pKinect->_pRGBCamera->LoadTexture(*_pKinect->_pFrame->_acvmShrPtrPyrRGBs[_pGL->_usLevel],&_pGL->_auTexture[_pGL->_usLevel]);
	_pKinect->_pRGBCamera->renderCameraInGLLocal(_pGL->_auTexture[_pGL->_usLevel], *_pKinect->_pFrame->_acvmShrPtrPyrRGBs[_pGL->_usLevel] );

    glutSwapBuffers();
	glutPostRedisplay();
	return;
}
void reshape ( int nWidth_, int nHeight_ ){
	//cout << "reshape() " << endl;
    _pKinect->_pRGBCamera->setIntrinsics( 1, 0.01, 100 );

    // setup blending 
    glBlendFunc ( GL_SRC_ALPHA, GL_ONE );			// Set The Blending Function For Translucency
    glColor4f ( 1.0f, 1.0f, 1.0f, 0.5 );

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

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	_pKinect->getNextFrame(btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV);
	_pKinect->_pFrame->gpuTransformToWorldCVCV(_pGL->_usLevel);
	
	_pGL->init();
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
		_pGL.reset( new btl::gl_util::CGLUtil(1,3,btl::utility::BTL_CV) );
		_pGL->setCudaDeviceForGLInteroperation();
		_pKinect.reset(new btl::kinect::VideoSourceKinect(1,true));
		init();
		_pGL->constructVBOsPBOs();
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
