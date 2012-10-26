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
#include "PlaneWorld.h"
#include "GLUtil.h"

using namespace Eigen;

class CKinectView;

btl::kinect::VideoSourceKinect::tp_shared_ptr _pKinect;
btl::gl_util::CGLUtil::tp_shared_ptr _pGL;
//btl::geometry::CModel::tp_shared_ptr _pModel;
//btl::geometry::CMultiPlanesMultiViewsInWorld::tp_shared_ptr _pMPMV;

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
unsigned short _usColorIdx = 0;
bool _bRenderPlane = true;
bool _bRenderDepth = true;
ushort _usViewNO = 0;
ushort _usPlaneNO = 0;

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
        _bCaptureCurrentFrame = !_bCaptureCurrentFrame;
        break;
    case '>':
        _dDepthFilterThreshold += 0.01;
        //PRINT( _dDepthFilterThreshold );
        glutPostRedisplay();
        break;
    case '<':
        _dDepthFilterThreshold -= 0.011;
        _dDepthFilterThreshold = _dDepthFilterThreshold > 0? _dDepthFilterThreshold : 1;
        //PRINT( _dDepthFilterThreshold );
        glutPostRedisplay();
        break;
    case 'n':
        _bRenderNormal = !_bRenderNormal;
        glutPostRedisplay();
		//PRINT( _bRenderNormal );
        break;
    case 'l':
        _bEnableLighting = !_bEnableLighting;
        glutPostRedisplay();
		//PRINT( _bEnableLighting );
        break;
    case 'q':
        _nDensity++;
        glutPostRedisplay();
        //PRINT( _nDensity );
        break;
    case ';':
        _nDensity--;
        _nDensity = _nDensity > 0 ? _nDensity : 1;
        glutPostRedisplay();
        //PRINT( _nDensity );
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
		//_usViewNO = ++_usViewNO % _pMPMV->_vShrPtrMPSV.size(); 
		//_pMPMV->_vShrPtrMPSV[_usViewNO]->_pFrame->setView(&_pGL->_eimModelViewGL);
		_pKinect->_pFrame->setView(&_pGL->_eimModelViewGL);
		break;
	case '1':
		_usViewNO++;
		PRINT(_usViewNO);
		glutPostRedisplay();
		break;
	case '2':
		_usPlaneNO++;
		PRINT(_usPlaneNO);
		glutPostRedisplay();
		break;
	case '7':
		_bRenderDepth = !_bRenderDepth;
		glutPostRedisplay();
		break;
	case '8':
		_bRenderPlane = !_bRenderPlane;
		glutPostRedisplay();
	case ']':
		_pKinect->_fSigmaSpace += 1;
		//PRINT( _pKinect->_fSigmaSpace );
		break;
	case '[':
		_pKinect->_fSigmaSpace -= 1;
		//PRINT( _pKinect->_fSigmaSpace );
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
	case GLUT_KEY_F6: //display camera
		_usColorIdx++;
		_pKinect->_pFrame->_nColorIdx = _usColorIdx;
		glutPostRedisplay();
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
	_pGL->timerStart();
	_pGL->errorDetectorGL();
	if(/*_bContinuous &&*/ _bCaptureCurrentFrame) 
	{
		_pKinect->getNextFrame(btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV);
		for (ushort u=0;u<_pKinect->_uPyrHeight;u++){
			_pKinect->_pFrame->gpuDetectPlane(u);
			_pKinect->_pFrame->gpuTransformToWorldCVCV(u);
		}//for each pyramid level
	
		//_pMPMV->integrateFrameIntoPlanesWorldCVCV(_pKinect->_pFrame.get());
		//_bCaptureCurrentFrame = false;
	}
    glMatrixMode ( GL_MODELVIEW );
    glViewport (0, 0, _nWidth, _nHeight);
    glScissor  (0, 0, _nWidth, _nHeight);
    // after set the intrinsics and extrinsics
    // load the matrix to set camera pose
    glLoadIdentity();
    //glLoadMatrixd( _mGLMatrix.data() );
    _pGL->viewerGL();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // render objects
    _pGL->renderAxisGL();
	//_pGL->renderVoxelGL(2.f);
	//_pModel->gpuIntegrateFrameIntoVolumeCVCV(*_pKinect->_pFrame,_pGL->_usPyrHeight);
	//_pModel->gpuRenderVoxelInWorldCVGL();
	//render all planes in the first frame
	
	//render first plane in multi-view
	//if(_pMPMV->_vShrPtrSPMV.size()>2) _pMPMV->_vShrPtrSPMV[1]->renderPlaneInAllViewsWorldGL(_pGL.get(),_usColorIdx,3);
	//render all planes in single view
	//_pMPMV->renderAllPlanesInGivenViewWorldCVGL(_pGL.get(),_usColorIdx,3,_usViewNO);
	//_pMPMV->renderGivenPlaneInGivenViewWorldCVGL(_pGL.get(),_usColorIdx,3,_usViewNO,_usPlaneNO);
	//_pMPMV->renderGivenPlaneInAllViewWorldCVGL(_pGL.get(),_usColorIdx,3,_usPlaneNO);
	//

	_pKinect->_pFrame->renderCameraInWorldCVCV(_pGL.get(),_pGL->_bDisplayCamera,.05f,_pGL->_usLevel);
	//_pKinect->_pFrame->gpuRenderPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel);
	_pKinect->_pFrame->renderPlanesInWorld(_pGL.get(),0,_pGL->_usLevel);
	//_pKinect->_pFrame->render3DPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel,_usColorIdx,_bRenderPlane );

	//_pMPMV->renderAllCamrea(_pGL.get(),true,_bRenderDepth,_usViewNO,.05f);
	/*
    glViewport (_nWidth/2, 0, _nWidth/2, _nHeight);
    glScissor  (_nWidth/2, 0, _nWidth/2, _nHeight);
    //gluLookAt ( _eivCamera(0), _eivCamera(1), _eivCamera(2),  _eivCentroid(0), _eivCentroid(1), _eivCentroid(2), _eivUp(0), _eivUp(1), _eivUp(2) );
    //glScaled( _dZoom, _dZoom, _dZoom );
    //glRotated ( _dYAngle, 0, 1 ,0 );
    //glRotated ( _dXAngle, 1, 0 ,0 );
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	Eigen::Matrix4d eimModelViewGL;
	_pKinect->_pFrame->setView(&eimModelViewGL);
	glLoadMatrixd(eimModelViewGL.data());
	_pKinect->_pFrame->renderCameraInWorldCVCV(_pGL.get(),true,4.2f,_pGL->_usLevel);
	_pKinect->_pFrame->gpuRenderPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel);
    */
    glutSwapBuffers();
    glutPostRedisplay();
	_pGL->timerStop();

}//display()

void reshape ( int nWidth_, int nHeight_ ){
    _pKinect->_pRGBCamera->setIntrinsics( 1, 0.01, 100 );
    // setup blending
    //glBlendFunc ( GL_SRC_ALPHA, GL_ONE );			// Set The Blending Function For Translucency
    //glColor4f ( 1.0f, 1.0f, 1.0f, 1.0f );
    unsigned short nTemp = nWidth_/4;//make sure that _nWidth is divisible to 4
    _nWidth = nTemp*4;
    _nHeight = nTemp*3;
    //glutReshapeWindow( int ( _nWidth ), int ( _nHeight ) );
	glutReshapeWindow( 640, 480 );
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
	_pGL->_usLevel=0;
	_pKinect->getNextFrame(btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV);
	_pKinect->_pFrame->_bGPURender = _bRenderDepth;
	_pKinect->_pFrame->_bRenderPlane = true;
	for (ushort u=0;u<_pKinect->_uPyrHeight;u++){
		_pKinect->_pFrame->gpuDetectPlane(u);
		_pKinect->_pFrame->gpuTransformToWorldCVCV(u);
	}//for each pyramid level
	//_pMPMV->integrateFrameIntoPlanesWorldCVCV(_pKinect->_pFrame.get(),3);
	//_pMPMV.reset( new btl::geometry::CMultiPlanesMultiViewsInWorld( _pKinect->_pFrame.get() ) );

	_pGL->init();
}

int main ( int argc, char** argv ){
    try{
        glutInit ( &argc, argv );
        glutInitDisplayMode ( GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH );
        glutInitWindowSize ( 640, 480 );//1280
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
		_pGL->setCudaDeviceForGLInteroperation();//initialize before using any cuda component
		_pKinect.reset( new btl::kinect::VideoSourceKinect(1) );
		
		init();
		_pGL->constructVBOsPBOs();
		glutMainLoop();
		_pGL->destroyVBOsPBOs();
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


