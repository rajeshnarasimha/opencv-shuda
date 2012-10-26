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
#include "PlaneWorld.h"
#include "Model.h"
#define _nReserved 20

btl::kinect::VideoSourceKinect::tp_shared_ptr _pKinect;
btl::gl_util::CGLUtil::tp_shared_ptr _pGL;
btl::geometry::CMultiPlanesMultiViewsInWorld::tp_shared_ptr _pMPMV;
btl::geometry::CModel::tp_shared_ptr _pVolumeWorld;
btl::kinect::CKeyFrame::tp_shared_ptr _pVirtualFrame;
btl::kinect::CKeyFrame::tp_shared_ptr _pDisplayFrame;
unsigned short _nWidth, _nHeight;

btl::kinect::CKeyFrame::tp_shared_ptr _aShrPtrKFs[_nReserved];
btl::kinect::CKeyFrame::tp_shared_ptr _aShrPtrRenderFrms[3];

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
unsigned short _usColorIdx=0;
ushort _usViewNO = 0;
ushort _usPlaneNO = 0;
ushort _uResolution = 1;
ushort _uPyrHeight = 3;

void init ( ){
	for(int i=0; i <_nReserved; i++){ 
		_aShrPtrKFs[i].reset(new btl::kinect::CKeyFrame(_pKinect->_pRGBCamera.get(),1,3));	
	}
	_pVirtualFrame.reset(new btl::kinect::CKeyFrame(_pKinect->_pRGBCamera.get(),1,3));	

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
	btl::kinect::CKeyFrame::tp_shared_ptr& p1stKF = _aShrPtrKFs[_nRFIdx];
	_pKinect->getNextFrame(btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV);
	_pKinect->_pFrame->copyTo(&*p1stKF);
	p1stKF->extractSurfFeatures();
	p1stKF->gpuTransformToWorldCVCV();

	std::string strPath("C:\\csxsl\\src\\opencv-shuda\\Data\\");
	std::string strFileName =  boost::lexical_cast<std::string> ( _nRFIdx ) + ".yml";
	//p1stKF->exportYML(strPath,strFileName);
	//p1stKF->importYML(strPath,strFileName);
	_pVolumeWorld->gpuIntegrateFrameIntoVolumeCVCV(*p1stKF);

	// assign the rgb and depth to the current frame.
	p1stKF->setView(&_pGL->_eimModelViewGL);
	_vShrPtrsKF.push_back( &p1stKF );
	return;
}
void specialKeys( int key, int x, int y ){
	_pGL->specialKeys( key, x, y );
	switch ( key ) {
	case GLUT_KEY_F6: //display camera
		_usColorIdx++;
		for(unsigned int i=0; i < _nKFCounter; i++)	{
			_aShrPtrKFs[i]->_nColorIdx = _usColorIdx;
		}
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
		_vShrPtrsKF.clear();
        init();
		_pVolumeWorld->reset();
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
			_nKFCounter--;
			_vShrPtrsKF.pop_back();
		}
		glutPostRedisplay();
		break;
	case '1':
		_usViewNO = ++_usViewNO % _nKFCounter; 
		//(*_vShrPtrsKF[ _usViewNO ])->setView(&_pGL->_eimModelViewGL);
		glutPostRedisplay();
		break;
	case '2':
		_usPlaneNO++;
		glutPostRedisplay();
		break;
	case '8':
			//use current keyframe as a reference
			_bRenderReference =! _bRenderReference;
			for(unsigned int i=0; i < _nKFCounter; i++)	{
				/*for (unsigned short u=0; u<4; u++)
				{
					_aShrPtrKFs[i]->gpuDetectPlane(u);
				}*/
			}
			glutPostRedisplay();
			break;
	case '0':
		_usViewNO = ++_usViewNO % _vShrPtrsKF.size(); 
		(*_vShrPtrsKF[ _usViewNO ])->setView(&_pGL->_eimModelViewGL);
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
	unsigned short uInliers;
    if ( false && _bCapture && _nKFCounter < _nReserved ) {
		// assign the rgb and depth to the current frame.
		btl::kinect::CKeyFrame::tp_shared_ptr& pReferenceKF = _aShrPtrKFs[_nRFIdx];
		btl::kinect::CKeyFrame::tp_shared_ptr& pCurrentKF = _aShrPtrKFs[_nKFCounter];
		//attach surf features to planes
		_pKinect->_pFrame->extractSurfFeatures();
		//track camera motion
		_pKinect->_pFrame->calcRT ( *pReferenceKF,0,.5,&uInliers ); //roughly estimate R,T w.r.t. last key frame,
		_pKinect->_pFrame->gpuICP ( pReferenceKF.get(), false ); //refine the R,T with w.r.t. last key frame
		for (short sIt = 0; sIt< 3; sIt++){
			_pVolumeWorld->gpuRaycast( *_pKinect->_pFrame, &*_pVirtualFrame ); //get virtual frame
			_pKinect->_pFrame->gpuICP ( _pVirtualFrame.get(), false ); //refine R,T w.r.t. the virtual frame
		}//iterate 3 times 
		
		//transform pts and nls to the world
		for (ushort usI=0;usI<4;usI++){
			_pKinect->_pFrame->gpuTransformToWorldCVCV(usI);
		}
		//transform detected planes to the world
		_pKinect->_pFrame->transformPlaneObjsToWorldCVCV(3);
		//integrate current frame into the global volume
		_pVolumeWorld->gpuIntegrateFrameIntoVolumeCVCV(*_pKinect->_pFrame);
		//_pMPMV->integrateFrameIntoPlanesWorldCVCV(pCurrentKF.get());
		//save current frame and make it as the reference frame
		_pKinect->_pFrame->copyTo(&*pCurrentKF);
		_vShrPtrsKF.push_back( &pCurrentKF );
		_nKFCounter++;_nRFIdx++;
		std::cout << "new key frame added" << std::flush;
		_bCapture = false;
    }//if step by step tracking 
	else if ( _bCapture && _nKFCounter < _nReserved){
		// assign the rgb and depth to the current frame.
		_nRFIdx++;
		std::string strPath("C:\\csxsl\\src\\opencv-shuda\\Data\\");
		std::string strFileName =  boost::lexical_cast<std::string> ( _nRFIdx ) + ".yml";
		//_pKinect->_pFrame->exportYML(strPath,strFileName);

		// assign the rgb and depth to the current frame.
		btl::kinect::CKeyFrame::tp_shared_ptr& pReferenceKF = _aShrPtrKFs[_nRFIdx];//_aShrPtrKFs[_nReserved]
		
		//attach surf features to planes
		_pKinect->_pFrame->extractSurfFeatures();
		//track camera motion
		double dError = _pKinect->_pFrame->calcRT ( *pReferenceKF,0,.5,&uInliers ); //roughly estimate R,T w.r.t. last key frame,
		if (dError < 0.05) 
		{
			PRINTSTR("Surf calibration.");
			_pGL->timerStop();
			_pKinect->_pFrame->gpuICP ( pReferenceKF.get(), false );//refine the R,T with w.r.t. last key frame
			_pVolumeWorld->gpuRaycast( *_pKinect->_pFrame, &*_pVirtualFrame );
			//for (short sIt = 0; sIt< 3; sIt++){
			//	_pVolumeWorld->gpuRaycast( *_pKinect->_pFrame, &*_pVirtualFrame ); //get virtual frame
			//	_pKinect->_pFrame->gpuICP ( _pVirtualFrame.get(), false );//refine R,T w.r.t. the virtual frame
			//}//iterate 3 times 
			PRINTSTR("ICP tracking.");
			_pGL->timerStop();
			//detect planes
			//_pKinect->_pFrame->gpuDetectPlane(3);
			//transform pts and nls to the world
			//_pKinect->_pFrame->gpuTransformToWorldCVCV();
			//transform detected planes to the world
			//_pKinect->_pFrame->transformPlaneObjsToWorldCVCV(3);
			PRINTSTR("Plane detection.");
			_pGL->timerStop();
			//ingrate current frame into the global volume
			_pVolumeWorld->gpuIntegrateFrameIntoVolumeCVCV(*_pKinect->_pFrame);
			PRINTSTR("Volume integration.");
			_pGL->timerStop();
			if( _pKinect->_pFrame->isMovedwrtReferencInRadiusM( pReferenceKF.get(),M_PI_4/6.,0.15) ){
				_pKinect->_pFrame->gpuTransformToWorldCVCV();
				_nRFIdx++;
				btl::kinect::CKeyFrame::tp_shared_ptr& pCurrentKF = _aShrPtrKFs[_nRFIdx];
				_pKinect->_pFrame->copyTo(&*pCurrentKF);
				_vShrPtrsKF.push_back(&pCurrentKF);
				//save current frame and make it as the reference frame
				std::cout << "new key frame added" << std::flush;
			}//if moving far enough new keyframe will be added
		}//if(dError < 0.5)
		_bCapture = false;
	}//if(_bCapture && _nKFCounter < _nReserved)
	else if( _nKFCounter == _nReserved )	{
		std::cout << "two many key frames to hold" << std::flush;  
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
	
	//_pVolumeWorld->gpuRaycast( *_pKinect->_pFrame, &*_pDisplayFrame ); //get virtual frame
	_pKinect->_pFrame->renderCameraInWorldCVCV(_pGL.get(),_pGL->_bDisplayCamera,.05f,_pGL->_usLevel);
	//_pKinect->_pFrame->renderPlanesInWorld(_pGL.get(),0,_pGL->_usLevel);
	//(**_vShrPtrsKF.rbegin())
	_pKinect->_pFrame->render3DPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel,0,false);
	//_pDisplayFrame->render3DPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel,0,false);
	//_pDisplayFrame->setView(&_pGL->_eimModelViewGL);
	
	// render objects
	ushort usViewIdxTmp = 0;
	for( std::vector< btl::kinect::CKeyFrame::tp_shared_ptr* >::iterator cit = _vShrPtrsKF.begin(); cit!= _vShrPtrsKF.end(); cit++,usViewIdxTmp++ ) {
		if (usViewIdxTmp == _usViewNO)
			(**cit)->renderCameraInWorldCVCV(_pGL.get(),_pGL->_bDisplayCamera,.1f,_pGL->_usLevel);
		else
			(**cit)->renderCameraInWorldCVCV(_pGL.get(),false,.05f,_pGL->_usLevel);
		(**cit)->render3DPtsInWorldCVCV(_pGL.get(), _pGL->_usLevel, _usColorIdx, false );
	}
	//if(_bRenderPlane) _pVirtualFrame->render3DPtsInWorldCVCV(_pGL.get(), _pGL->_usPyrLevel, _usColorIdx, false );
	//_pKinect->_pFrame->renderCameraInWorldCVGL2( _pGL.get(), _pGL->_bDisplayCamera, true, .1f,_pGL->_usPyrLevel );

	if(_pGL->_bRenderReference) {
		_pGL->renderAxisGL();
		_pGL->renderPatternGL(.1f,20.f,20.f);
		_pGL->renderPatternGL(1.f,10.f,10.f);
		_pGL->renderVoxelGL(2.f);
		//_pGL->renderOctTree(0.f,0.f,0.f,3.f,1); this is very slow when the level of octree is deep.
	}

// render second viewport
    glViewport ( _nWidth/2, 0, _nWidth/2, _nHeight );
    glScissor  ( _nWidth/2, 0, _nWidth/2, _nHeight );
    glLoadIdentity();
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	//_pKinect->_pRGBCamera->LoadTexture(*_pKinect->_pFrame->_acvmShrPtrPyrRGBs[_pGL->_usLevel],&_pKinect->_pFrame->_uTexture);
  	  _pKinect->_pRGBCamera->LoadTexture(*_pKinect->_pFrame->_acvmShrPtrPyrRGBs[_pGL->_usLevel],&_pGL->_auTexture[_pGL->_usLevel]);

	//_pKinect->_pRGBCamera->renderCameraInGLLocal(_pKinect->_pFrame->_uTexture, *_pKinect->_pFrame->_acvmShrPtrPyrRGBs[_pGL->_usLevel], .2f );
	_pKinect->_pRGBCamera->renderCameraInGLLocal(_pGL->_auTexture[_pGL->_usLevel], *_pKinect->_pFrame->_acvmShrPtrPyrRGBs[_pGL->_usLevel] );
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
        glutInit ( &argc, argv );
        glutInitDisplayMode ( GLUT_DOUBLE | GLUT_RGB );
        glutInitWindowSize ( 1280, 480 );
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
		_pGL->setCudaDeviceForGLInteroperation();
		_pKinect.reset(new btl::kinect::VideoSourceKinect(_uResolution));
		_pVolumeWorld.reset( new btl::geometry::CModel() );
		init();
		_pVolumeWorld->gpuCreateVBO(_pGL.get());
		_pGL->constructVBOsPBOs();
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


