//display kinect depth in real-time
#define INFO
#define TIMER
#include <GL/glew.h>
#include <iostream>
#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>
#include "Utility.hpp"

//camera calibration from a sequence of images
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <opencv2/gpu/gpu.hpp>
#include <XnCppWrapper.h>
#include <gl/freeglut.h>
#include "Kinect.h"
#include "Camera.h"
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include "EigenUtil.hpp"
#include "GLUtil.h"
#include "KeyFrame.h"
#include <boost/scoped_ptr.hpp>
#include "VideoSourceKinect.hpp"
#include "Model.h"
#include "GLUtil.h"


using namespace Eigen;

class CKinectView;

btl::kinect::VideoSourceKinect::tp_shared_ptr _pKinect;
btl::kinect::SCamera::tp_ptr _pRGBCamera; 
btl::extra::CModel::tp_shared_ptr _pModel;
btl::gl_util::CGLUtil::tp_shared_ptr _pGL;

Matrix4d _mGLMatrix;
double _dNear = 0.01;
double _dFar  = 10.;

unsigned short _nWidth, _nHeight;

bool _bCaptureCurrentFrame = false;
GLuint _uDisk;
GLuint _uNormal;
bool _bRenderNormal = false;
bool _bEnableLighting = false;
double _dDepthFilterThreshold = 0.01;
int _nDensity = 2;
float _fSize = 0.2; // range from 0.05 to 1 by step 0.05
unsigned int _uLevel = 2;
unsigned int _uPyrHeight = 4;
int _nColorIdx = 0;

enum tp_diplay {NORMAL_CLUSTRE, DISTANCE_CLUSTER};
tp_diplay _enumType = NORMAL_CLUSTRE;

void normalKeys ( unsigned char key, int x, int y )
{
	_pGL->normalKeys( key, x, y );
    switch( key )
    {
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
        _fSize += 0.05;// range from 0.05 to 1 by step 0.05
        _fSize = _fSize < 1 ? _fSize: 1;
        glutPostRedisplay();
        PRINT( _fSize );
        break;
    case 'j':
        _fSize -= 0.05;
        _fSize = _fSize > 0.05? _fSize : 0.05;
        glutPostRedisplay();
        PRINT( _fSize );
        break;
	case '9':
		_uLevel = ++_uLevel%_uPyrHeight;
		_pRGBCamera->LoadTexture( *_pKinect->_pFrame->_acvmShrPtrPyrRGBs[_uLevel] );
		PRINT(_uLevel);
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
    return;
}
void specialKeys(int nKey_,int x, int y)
{
	switch( nKey_ )
	{
	case GLUT_KEY_F1: //display camera
		_nColorIdx++;
		PRINT(_nColorIdx);
		break;
	case GLUT_KEY_F2:
		_enumType = NORMAL_CLUSTRE == _enumType? DISTANCE_CLUSTER : NORMAL_CLUSTRE;
		if(NORMAL_CLUSTRE == _enumType) {
			PRINTSTR( "NORMAL_CLUSTRE" );
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

void render3DPts()
{
    double x, y, z;
	if(_uLevel>=_pKinect->_uPyrHeight){
		PRINTSTR("CModel::pointCloud() uLevel_ is more than _uPyrHeight");
		_uLevel = 0;
	} 

	if( _bEnableLighting )
		glEnable(GL_LIGHTING);
	else
		glDisable(GL_LIGHTING);
	
	const float* pPt = (const float*)_pKinect->_pFrame->_acvmShrPtrPyrPts[_uLevel]->data;
	const float* pNl = (const float*)_pKinect->_pFrame->_acvmShrPtrPyrNls[_uLevel]->data;
	const unsigned char* pColor/* = (const unsigned char*)_pVS->_vcvmPyrRGBs[_uPyrHeight-1]->data*/;
	const short* pLabel;
	if(NORMAL_CLUSTRE ==_enumType){
		pLabel = (const short*)_pModel->_acvmShrPtrNormalClusters[_uLevel]->data;
	}
	else if(DISTANCE_CLUSTER ==_enumType){
		pLabel = (const short*)_pModel->_acvmShrPtrDistanceClusters[_uLevel]->data;
	}

	for( int i = 0; i < btl::kinect::__aKinectWxH[_uLevel];i++){
		int nColor = pLabel[i];
		if(nColor<0) 
		{	pNl+=3; pPt+=3; continue; }
		const unsigned char* pColor = btl::utility::__aColors[nColor+_nColorIdx%BTL_NUM_COLOR];
		float dNx = *pNl++;
		float dNy = *pNl++;
		float dNz = *pNl++;
		float x = *pPt++;
		float y = *pPt++;
		float z = *pPt++;
		if( fabs(dNx) + fabs(dNy) + fabs(dNz) > 0.000001 ) 
			_pGL->renderDisk<float>(x,y,z,dNx,dNy,dNz,pColor,_fSize,_bRenderNormal); 
	}
	/*//render point cloud
    const std::vector< Eigen::Vector3d >& vPts=_pModel->_vvPyramidPts[_uLevel] ;
    const std::vector< Eigen::Vector3d >& vNormals = _pModel->_vvPyramidNormals[_uLevel];
    const std::vector<const unsigned char*>& vColors = _pModel->_vvPyramidColors[_uLevel];
    glPushMatrix();
    for (size_t i = 0; i < vPts.size (); ++i)
    {
        if ( 1 != _nDensity && i % _nDensity != 1 ) // skip some points; when 1 == i, all dots wills drawn;
        {
            continue;
        }
		renderDisk(vPts[i], vNormals[i], vColors[i],_uDisk, _uNormal, _bRenderNormal);
    }
    glPopMatrix();*/

    return;
}
void display ( void )
{
	//if(_bCaptureCurrentFrame) 
	{
		_pModel->detectPlaneFromCurrentFrame(_uLevel);
		_bCaptureCurrentFrame = false;
		std::cout << "capture done.\n" << std::flush;
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
    // light position in 3d
    GLfloat light_position[] = { 3.0, 1.0, 1.0, 1.0 };
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);

    // render objects
    _pGL->renderAxisGL();
	_pGL->renderVoxelGL(2.f);
    render3DPts();

    //glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, 640, 480, GL_RGBA, GL_UNSIGNED_BYTE, _pVS->cvRGB().data);
    //_pView->renderCamera( _uTexture, CCalibrateKinect::RGB_CAMERA );

    glViewport (_nWidth/2, 0, _nWidth/2, _nHeight);
    glScissor  (_nWidth/2, 0, _nWidth/2, _nHeight);
    //gluLookAt ( _eivCamera(0), _eivCamera(1), _eivCamera(2),  _eivCentroid(0), _eivCentroid(1), _eivCentroid(2), _eivUp(0), _eivUp(1), _eivUp(2) );
    //glScaled( _dZoom, _dZoom, _dZoom );
    //glRotated ( _dYAngle, 0, 1 ,0 );
    //glRotated ( _dXAngle, 1, 0 ,0 );
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    // render objects
    //renderAxis();
    //render3DPts();
	_pRGBCamera->LoadTexture( *_pKinect->_pFrame->_acvmShrPtrPyrRGBs[_uLevel] );
    _pRGBCamera->renderCamera( *_pKinect->_pFrame->_acvmShrPtrPyrRGBs[_uLevel] );

    glutSwapBuffers();
    glutPostRedisplay();

}

void reshape ( int nWidth_, int nHeight_ )
{
    //cout << "reshape() " << endl;
    _pRGBCamera->setIntrinsics( 1, 0.01, 100 );

    // setup blending
    //glBlendFunc ( GL_SRC_ALPHA, GL_ONE );			// Set The Blending Function For Translucency
    //glColor4f ( 1.0f, 1.0f, 1.0f, 1.0f );

    unsigned short nTemp = nWidth_/8;//make sure that _nWidth is divisible to 4
    _nWidth = nTemp*8;
    _nHeight = nTemp*3;
    glutReshapeWindow( int ( _nWidth ), int ( _nHeight ) );
    return;
}

void init ( )
{
	_pGL->clearColorDepth();
    glDepthFunc  ( GL_LESS );
    glEnable     ( GL_DEPTH_TEST );
    glEnable 	 ( GL_SCISSOR_TEST );
    glEnable     ( GL_CULL_FACE );
    glShadeModel ( GL_FLAT );

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    //_pModel->loadFrame();
	_pModel->detectPlaneFromCurrentFrame(_uLevel);

	_pGL->init();
    
    // light
    GLfloat mat_diffuse[] = { 1.0, 1.0, 1.0, 1.0};
    glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);

    GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
    glLightfv (GL_LIGHT0, GL_DIFFUSE, light_diffuse);

    glEnable(GL_RESCALE_NORMAL);
    glEnable(GL_LIGHT0);
}

int main ( int argc, char** argv )
{
    try{
		_pKinect.reset( new btl::kinect::VideoSourceKinect() );
		_pRGBCamera=_pKinect->_pRGBCamera.get();
		_pModel.reset( new btl::extra::CModel(*_pKinect) );
		_pGL.reset( new btl::gl_util::CGLUtil );

        glutInit ( &argc, argv );
        glutInitDisplayMode ( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH );
        glutInitWindowSize ( 1280, 480 );
        glutCreateWindow ( "CameraPose" );
        init();

        glutKeyboardFunc( normalKeys );
		glutSpecialFunc ( specialKeys );
        glutMouseFunc   ( mouseClick );
        glutMotionFunc  ( mouseMotion );

        glutReshapeFunc ( reshape );
        glutDisplayFunc ( display );
        glutMainLoop();
    }
	/*
    catch ( CError& e )
    {
        if ( std::string const* mi = boost::get_error_info< CErrorInfo > ( e ) )
        {
            std::cerr << "Error Info: " << *mi << std::endl;
        }
    }*/
	catch ( std::runtime_error& e )	{
		PRINTSTR( e.what() );
	}


    return 0;
}


