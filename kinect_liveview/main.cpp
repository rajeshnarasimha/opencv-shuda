//display kinect depth in real-time
//#define INFO

#include <GL/glew.h>
#include <iostream>
#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>
#include "Converters.hpp"
#include <opencv2/gpu/gpu.hpp>
#include "VideoSourceKinect.hpp"
#include "Model.h"
#include "GLUtil.h"
//camera calibration from a sequence of images

using namespace btl; //for "<<" operator
using namespace utility;
using namespace extra;
using namespace videosource;
using namespace Eigen;
//using namespace cv;

class CKinectView;

btl::extra::videosource::VideoSourceKinect _cVS;
btl::extra::videosource::CKinectView _cView(_cVS);
btl::gl_util::CGLUtil::tp_shared_ptr _pGL;
//btl::extra::CModel _cM(_cVS);
Matrix4d _mGLMatrix;
double _dNear = 0.01;
double _dFar  = 10.;

Eigen::Vector3d _eivCentroid(.0, .0, .0 );

unsigned short _nWidth, _nHeight;

bool _bRenderNormal = false;
bool _bEnableLighting = false;
double _dDepthFilterThreshold = 10;
int _nDensity = 2;
float _dSize = 0.2; // range from 0.05 to 1 by step 0.05
unsigned int _uPyrHeight = 1;
unsigned int _uLevel = 0;
VideoSourceKinect::tp_frame _eFrameType = VideoSourceKinect::GPU_PYRAMID_GL;
void normalKeys ( unsigned char key, int x, int y )
{
	_pGL->normalKeys( key, x, y);
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
		_dSize += 0.05;// range from 0.05 to 1 by step 0.05
		_dSize = _dSize < 1 ? _dSize: 1;
		glutPostRedisplay();
		PRINT( _dSize );
		break;
	case 'j':
		_dSize -= 0.05;
		_dSize = _dSize > 0.05? _dSize : 0.05;
		glutPostRedisplay();
		PRINT( _dSize );
		break;
	case '1':
		_eFrameType = VideoSourceKinect::GPU_PYRAMID_GL;
		PRINTSTR(  "VideoSourceKinect::GPU_PYRAMID" );
		break;
	case '2':
		_eFrameType = VideoSourceKinect::CPU_PYRAMID_GL;
		PRINTSTR(  "VideoSourceKinect::CPU_PYRAMID" );
		break;
	case '9':
		_uLevel = ++_uLevel%_uPyrHeight;
		PRINT(_uLevel);
		break;
	case ']':
		_cVS._fSigmaSpace += 1;
		PRINT( _cVS._fSigmaSpace );
		break;
	case '[':
		_cVS._fSigmaSpace -= 1;
		PRINT( _cVS._fSigmaSpace );
		break;
    }

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
void render3DPts()
{
	const unsigned char* pColor;
	double x, y, z;
	if(_uLevel>=_cVS._uPyrHeight)
	{
		PRINTSTR("CModel::pointCloud() uLevel_ is more than _uPyrHeight");
		_uLevel = 0;
	}
	const cv::Mat& cvmPts =*_cVS._acvmShrPtrPyrPts[_uLevel] ;
	const cv::Mat& cvmNls =*_cVS._acvmShrPtrPyrNls[_uLevel] ;
	const cv::Mat& cvmRGBs =_cVS._vcvmPyrRGBs[_uLevel] ;
	const float* pPt = (float*) cvmPts.data;
	const float* pNl = (float*) cvmNls.data;
	const unsigned char* pRGB = (const unsigned char*) cvmRGBs.data;
	glPushMatrix();
	// Generate the data
	for( int i = 0; i < cvmPts.total(); i++)
	{
		if( _bEnableLighting )
			glEnable(GL_LIGHTING);
		else
			glDisable(GL_LIGHTING);
		if( 1 != _nDensity && i % _nDensity != 1 ) // skip some points; when 1 == i, all dots wills drawn;
		{
			pRGB += 3;
			pNl  += 3;
			pPt  += 3;
			continue;
		}
		
		float dNx = *pNl++;
		float dNy = *pNl++;
		float dNz = *pNl++;
		if(fabs(dNz) + fabs(dNy) + fabs(dNx) < 0.00001) 
			int a = 3;
		
		float x = *pPt++;
		float y = *pPt++;
		float z = *pPt++;
		if( fabs(dNx) + fabs(dNy) + fabs(dNz) > 0.000001 ) 
			_pGL->renderDisk<float>(x,y,z,dNx,dNy,dNz,pRGB,_dSize,_bRenderNormal);
		pRGB += 3;
	}
	glPopMatrix();

	return;
} 

void display ( void )
{
	//load data from video source and model
	switch( _eFrameType )
	{
	case VideoSourceKinect::GPU_PYRAMID_GL:
		_cVS.getNextFrame(VideoSourceKinect::GPU_PYRAMID_GL);
		break;
	case VideoSourceKinect::CPU_PYRAMID_GL:
		_cVS.getNextFrame(VideoSourceKinect::CPU_PYRAMID_GL);
		break;
	}
	//set viewport
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
	render3DPts();

	//_cView.renderCamera( _uTexture, CCalibrateKinect::RGB_CAMERA );
	//set viewport 2
	glViewport (_nWidth/2, 0, _nWidth/2, _nHeight);
	glScissor  (_nWidth/2, 0, _nWidth/2, _nHeight);
	glLoadIdentity();
	//gluLookAt ( _eivCamera(0), _eivCamera(1), _eivCamera(2),  _eivCenter(0), _eivCenter(1), _eivCenter(2), _eivUp(0), _eivUp(1), _eivUp(2) );
    //glScaled( _dZoom, _dZoom, _dZoom );    
    //glRotated ( _dYAngle, 0, 1 ,0 );
    //glRotated ( _dXAngle, 1, 0 ,0 );
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// render objects
    _pGL->renderAxisGL();
	//render3DPts();
	_cView.LoadTexture( _cVS._vcvmPyrRGBs[_uLevel] );
	_cView.renderCamera( CCalibrateKinect::RGB_CAMERA, _cVS._vcvmPyrRGBs[_uLevel] );

    glutSwapBuffers();
	glutPostRedisplay();

}
void reshape ( int nWidth_, int nHeight_ ){
	//cout << "reshape() " << endl;
    _cView.setIntrinsics( 1, btl::extra::videosource::CCalibrateKinect::RGB_CAMERA, 0.01, 100 );

    // setup blending 
    glBlendFunc ( GL_SRC_ALPHA, GL_ONE );			// Set The Blending Function For Translucency
    glColor4f ( 1.0f, 1.0f, 1.0f, 0.5 );

	unsigned short nTemp = nWidth_/8;//make sure that _nWidth is divisible to 4
	_nWidth = nTemp*8;
	_nHeight = nTemp*3;
	glutReshapeWindow( int ( _nWidth ), int ( _nHeight ) );
    return;
}
void setPyramid(){
	_eFrameType = VideoSourceKinect::GPU_PYRAMID_GL;
	_uPyrHeight = 4;
	_uLevel = 3;
	_cVS.getNextPyramid(_uPyrHeight);
	_cView.LoadTexture( _cVS._vcvmPyrRGBs[_uLevel] );
}
void init ( ){
	_pGL->clearColorDepth();
	glDepthFunc  ( GL_LESS );
	glEnable     ( GL_DEPTH_TEST );
	glEnable 	 ( GL_SCISSOR_TEST );
	glEnable     ( GL_CULL_FACE );
	glShadeModel ( GL_FLAT );

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	setPyramid();
	_pGL->init();
	// light
	GLfloat mat_diffuse[] = { 1.0, 1.0, 1.0, 1.0};
	glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);

	GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
	glLightfv (GL_LIGHT0, GL_DIFFUSE, light_diffuse);

	glEnable(GL_RESCALE_NORMAL);
	glEnable(GL_LIGHT0);
}

int main ( int argc, char** argv ){
    try {
		_pGL.reset( new btl::gl_util::CGLUtil() );
		glutInit ( &argc, argv );
        glutInitDisplayMode ( GLUT_DOUBLE | GLUT_RGB );
        glutInitWindowSize ( 1280, 480 );
        glutCreateWindow ( "CameraPose" );
		init();
        glutKeyboardFunc( normalKeys );
        glutMouseFunc   ( mouseClick );
        glutMotionFunc  ( mouseMotion );

		glutReshapeFunc ( reshape );
        glutDisplayFunc ( display );
        glutMainLoop();
	}
    catch ( CError& e )  {
        if ( std::string const* mi = boost::get_error_info< CErrorInfo > ( e ) ) {
            std::cerr << "Error Info: " << *mi << std::endl;
        }
    }
	catch ( std::runtime_error& e ){
		PRINTSTR( e.what() );
	}

    return 0;
}
