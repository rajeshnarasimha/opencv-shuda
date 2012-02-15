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
//btl::extra::CModel _cM(_cVS);

Eigen::Vector3d _eivCentroid(.0, .0, .0 );
double _dZoom = 1.;
double _dZoomLast = 1.;
double _dScale = .1;

Matrix4d _mGLMatrix;
double _dNear = 0.01;
double _dFar  = 10.;

double _dXAngle = 0;
double _dYAngle = 0;
double _dXLastAngle = 0;
double _dYLastAngle = 0;
double _dX = 0;
double _dY = 0;
double _dXLast = 0;
double _dYLast = 0;

int  _nXMotion = 0;
int  _nYMotion = 0;
int  _nXLeftDown, _nYLeftDown;
int  _nXRightDown, _nYRightDown;
bool _bLButtonDown;
bool _bRButtonDown;

unsigned short _nWidth, _nHeight;

bool _bCaptureCurrentFrame = false;
bool _bRenderNormal = false;
bool _bEnableLighting = false;
double _dDepthFilterThreshold = 10;
int _nDensity = 2;
float _dSize = 0.2; // range from 0.05 to 1 by step 0.05
unsigned int _uPyrHeight = 1;
unsigned int _uLevel = 0;
VideoSourceKinect::tp_frame _eFrameType = VideoSourceKinect::PYRAMID_BILATERAL_FILTERED_IN_DISPARTY;
void processNormalKeys ( unsigned char key, int x, int y )
{
	switch( key )
	{
	case 27:
		exit ( 0 );
		break;
	case 'g':
		//zoom in
		glDisable( GL_BLEND );
		_dZoom += _dScale;
		glutPostRedisplay();
		PRINT( _dZoom );
		break;
	case 'h':
		//zoom out
		glDisable( GL_BLEND );
		_dZoom -= _dScale;
		glutPostRedisplay();
		PRINT( _dZoom );
		break;
	case 'c':
		//capture current frame the depth map and color
		_bCaptureCurrentFrame = true;
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
		_uLevel = 0;
		_eFrameType = VideoSourceKinect::GPU_RAW;
		PRINTSTR(  "VideoSourceKinect::BILATERAL_FILTERED_IN_DISPARTY" );
		break;
	case '2':
		_uLevel = 0;
		_eFrameType = VideoSourceKinect::CPU_RAW;
		PRINTSTR(  "VideoSourceKinect::BILATERAL_FILTERED_IN_DISPARTY" );
		break;
	case '3':
		_uLevel = 0;
		_eFrameType = VideoSourceKinect::PYRAMID_BILATERAL_FILTERED_IN_DISPARTY;
		PRINTSTR(  "VideoSourceKinect::PYRAMID_BILATERAL_FILTERED_IN_DISPARTY" );
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
	case '0'://reset camera location
		_dXAngle = 0.;
		_dYAngle = 0.;
		_dZoom = 1.;
		break;
    }

    return;
}
void mouseClick ( int nButton_, int nState_, int nX_, int nY_ )
{
	if ( nButton_ == GLUT_LEFT_BUTTON )
	{
		if ( nState_ == GLUT_DOWN )
		{
			_nXMotion = _nYMotion = 0;
			_nXLeftDown    = nX_;
			_nYLeftDown    = nY_;

			_bLButtonDown = true;
		}
		else if( nState_ == GLUT_UP )// button up
		{
			_dXLastAngle = _dXAngle;
			_dYLastAngle = _dYAngle;
			_bLButtonDown = false;
		}
		glutPostRedisplay();
	}
	else if ( GLUT_RIGHT_BUTTON )
	{
		if ( nState_ == GLUT_DOWN )
		{
			_nXMotion = _nYMotion = 0;
			_nXRightDown  = nX_;
			_nYRightDown  = nY_;
			_dZoomLast    = _dZoom;
			_bRButtonDown = true;
		}
		else if( nState_ == GLUT_UP )
		{
			_dXLast = _dX;
			_dYLast = _dY;
			_bRButtonDown = false;
		}
		glutPostRedisplay();
	}

	return;
}
void mouseMotion ( int nX_, int nY_ )
{
	if ( _bLButtonDown == true )
	{
		glDisable     ( GL_BLEND );
		_nXMotion = nX_ - _nXLeftDown;
		_nYMotion = nY_ - _nYLeftDown;
		_dXAngle  = _dXLastAngle + _nXMotion;
		_dYAngle  = _dYLastAngle + _nYMotion;
	}
	else if ( _bRButtonDown == true )
	{
		glDisable     ( GL_BLEND );
		_nXMotion = nX_ - _nXRightDown;
		_nYMotion = nY_ - _nYRightDown;
		_dX  = _dXLast + _nXMotion;
		_dY  = _dYLast + _nYMotion;
		_dZoom = _dZoomLast + (_nXMotion + _nYMotion)/200.;
	}

	glutPostRedisplay();
}
void renderAxis()
{
    glPushMatrix();
    float fAxisLength = 1.f;
    float fLengthWidth = 1;

    glLineWidth( fLengthWidth );
    // x axis
    glColor3f ( 1., .0, .0 );
    glBegin ( GL_LINES );

    glVertex3d ( .0, .0, .0 );
    Vector3d vXAxis; vXAxis << fAxisLength, .0, .0;
    glVertex3d ( vXAxis(0), vXAxis(1), vXAxis(2) );
    glEnd();
    // y axis
    glColor3f ( .0, 1., .0 );
    glBegin ( GL_LINES );
    glVertex3d ( .0, .0, .0 );
    Vector3d vYAxis; vYAxis << .0, fAxisLength, .0;
    glVertex3d ( vYAxis(0), vYAxis(1), vYAxis(2) );
    glEnd();
    // z axis
    glColor3f ( .0, .0, 1. );
    glBegin ( GL_LINES );
    glVertex3d ( .0, .0, .0 );
    Vector3d vZAxis; vZAxis << .0, .0, fAxisLength;
    glVertex3d ( vZAxis(0), vZAxis(1), vZAxis(2) );
    glEnd();
    glPopMatrix();
}
template< typename T >
void renderDisk(const T& x, const T& y, const T& z, const T& dNx, const T& dNy, const T& dNz, const unsigned char* pColor_, const T& dSize_, GLuint uDisk_, GLuint uNormal_, bool bRenderNormal_ )
{
	glColor3ubv( pColor_ );

	glPushMatrix();
	glTranslatef( x, y, z );

	if( fabs(dNx) + fabs(dNy) + fabs(dNz) < 0.00001 ) // normal is not computed
	{
		//PRINT( dNz );
		return;
	}

	T dA = atan2(dNx,dNz);
	T dxz= sqrt( dNx*dNx + dNz*dNz );
	T dB = atan2(dNy,dxz);

	glRotatef(-dB*180 / M_PI,1,0,0 );
	glRotatef( dA*180 / M_PI,0,1,0 );
	T dR = -z/0.5;
	glScalef( dR*dSize_, dR*dSize_, dR*dSize_ );
	glCallList(uDisk_);
	if( bRenderNormal_ )
	{
		glCallList(uNormal_);
	}
	glPopMatrix();
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
	int i = 0;
	for( int r = 0; r < cvmPts.rows; r++ )
	for( int c = 0; c < cvmPts.cols; c++ , i++)
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
			_cView.renderDisk<float>(x,y,z,dNx,dNy,dNz,pRGB,_dSize,_bRenderNormal);
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
	case VideoSourceKinect::CPU_RAW:
		_cVS.getNextFrame(VideoSourceKinect::CPU_RAW);
		break;
	case VideoSourceKinect::GPU_RAW:
		_cVS.getNextFrame(VideoSourceKinect::GPU_RAW);
		break;
	case VideoSourceKinect::PYRAMID_BILATERAL_FILTERED_IN_DISPARTY:
		_cVS.getNextPyramid(_uPyrHeight);
		break;
	}
	_cVS.centroidGL(&_eivCentroid);
	//set viewport
    glMatrixMode ( GL_MODELVIEW );
	glViewport (0, 0, _nWidth/2, _nHeight);
	glScissor  (0, 0, _nWidth/2, _nHeight);
	// after set the intrinsics and extrinsics
    // load the matrix to set camera pose
	glLoadIdentity();
	//glLoadMatrixd( _mGLMatrix.data() );
	glTranslated( _eivCentroid(0), _eivCentroid(1), _eivCentroid(2) ); // 5. translate back to the original camera pose
	_dZoom = _dZoom < 0.1? 0.1: _dZoom;
	_dZoom = _dZoom > 10? 10: _dZoom;
	glScaled( _dZoom, _dZoom, _dZoom );                          // 4. zoom in/out
	glRotated ( _dXAngle, 0, 1 ,0 );                             // 3. rotate horizontally
	glRotated ( _dYAngle, 1, 0 ,0 );                             // 2. rotate vertically
	glTranslated( -_eivCentroid(0),-_eivCentroid(1),-_eivCentroid(2)); // 1. translate the world origin to align with object centroid
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// light position in 3d
	GLfloat light_position[] = { 3.0, 1.0, 1.0, 1.0 };
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	
    // render objects
    renderAxis();
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
    renderAxis();
	//render3DPts();
	_cView.LoadTexture( _cVS._vcvmPyrRGBs[_uLevel] );
	_cView.renderCamera( CCalibrateKinect::RGB_CAMERA, _cVS._vcvmPyrRGBs[_uLevel] );

    glutSwapBuffers();
	glutPostRedisplay();

}

void reshape ( int nWidth_, int nHeight_ )
{
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
void setPyramid()
{
	_eFrameType = VideoSourceKinect::PYRAMID_BILATERAL_FILTERED_IN_DISPARTY;
	_uPyrHeight = 4;
	_uLevel = 3;
	_cVS.getNextPyramid(_uPyrHeight);
	_cView.LoadTexture( _cVS._vcvmPyrRGBs[_uLevel] );
}
void init ( )
{
	glClearColor ( 0.1f,0.1f,0.4f,1.0f );
	glClearDepth ( 1.0 );
	glDepthFunc  ( GL_LESS );
	glEnable     ( GL_DEPTH_TEST );
	glEnable 	 ( GL_SCISSOR_TEST );
	glEnable     ( GL_CULL_FACE );
	glShadeModel ( GL_FLAT );

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	setPyramid();
	_cView.init();
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
    try
    {
		glutInit ( &argc, argv );
        glutInitDisplayMode ( GLUT_DOUBLE | GLUT_RGB );
        glutInitWindowSize ( 1280, 480 );
        glutCreateWindow ( "CameraPose" );
		init();
        glutKeyboardFunc( processNormalKeys );
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
