//display kinect depth in real-time
#include <iostream>
#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>
#include "Converters.hpp"
#include "VideoSourceKinect.hpp"
//camera calibration from a sequence of images

using namespace btl; //for "<<" operator
using namespace utility;
using namespace extra;
using namespace videosource;
using namespace Eigen;
using namespace cv;

class CKinectView;

btl::extra::videosource::VideoSourceKinect _cVS;
btl::extra::videosource::CKinectView _cView(_cVS);

Eigen::Vector3d _eivCamera(.0, .0, .0 );
Eigen::Vector3d _eivCenter(.0, .0,-1.0 );
Eigen::Vector3d _eivUp(.0, 1.0, 0.0);
Matrix4d _mGLMatrix;
double _dNear = 0.01;
double _dFar  = 10.;

double _dXAngle = 0;
double _dYAngle = 0;
double _dXLastAngle = 0;
double _dYLastAngle = 0;

double _dZoom = 1.;

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
GLuint _uTexture;


void processNormalKeys ( unsigned char key, int x, int y )
{
	switch( key )
	{
		case 27:
        	exit ( 0 );
			break;
    	case 'i':
        	//zoom in
			glDisable     ( GL_BLEND );
	        _dZoom += 0.2;
    	    glutPostRedisplay();
			break;
		case 'k':
        	//zoom out
			glDisable     ( GL_BLEND );
        	_dZoom -= 0.2;
        	glutPostRedisplay();
			break;
		case '<':
		    _dYAngle += 1.0;
			glutPostRedisplay();
			break;
		case '>':
			_dYAngle -= 1.0;
			glutPostRedisplay();
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
        else 
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
            _nXRightDown    = nX_;
            _nYRightDown    = nY_;
            
            _bRButtonDown = true;
        }
        else 
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
       _eivCamera(0) = _dX/50.;
       _eivCamera(1) = _dY/50.;
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

void render3DPts()
{
 	const unsigned char* pColor = _cVS.cvRGB().data;
	const double* pDepth = _cVS.registeredDepth();

	glPushMatrix();
    glPointSize ( 1. );
	glBegin ( GL_POINTS );
	for (int i = 0; i< 307200; i++)
	{
		double dX = *pDepth++;
		double dY = *pDepth++;
		double dZ = *pDepth++;
		if( abs(dZ) > 0.0000001 )
		{
			glColor3ubv( pColor );
			glVertex3d ( dX, -dY, -dZ );
		}
		pColor +=3;
	}
    glEnd();
	glPopMatrix();
} 

void display ( void )
{
    _cVS.getNextFrame();
    glMatrixMode ( GL_MODELVIEW );
	glViewport (0, 0, _nWidth/2, _nHeight);
	glScissor  (0, 0, _nWidth/2, _nHeight);
	// after set the intrinsics and extrinsics
    // load the matrix to set camera pose
    glLoadIdentity();
    //glLoadMatrixd( _mGLMatrix.data() );
    // navigating the world
    gluLookAt ( _eivCamera(0), _eivCamera(1), _eivCamera(2),  _eivCenter(0), _eivCenter(1), _eivCenter(2), _eivUp(0), _eivUp(1), _eivUp(2) );
    glScaled( _dZoom, _dZoom, _dZoom );    
    glRotated ( _dYAngle, 0, 1 ,0 );
    glRotated ( _dXAngle, 1, 0 ,0 );
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // render objects
    renderAxis();
	render3DPts();
    glBindTexture(GL_TEXTURE_2D, _uTexture);
    //glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, 640, 480, GL_RGBA, GL_UNSIGNED_BYTE, _cVS.cvRGB().data);

	//_cView.renderCamera( _uTexture, CCalibrateKinect::RGB_CAMERA );

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
	render3DPts();
	glBindTexture(GL_TEXTURE_2D, _uTexture);
    glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, 640, 480, GL_RGB, GL_UNSIGNED_BYTE, _cVS.cvRGB().data);

	_cView.renderCamera( _uTexture, CCalibrateKinect::RGB_CAMERA );

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

void init ( )
{
    _mGLMatrix = Matrix4d::Identity();
	glClearColor ( 0.0, 0.0, 0.0, 1.0 );
    glClearDepth ( 1.0 );
    glDepthFunc  ( GL_LESS );
    glEnable     ( GL_DEPTH_TEST );
	glEnable 	 ( GL_SCISSOR_TEST ); 
    glEnable     ( GL_BLEND );
    glBlendFunc  ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glShadeModel ( GL_FLAT );

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	_cVS.getNextFrame();
	_uTexture = _cView.LoadTexture( _cVS.cvRGB() );

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
    catch ( CError& e )
    {
        if ( string const* mi = boost::get_error_info< CErrorInfo > ( e ) )
        {
            std::cerr << "Error Info: " << *mi << std::endl;
        }
    }

    return 0;
}

/*
// display the content of depth and rgb
int main ( int argc, char** argv )
{
    try
    {
		btl::extra::videosource::VideoSourceKinect cVS;
		Mat cvImage( 480,  640, CV_8UC1 );
		int n = 0;

		cv::namedWindow ( "rgb", 1 );
		cv::namedWindow ( "ir", 2 );
    	while ( true )
    	{
			cVS.getNextFrame();
	    	cv::imshow ( "rgb", cVS.cvRGB() );
			for( int r = 0; r< cVS.cvDepth().rows; r++ )
				for( int c = 0; c< cVS.cvDepth().cols; c++ )
				{
					double dDepth = cVS.cvDepth().at< unsigned short > (r, c);
					dDepth = dDepth > 2500? 2500: dDepth;
					cvImage.at<unsigned char>(r,c) = (unsigned char)(dDepth/2500.*256); 
					//PRINT( int(cvImage.at<unsigned char>(r,c)) );
				}
			cv::imshow ( "ir",  cvImage );
			int key = cvWaitKey ( 10 );
			PRINT( key );
			if ( key == 1048675 )//c
       		{
				cout << "c pressed... " << endl;
				//capture depth map	
           		std::string strNum = boost::lexical_cast<string> ( n );
           		std::string strIRFileName = "ir" + strNum + ".bmp";
           		cv::imwrite ( strIRFileName.c_str(), cvImage );
           		n++;
       		}

    	    if ( key == 1048689 ) //q
        	{
            	break;
        	}
    	}

        return 0;
    }
    catch ( CError& e )
    {
        if ( string const* mi = boost::get_error_info< CErrorInfo > ( e ) )
        {
            std::cerr << "Error Info: " << *mi << std::endl;
        }
    }

    return 0;
}
*/


