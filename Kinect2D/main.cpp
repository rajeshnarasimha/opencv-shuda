//display kinect depth in real-time
#include <iostream>
#include <string>
#include <vector>
#include <Converters.hpp>
#include <VideoSourceKinect.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <algorithm>
#include <utility>
#include "keyframe.hpp"
#include <boost/lexical_cast.hpp>
//camera calibration from a sequence of images

using namespace btl; //for "<<" operator
using namespace utility;
using namespace extra;
using namespace videosource;
using namespace Eigen;
using namespace cv;

//class CKinectView;
//class KeyPoint;

btl::extra::videosource::VideoSourceKinect _cVS;
btl::extra::videosource::CKinectView _cView ( _cVS );

Eigen::Vector3d _eivCamera ( 0.0, -1.0, -1.0 );
Eigen::Vector3d _eivCenter ( .0, .0, .0 );
Eigen::Vector3d _eivUp ( .0, -1.0, 0.0 );
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
GLuint _uTextureFirst;
GLuint _uTextureSecond;

SKeyFrame _asKFs[50];
int _nKFCounter = 1; //key frame counter
int _nRFCounter = 0; //reference frame counter
vector< SKeyFrame* > _vKFPtrs;
vector< int > _vRFIdx;


bool _bContinuous = true;
bool _bPrevStatus = true;
bool _bDisplayCamera = true;
bool _bRenderReference = true;

bool _bCapture = false;

int _nN = 1;
int _nView = 0;

void init();

void resetModelViewParameters()
{
    _eivCamera = Vector3d(0., 0., 0.);
    _eivCenter = Vector3d(0., 0., -1.);
    _eivUp     = Vector3d(0., 1., 0.);

    _dXAngle = _dYAngle = 0;
    _dX = _dY = 0;
    _dZoom = 1;
}

void specialKeys( int key, int x, int y )
{
	switch ( key )
	{
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

void normalKeys ( unsigned char key, int x, int y )
{
    switch ( key )
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
    case 'r':
        //reset
		_nKFCounter=1;
		_nRFCounter=0;
		_vKFPtrs.clear();
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
		if(_nKFCounter >0 )
		{
			if( _nRFCounter ==  _nKFCounter )
				_nRFCounter--;
			_nKFCounter--;
			_vKFPtrs.pop_back();
		}
		glutPostRedisplay();
		break;
	case 'v':
		//use current keyframe as a reference
		if( _nRFCounter <_nKFCounter )
		{
			_nRFCounter = _nKFCounter-1;
			_vRFIdx.push_back( _nRFCounter );
			SKeyFrame& s1stKF = _asKFs[_nRFCounter];
			s1stKF._bIsReferenceFrame = true;
    		//construct KD tree
    		s1stKF.constructKDTree();
			glutPostRedisplay();
		}
		break;
	case ',':
		_mGLMatrix = _vKFPtrs[ _nView ]->setView();
		resetModelViewParameters();
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

void renderPattern()
{
    glPushMatrix();
    const std::vector<cv::Point3f>& vPts = _cVS.pattern();
    glPointSize ( 3 );
    glColor3d ( .0 , .8 , .8 );
    glBegin ( GL_POINTS );

    for ( std::vector<cv::Point3f>::const_iterator constItr = vPts.begin(); constItr < vPts.end() ; ++ constItr )
    {
        glVertex3f ( constItr->x,  constItr->z, -constItr->y );
        glVertex3f ( constItr->x,  constItr->z,  constItr->y );
        glVertex3f ( -constItr->x,  constItr->z, -constItr->y );
        glVertex3f ( -constItr->x,  constItr->z,  constItr->y );
    }

    glEnd();
    glPopMatrix();
    return;
}

void renderAxis()
{
    glPushMatrix();
    float fAxisLength = 1.f;
    float fLengthWidth = 1;

    glLineWidth ( fLengthWidth );
    // x axis
    glColor3f ( 1., .0, .0 );
    glBegin ( GL_LINES );
    glVertex3d ( .0, .0, .0 );
    Vector3d vXAxis;
    vXAxis << fAxisLength, .0, .0;
    glVertex3d ( vXAxis ( 0 ), vXAxis ( 1 ), vXAxis ( 2 ) );
    glEnd();
    // y axis
    glColor3f ( .0, 1., .0 );
    glBegin ( GL_LINES );
    glVertex3d ( .0, .0, .0 );
    Vector3d vYAxis;
    vYAxis << .0, fAxisLength, .0;
    glVertex3d ( vYAxis ( 0 ), vYAxis ( 1 ), vYAxis ( 2 ) );
    glEnd();
    // z axis
    glColor3f ( .0, .0, 1. );
    glBegin ( GL_LINES );
    glVertex3d ( .0, .0, .0 );
    Vector3d vZAxis;
    vZAxis << .0, .0, fAxisLength;
    glVertex3d ( vZAxis ( 0 ), vZAxis ( 1 ), vZAxis ( 2 ) );
    glEnd();
    glPopMatrix();
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
        _eivCamera ( 0 ) = _dX / 50.;
        _eivCamera ( 1 ) = _dY / 50.;
    }

    glutPostRedisplay();
}

void init ( )
{
    _mGLMatrix.setIdentity();
    glClearColor ( 0.0, 0.0, 0.0, 1.0 );
    glClearDepth ( 1.0 );
    glDepthFunc  ( GL_LESS );
    glEnable     ( GL_DEPTH_TEST );
    glEnable 	 ( GL_SCISSOR_TEST );
    glEnable     ( GL_BLEND );
    glBlendFunc  ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glShadeModel ( GL_FLAT );
    glEnable ( GL_LINE_SMOOTH );
    glEnable ( GL_POINT_SMOOTH );

    glPixelStorei ( GL_UNPACK_ALIGNMENT, 1 );

// store a frame and detect feature points for tracking.
    _cVS.getNextFrame();
    // load as texture
    _uTextureFirst = _cView.LoadTexture ( _cVS.cvRGB() );
	SKeyFrame& s1stKF = _asKFs[0];
	_vRFIdx.push_back(0);
    // assign the rgb and depth to the current frame.
    s1stKF.assign ( _cVS.cvRGB(), _cVS.registeredDepth() );
    //corner detection and ranking ( first frame )
    s1stKF.detectCorners();
    //construct KD tree
    s1stKF.constructKDTree();
	s1stKF._bIsReferenceFrame = true;
// ( second frame )
    _uTextureSecond = _cView.LoadTexture ( _cVS.cvRGB() );
    //s1stKF.save2XML ( "0" );
	
	_vKFPtrs.push_back( &s1stKF );
    return;
}

void display ( void )
{
// update frame
    _cVS.getNextFrame();
// ( second frame )
    // assign the rgb and depth to the current frame.
	SKeyFrame& sCurrentKF = _asKFs[_nKFCounter];
    sCurrentKF.assign ( _cVS.cvRGB(), _cVS.registeredDepth() );

    if ( _bCapture && _nKFCounter < 50 )
    {
		SKeyFrame& s1stKF = _asKFs[_nRFCounter];
        _bCapture = false;
        // detect corners
        sCurrentKF.detectCorners();

        sCurrentKF.detectCorrespondences ( s1stKF );

        sCurrentKF.calcRT ( s1stKF );

 		sCurrentKF.applyRelativePose( s1stKF );

		_vKFPtrs.push_back( &sCurrentKF );

		_nKFCounter++;
		cout << "new key frame added" << flush;
    }
	else if( _nKFCounter > 49 )
	{
		cout << "two many key frames to hold" << flush;  
	}

// render first viewport
    glMatrixMode ( GL_MODELVIEW );
    glViewport ( 0, 0, _nWidth / 2, _nHeight );
    glScissor  ( 0, 0, _nWidth / 2, _nHeight );
    // after set the intrinsics and extrinsics
    // load the matrix to set camera pose
    glLoadIdentity();
	glLoadMatrixd( _mGLMatrix.data() );

    gluLookAt ( _eivCamera ( 0 ), _eivCamera ( 1 ), _eivCamera ( 2 ),  _eivCenter ( 0 ), _eivCenter ( 1 ), _eivCenter ( 2 ), _eivUp ( 0 ), _eivUp ( 1 ), _eivUp ( 2 ) );
    glScaled ( _dZoom, _dZoom, _dZoom );
    glRotated ( _dYAngle, 0, 1 , 0 );
    glRotated ( _dXAngle, 1, 0 , 0 );

    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // render objects
    //glBindTexture(GL_TEXTURE_2D, _uTexture);
    //place the first camera in the world
    //place the second camera in the world
	//sCurrentKF.renderCamera( _cView, _uTextureFirst );


	for( vector< SKeyFrame* >::iterator cit = _vKFPtrs.begin(); cit!= _vKFPtrs.end(); cit++ )
	{
		(*cit)->renderCamera( _cView, _uTextureFirst,_bDisplayCamera );
	}

if(_bRenderReference)
{
	renderPattern();
    renderAxis();
}

// render second viewport
    glViewport ( _nWidth / 2, 0, _nWidth / 2, _nHeight );
    glScissor  ( _nWidth / 2, 0, _nWidth / 2, _nHeight );
    glLoadIdentity();
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glBindTexture ( GL_TEXTURE_2D, _uTextureSecond );
    glTexSubImage2D ( GL_TEXTURE_2D, 0, 0, 0, 640, 480, GL_RGB, GL_UNSIGNED_BYTE, _cVS.cvRGB().data );
    _cView.renderCamera ( _uTextureSecond, CCalibrateKinect::RGB_CAMERA, CKinectView::ALL_CAMERA, .2 );

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

    if ( _bContinuous )
    {
        glutPostRedisplay();
    }

}

void reshape ( int nWidth_, int nHeight_ )
{
    //cout << "reshape() " << endl;
    _cView.setIntrinsics ( 1, btl::extra::videosource::CCalibrateKinect::RGB_CAMERA, 0.01, 100 );

    // setup blending
    glBlendFunc ( GL_SRC_ALPHA, GL_ONE );			// Set The Blending Function For Translucency
    glColor4f ( 1.0f, 1.0f, 1.0f, 0.5 );

    unsigned short nTemp = nWidth_ / 8; //make sure that _nWidth is divisible to 4
    _nWidth = nTemp * 8;
    _nHeight = nTemp * 3;
    glutReshapeWindow ( int ( _nWidth ), int ( _nHeight ) );
    return;
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
        glutKeyboardFunc ( normalKeys );
		glutSpecialFunc ( specialKeys );
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


