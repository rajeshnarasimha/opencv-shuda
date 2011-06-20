//display kinect depth in real-time
#include <iostream>
#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <btl/Utility/Converters.hpp>
#include <btl/extra/VideoSource/VideoSourceKinect.hpp>
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

Eigen::Vector3d _eivCamera ( 1.0, 1.0, 1.0 );
Eigen::Vector3d _eivCenter ( .0, .0, .0 );
Eigen::Vector3d _eivUp ( .0, 1.0, 0.0 );
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

SKeyFrame _s1stKF;
SKeyFrame _sCurrentKF;
SKeyFrame _sPreviousKF;
SKeyFrame _sMinus2KF;

Eigen::Matrix3d _mRAccu; //Accumulated Rotation
Eigen::Vector3d _vTAccu; //Accumulated Translation	
Eigen::Matrix3d _mRx;

bool _bContinuous = true;
bool _bPrevStatus = true;

bool _bCapture = false;

int _nN = 1;

void processNormalKeys ( unsigned char key, int x, int y )
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
		_mRAccu.setIdentity();
		_vTAccu.setZero();
		_bPrevStatus = true;
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
		_bCapture = true;
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
    glPointSize( 3 );
    glColor3d( .0 , .8 , .8 );
    glBegin ( GL_POINTS );
    for (std::vector<cv::Point3f>::const_iterator constItr = vPts.begin(); constItr < vPts.end() ; ++ constItr)
    {
        glVertex3f( constItr->x, constItr->y, constItr->z );
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
	_mRx <<  1., 0., 0., // rotate about the x-axis for 180 degree.
		    0.,-1., 0.,
			0., 0.,-1.;
			
	_mRAccu.setIdentity();
	_vTAccu.setZero();

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
    glEnable ( GL_BLEND );
    glEnable ( GL_POINT_SMOOTH );
    glBlendFunc ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );


    glPixelStorei ( GL_UNPACK_ALIGNMENT, 1 );

// store a frame and detect feature points for tracking.
    _cVS.getNextFrame();
    // load as texture
    _uTextureFirst = _cView.LoadTexture ( _cVS.cvRGB() );
    // assign the rgb and depth to the current frame.
    _s1stKF.assign ( _cVS.cvRGB(), _cVS.registeredDepth() );
    //corner detection and ranking ( first frame )
    _s1stKF.detectCorners();
	//construct KD tree
	_s1stKF.constructKDTree();
// ( second frame )
    _uTextureSecond = _cView.LoadTexture ( _cVS.cvRGB() );
	_s1stKF.save2XML("0");
    return;
}

void display ( void )
{
// update frame
    _cVS.getNextFrame();
// ( second frame )
    // assign the rgb and depth to the current frame.
    _sCurrentKF.assign ( _cVS.cvRGB(), _cVS.registeredDepth() );

if( _bCapture )
{
	_bCapture = false;
    // detect corners
    _sCurrentKF.detectCorners();

	vector< int > vPtPairs;
	_s1stKF.match( _sCurrentKF, &vPtPairs );

	std::string strNum = boost::lexical_cast<string> ( _nN++ );
	_sCurrentKF.save2XML(strNum.c_str());

	PRINT( vPtPairs.size()/2 );
}
	//_sCurrentKF.detectCorners();
    // get optical flow lines
    vector<unsigned char> vStatus;    
	Eigen::Matrix3d eimR2;
    Eigen::Vector3d eivT2;

// render first viewport
    glMatrixMode ( GL_MODELVIEW );
    glViewport ( 0, 0, _nWidth / 2, _nHeight );
    glScissor  ( 0, 0, _nWidth / 2, _nHeight );
    // after set the intrinsics and extrinsics
    // load the matrix to set camera pose
    glLoadIdentity();
    gluLookAt ( _eivCamera ( 0 ), _eivCamera ( 1 ), _eivCamera ( 2 ),  _eivCenter ( 0 ), _eivCenter ( 1 ), _eivCenter ( 2 ), _eivUp ( 0 ), _eivUp ( 1 ), _eivUp ( 2 ) );
    glScaled ( _dZoom, _dZoom, _dZoom );
    glRotated ( _dYAngle, 0, 1 , 0 );
    glRotated ( _dXAngle, 1, 0 , 0 );

    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // render objects
    //glBindTexture(GL_TEXTURE_2D, _uTexture);
    _cView.renderCamera ( _uTextureFirst, CCalibrateKinect::RGB_CAMERA, CKinectView::ALL_CAMERA, .2 );
    glPointSize ( 3 );
    glColor3d ( 1, 0, 0 );
	//place the second camera
	//if( _sCurrentKF._eivT.norm() < 0.0001);
//		 _sCurrentKF._eivT.setZero();
	_mRAccu *= _sCurrentKF._eimR;
	_vTAccu = _sCurrentKF._eimR*_vTAccu + _sCurrentKF._eivT;
//	PRINT( _mRAccu );
//	PRINT( _vTAccu );
	Eigen::Matrix3d mR = _mRx*_mRAccu;
	Eigen::Vector3d vT = _mRx*_vTAccu;
	Eigen::Matrix4d mGLM = setOpenGLModelViewMatrix( mR, vT );

	glPushMatrix();
    mGLM = mGLM.inverse().eval();
    glMultMatrixd( mGLM.data() );
	_cView.renderCamera( _uTextureFirst, CCalibrateKinect::RGB_CAMERA, CKinectView::ALL_CAMERA, .2 );
	glPopMatrix();
/*
    glBegin ( GL_POINTS );
    for ( vector< Point2f >::const_iterator cit = _s1stKF._vCorners.begin(); cit != _s1stKF._vCorners.end(); cit++ )
    {
        _cView.renderOnImage ( cit->x, cit->y );
    }
    glEnd();
*/	
	renderPattern();
    renderAxis();

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

    //corners at the second frame
    glPointSize ( 1 );
    glColor3d ( 0, 1, 0 );
    glBegin ( GL_POINTS );
    for ( vector< Point2f >::const_iterator cit = _sCurrentKF._vCorners.begin(); cit != _sCurrentKF._vCorners.end(); cit++ )
    {
        _cView.renderOnImage ( cit->x, cit->y );
    }
    glEnd();
	*/

	SKeyFrame* pKF;
	if( _bPrevStatus )
		pKF = &_sPreviousKF;
	else
	{
		cout << "render k-2 frame ";
  	    pKF = &_sMinus2KF;
	}

	if ( _sCurrentKF._vCorners.size() != pKF->_vCorners.size() )
    {
        cout << "error";
    }

    glBegin ( GL_LINES );
    for ( unsigned int i = 0; i < _sCurrentKF._vCorners.size(); i++ )
    {
        glColor3d ( 0, 1, 1 );
        _cView.renderOnImage ( _sCurrentKF._vCorners[i].x, _sCurrentKF._vCorners[i].y );
        //glColor3d ( 1, 0, 0 );
	    _cView.renderOnImage ( pKF->_vCorners[i].x, pKF->_vCorners[i].y );
	
    }
    glEnd();
    
	glutSwapBuffers();
	if( _bContinuous )
	    glutPostRedisplay();

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
        glutKeyboardFunc ( processNormalKeys );
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


