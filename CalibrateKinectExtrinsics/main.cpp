#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "calibratekinectextrinsics.hpp"
#include <btl/Utility/Converters.hpp>
#include <GL/freeglut.h>
//camera calibration from a sequence of images
// camera type
/*
#ifndef RGB_CAMERA
	#define IR_CAMERA 0
	#define DEPTH_CAMERA IR_CAMERA //depth camera is ir camera
	#define RGB_CAMERA 1
#endif
*/

#define   NONE_DEPTHVIEW 0
#define SINGLE_DEPTHVIEW 1
#define    ALL_DEPTHVIEW 2
#define   NORGB_CAMERA   0
#define NOFRAME_CAMERA   1
#define     ALL_CAMERA   3

using namespace btl; //for "<<" operator
using namespace utility;
using namespace Eigen;
using namespace cv;

shuda::CCalibrateKinectExtrinsics cKinectCalibExt;
Eigen::Vector3d _eivCamera(1.,1.,1.);
Eigen::Vector3d _eivCenter(.0, .0,.0 );
Eigen::Vector3d _eivUp(.0, 1.0, 0.0);
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


double _dZoom = 1.;
unsigned int _nScaleViewport = 1;

unsigned int _uDepthView = SINGLE_DEPTHVIEW;
unsigned int _uCamera    = ALL_CAMERA; 

unsigned int _uNthView = 0;
int  _nXMotion = 0;
int  _nYMotion = 0;
int  _nXLeftDown, _nYLeftDown;
int  _nXRightDown, _nYRightDown;
bool _bLButtonDown;
bool _bRButtonDown;

int _nCameraType = CCalibrateKinect::RGB_CAMERA; //camera type
vector< GLuint > _vuTexture[2]; //[0] for ir [1] for rgb

vector< GLuint > vLists;

CKinectView cView( cKinectCalibExt );

void create3DPtsDisplayLists()
{
	for (unsigned int i = 0; i< cKinectCalibExt.views(); i++ )
	{
		GLuint uList = glGenLists(1);
		glNewList(uList, GL_COMPILE);
		vLists.push_back( uList );

		glPushMatrix();
	    Eigen::Vector3d vT = cKinectCalibExt.eiVecT( i ,CCalibrateKinect::IR_CAMERA );
    	Eigen::Matrix3d mR = cKinectCalibExt.eiMatR( i ,CCalibrateKinect::IR_CAMERA );
    
		//cout << "placeCameraInWorldCoordinate() after setup the RT\n";
    	Eigen::Matrix4d mGLM = setOpenGLModelViewMatrix( mR, vT );
    	mGLM = mGLM.inverse().eval();
    	glMultMatrixd( mGLM.data() );

		const vector< Vector4d >& vPts = cKinectCalibExt.points(i);
		const vector< unsigned char* >& vColors =  cKinectCalibExt.colors(i);
	 	glBegin ( GL_POINTS );
    	glPointSize ( 1. );
	    for (unsigned int i = 0; i < vPts.size(); i++ )
    	{
			if( vColors[i] ) glColor3ubv( vColors[i] );
			glVertex3d ( vPts[i](0), -vPts[i](1), -vPts[i](2) );
		}
    	glEnd();

		glPopMatrix();
		glEndList();

	}
	return;
}

void init ( )
{
    _mGLMatrix = Matrix4d::Identity();

    for( unsigned int n = 0; n < cKinectCalibExt.views(); n++ )
    {
        //load ir texture
        _vuTexture[CCalibrateKinect::IR_CAMERA].push_back( cView.LoadTexture( cKinectCalibExt.undistortedDepth( n ) ) ); 
        //load rgb texture
        _vuTexture[CCalibrateKinect::RGB_CAMERA].push_back( cView.LoadTexture( cKinectCalibExt.undistortedImg( n ) ) ); 
    }

    glClearColor ( 0.0, 0.0, 0.0, 0.0 );
    glClearDepth ( 1.0 );
    glDepthFunc  ( GL_LESS );
    glEnable     ( GL_DEPTH_TEST );
    //glEnable     ( GL_BLEND );
    glBlendFunc  ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glShadeModel ( GL_FLAT );
	create3DPtsDisplayLists();
}
void render3DPts(unsigned int uNthView_ )
{
	switch ( _uDepthView )
	{
		case NONE_DEPTHVIEW:
			break;
		case SINGLE_DEPTHVIEW:
			glCallList( vLists[uNthView_] );
			BREAK;
		case ALL_DEPTHVIEW:
			for (unsigned int i = 0; i < cKinectCalibExt.views(); i++)
				glCallList( vLists[i] );
			break;
	}
	return;
}
void renderAxis()
{
    glPushMatrix();
    float fAxisLength = 1.f;
    float fLengthWidth = 2;

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

void renderPattern()
{
    glPushMatrix();
    const std::vector<cv::Point3f>& vPts = cKinectCalibExt.pattern(0);
    glPointSize( 3 );
    glColor3d( .0 , .8 , .8 );
    for (std::vector<cv::Point3f>::const_iterator constItr = vPts.begin(); constItr < vPts.end() ; ++ constItr)
    {
        Vector3f vPt; vPt << *constItr;

        Vector3d vdPt = vPt.cast<double>();

        glBegin ( GL_POINTS );
        glVertex3d( vdPt(0), vdPt(1), vdPt(2) );
        glEnd();
    }
    glPopMatrix();
	return;
}

void placeCameraInWorldCoordinate(unsigned int uNthView_, int nCameraType_, int nMethod_ = 1)
{
    glPushMatrix();
    Eigen::Vector3d vT = cKinectCalibExt.eiVecT( uNthView_ ,nCameraType_ );
    Eigen::Matrix3d mR = cKinectCalibExt.eiMatR( uNthView_ ,nCameraType_ );
    
	//cout << "placeCameraInWorldCoordinate() after setup the RT\n";
    Eigen::Matrix4d mGLM = setOpenGLModelViewMatrix( mR, vT );
    mGLM = mGLM.inverse().eval();
    glMultMatrixd( mGLM.data() );
    
    //render camera
    cView.renderCamera( _vuTexture[nCameraType_][uNthView_], nCameraType_ , _uCamera );
    glPopMatrix();
}

void display ( void )
{
    glClearColor(0,0,0,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glColor3f ( 1.0, 1.0, 1.0 );
    
    glMatrixMode ( GL_MODELVIEW );
    // after set the intrinsics and extrinsics
    // load the matrix to set camera pose
    glLoadIdentity();
    glLoadMatrixd( _mGLMatrix.data() );
    // navigating the world
    gluLookAt ( _eivCamera(0), _eivCamera(1), _eivCamera(2),  _eivCenter(0), _eivCenter(1), _eivCenter(2), _eivUp(0), _eivUp(1), _eivUp(2) );
    glScaled( _dZoom, _dZoom, _dZoom );    
    glRotated ( _dYAngle, 0, 1 ,0 );
    glRotated ( _dXAngle, 1, 0 ,0 );

    // render objects
    renderAxis();
    renderPattern();
	render3DPts( _uNthView );

    // render cameras
    for (unsigned int i = 0; i < cKinectCalibExt.views(); i++)
    {
        placeCameraInWorldCoordinate(i, CCalibrateKinect::RGB_CAMERA);
        //placeCameraInWorldCoordinate(i, IR_CAMERA );
        placeCameraInWorldCoordinate(i, CCalibrateKinect::IR_CAMERA, 2 );
    }


    glutSwapBuffers();
}
void reshape ( int nWidth_, int nHeight_ )
{
    cView.setIntrinsics( _nScaleViewport, _nCameraType , _dNear, _dFar );

    glMatrixMode ( GL_MODELVIEW );

    /* setup blending */
    glBlendFunc ( GL_SRC_ALPHA, GL_ONE );			// Set The Blending Function For Translucency
    glColor4f ( 1.0f, 1.0f, 1.0f, 0.5 );
    return;
}

void setExtrinsics(unsigned int uNthView_, int nCameraType_, int nMethod_ = 1)
{
    // set extrinsics
    glMatrixMode ( GL_MODELVIEW );

    Eigen::Vector3d vT = cKinectCalibExt.eiVecT( uNthView_ ,nCameraType_ );
    Eigen::Matrix3d mR = cKinectCalibExt.eiMatR( uNthView_ ,nCameraType_ );

    _mGLMatrix = setOpenGLModelViewMatrix( mR, vT );

    _eivCamera = Vector3d(0., 0., 0.);
    _eivCenter = Vector3d(0., 0.,-1.);
    _eivUp     = Vector3d(0., 1., 0.);

    _dXAngle = _dYAngle = 0;
    _dX = _dY = 0;
    _dZoom = 1;

 /*   
    // 1. camera center
    _eivCamera = - mR.transpose() * mT;

    // 2. viewing vector
    Eigen::Matrix3d mK = cKinectCalibExt.eiMatK();
    Eigen::Matrix3d mM = mK * mR;
    Eigen::Vector3d vV =  mM.row(2).transpose();
    _eivCenter = _eivCamera + vV;

    // 3. upper vector, that is the normal of row 1 of P
    _eivUp = - mM.row(1).transpose(); //negative sign because the y axis of the image plane is pointing downward. 
    _eivUp.normalize();

    PRINT( _eivCamera );
    PRINT( _eivCenter );
    PRINT( _eivUp );*/

    return;
}

void setNextView()
{
    cView.setIntrinsics( _nScaleViewport, _nCameraType , _dNear, _dFar );
    _uNthView++; _uNthView %= cKinectCalibExt.views(); 
    setExtrinsics( _uNthView, _nCameraType );
    return;
}

void setPrevView()
{
    cView.setIntrinsics( _nScaleViewport, _nCameraType , _dNear, _dFar );
    _uNthView--; _uNthView %= cKinectCalibExt.views(); 
    setExtrinsics( _uNthView, _nCameraType );
    return;
}
void specialKeys ( int key, int x, int y )
{
	switch( key )
	{
		case GLUT_KEY_F2:
			_uDepthView = SINGLE_DEPTHVIEW; glutPostRedisplay();
			break;
		case GLUT_KEY_F3:
			_uDepthView = ALL_DEPTHVIEW; glutPostRedisplay();
			break;
		case GLUT_KEY_F4:
			_uDepthView = NONE_DEPTHVIEW; glutPostRedisplay();
			break;
		case GLUT_KEY_F5:
			_uCamera = NORGB_CAMERA; glutPostRedisplay();
			break;
		case GLUT_KEY_F6:
			_uCamera = ALL_CAMERA; glutPostRedisplay();
		default:
			;
	}
	return;
}
void processNormalKeys ( unsigned char key, int x, int y )
{
	switch( key )
	{
		case 27:
        	exit ( 0 );
			break;
		case '.':
	    	glEnable     ( GL_BLEND );
        	setNextView();
        	glutPostRedisplay();
			break;
    	case ',': 
	    	glEnable     ( GL_BLEND );
        	setPrevView();
        	glutPostRedisplay();
			break;
		case 'm':
        	//switch between IR_CAMERA and RGB_CAMERA
			glEnable     ( GL_BLEND );
        	if ( CCalibrateKinect::IR_CAMERA == _nCameraType )
            	_nCameraType = CCalibrateKinect::RGB_CAMERA;
        	else
            	_nCameraType = CCalibrateKinect::IR_CAMERA;
        	//set camera pose
        	cView.setIntrinsics( _nScaleViewport, _nCameraType, _dNear, _dFar );
        	setExtrinsics( _uNthView,       _nCameraType );
        	glutPostRedisplay();
			break;
    	case 'n':
	    	glEnable     ( GL_BLEND );
        	_nCameraType = CCalibrateKinect::IR_CAMERA;
        	//set camera pose
        	cView.setIntrinsics( _nScaleViewport, _nCameraType, _dNear, _dFar );
        	setExtrinsics( _uNthView,       _nCameraType, 2 );
        	glutPostRedisplay();
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
    	case 's':
        	if(_nScaleViewport == 1)
            	_nScaleViewport = 2;
        	else
            	_nScaleViewport = 1;
        	glutPostRedisplay();
			break;
		case '<':
		    _uNthView--; _uNthView %= cKinectCalibExt.views(); 
			glutPostRedisplay();
			break;
		case '>':
			_uNthView++; _uNthView %= cKinectCalibExt.views();
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

void mouseWheel(int button, int dir, int x, int y)
{
    cout << "mouse wheel." << endl;
    if (dir > 0)
    {
        _dZoom += 0.2;
        glutPostRedisplay();
    }
    else
    {
        _dZoom -= 0.2;
        glutPostRedisplay();
    }

    return;
}

int main ( int argc, char** argv )
{
    //get the path from command line
    boost::filesystem::path cFullPath ( boost::filesystem::initial_path<boost::filesystem::path>() );
	try
    {   
		cKinectCalibExt.mainFunc ( cFullPath );

        glutInit ( &argc, argv );
        glutInitDisplayMode ( GLUT_DOUBLE | GLUT_RGB );
        glutInitWindowSize ( cKinectCalibExt.imageResolution() ( 0 ), cKinectCalibExt.imageResolution() ( 1 ) );
        glutCreateWindow ( "CameraPose" );
        init();

        glutKeyboardFunc( processNormalKeys );
		glutSpecialFunc ( specialKeys );
        glutMouseFunc   ( mouseClick );
        glutMotionFunc  ( mouseMotion );
        glutMouseWheelFunc( mouseWheel );
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
