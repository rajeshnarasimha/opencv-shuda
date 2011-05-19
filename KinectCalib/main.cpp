#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <btl/extra/VideoSource/calibratekinect.hpp>
#include <btl/Utility/Converters.hpp>
#include <GL/freeglut.h>
//camera calibration from a sequence of images

#define IR_CAMERA 0
#define RGB_CAMERA 1

using namespace btl; //for "<<" operator
using namespace utility;
using namespace Eigen;
using namespace cv;
using namespace extra;
using namespace videosource;
CCalibrateKinect cKinectCalib;
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

bool _bDepthView = false;

unsigned int _uNthView = 0;
int  _nXMotion = 0;
int  _nYMotion = 0;
int  _nXLeftDown, _nYLeftDown;
int  _nXRightDown, _nYRightDown;
bool _bLButtonDown;
bool _bRButtonDown;

int _nType = RGB_CAMERA; //camera type
vector< GLuint > _vuTexture[2]; //[0] for ir [1] for rgb

void setIntrinsics(unsigned int nScaleViewport_, int nType_ );
void renderCamera (int nType_ );

GLuint LoadTexture(const cv::Mat& img)
{
    GLuint uTexture;
    glGenTextures ( 1, &uTexture );

    glBindTexture ( GL_TEXTURE_2D, uTexture );
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ); // cheap scaling when image bigger than texture
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ); // cheap scaling when image smalled than texture

    // 2d texture, level of detail 0 (normal), 3 components (red, green, blue), x size from image, y size from image,
    // border 0 (normal), rgb color data, unsigned byte data, and finally the data itself.
    glTexImage2D ( GL_TEXTURE_2D, 0, 3, img.cols, img.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, IplImage(img).imageData ); //???????????????????
    glTexEnvi ( GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_REPEAT );

    // 2d texture, 3 colors, width, height, RGB in that order, byte data, and the data.
    gluBuild2DMipmaps ( GL_TEXTURE_2D, 3, img.cols, img.rows,  GL_RGB, GL_UNSIGNED_BYTE, IplImage(img).imageData );
    return uTexture;
}

void init ( void )
{
    _mGLMatrix = Matrix4d::Identity();


    for( unsigned int n = 0; n < cKinectCalib.views(); n++ )
    {
        //load ir texture
        _vuTexture[0].push_back( LoadTexture( cKinectCalib.irUndistorted(  n ) ) ); 
        //load rgb texture
        _vuTexture[1].push_back( LoadTexture( cKinectCalib.rgbUndistorted( n ) ) ); 
    }

    glClearColor ( 0.0, 0.0, 0.0, 0.0 );
    glClearDepth ( 1.0 );
    glDepthFunc  ( GL_LESS );
    glEnable     ( GL_DEPTH_TEST );
    glEnable     ( GL_BLEND );
    glBlendFunc  ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glShadeModel ( GL_FLAT );
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
    const std::vector<cv::Point3f>& vPts = cKinectCalib.pattern(0);
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
}

void placeCameraInWorldCoordinate(unsigned int uNthView_, int nType_, int nMethod_ = 1)
{
    GLuint uTexture;
        
    glPushMatrix();
    Eigen::Vector3d vT;
    Eigen::Matrix3d mR;  

    if( IR_CAMERA == nType_ )
    {
        if( 1 == nMethod_ )
        {
            vT = cKinectCalib.eiVecIRT(uNthView_);
            mR = cKinectCalib.eiMatIRR(uNthView_); 
        }
        else if ( 2 == nMethod_ )
        {
            vT = cKinectCalib.eiMatRelativeRotation() *  cKinectCalib.eiVecRGBT(uNthView_) + cKinectCalib.eiVecRelativeTranslation();
            mR = cKinectCalib.eiMatRelativeRotation() *  cKinectCalib.eiMatRGBR(uNthView_); 
        }
        uTexture = _vuTexture[0][uNthView_];
    }
    else if ( RGB_CAMERA == nType_ )
    {
        vT = cKinectCalib.eiVecRGBT(uNthView_);
        mR = cKinectCalib.eiMatRGBR(uNthView_); 
        uTexture = _vuTexture[1][uNthView_];
    }
    else
    {
        THROW( " Unrecognized camera type.\n" );
    }

    Eigen::Matrix4d mGLM = setOpenGLModelViewMatrix( mR, vT );
    mGLM = mGLM.inverse().eval();
    glMultMatrixd( mGLM.data() );
    
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE); 
    glBindTexture(GL_TEXTURE_2D, uTexture); 
    //render camera
    renderCamera( nType_ );
    glPopMatrix();
}

void renderCamera( int nType_)
{
    Eigen::Matrix3d mK;

    if( IR_CAMERA == nType_ )
        mK = cKinectCalib.eiMatIRK();
    else if ( RGB_CAMERA == nType_ )
        mK = cKinectCalib.eiMatRGBK();
    else 
        THROW( " Unrecognized camera type.\n" );

    const double u = mK(0,2);
    const double v = mK(1,2);
    const double f = ( mK(0,0) + mK(1,1) )/2.;
    const double dW = cKinectCalib.imageResolution()(0);
    const double dH = cKinectCalib.imageResolution()(1);

    // Draw principle point
    double dPhysicalFocalLength = .02;
    double dT =  v;      dT /= f; dT *= dPhysicalFocalLength;
    double dB =  v - dH; dB /= f; dB *= dPhysicalFocalLength;
    double dL = -u;      dL /= f; dL *= dPhysicalFocalLength;
    double dR = -u + dW; dR /= f; dR *= dPhysicalFocalLength;


    //draw principle point
	glColor3d(0., 0., 1.);
	glPointSize(5);
	glBegin(GL_POINTS);
	glVertex3d(0,0,-dPhysicalFocalLength);
	glEnd();

    //draw principle axis
    glColor3d(0., 0., 1.);
	glLineWidth(1);
	glBegin(GL_LINES);
	glVertex3d(0,0,0);
    glVertex3d(0,0, -dPhysicalFocalLength);
	glEnd();

    //draw x axis in camera view
    glColor3d(1., 0., 0.);//x
	glBegin(GL_LINES);
	glVertex3d(0, 0,-dPhysicalFocalLength );
    glVertex3d(dR,0,-dPhysicalFocalLength );
	glEnd();

    //draw y axis in camera view
    glColor3d(0., 1., 0.);//y
	glBegin(GL_LINES);
	glVertex3d(0, 0, -dPhysicalFocalLength );
    glVertex3d(0, dT,-dPhysicalFocalLength );
	glEnd();

    //draw frame
    glEnable(GL_TEXTURE_2D); 
    glColor3d(1., 1., 1.); glLineWidth(.5);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0); glVertex3d(dL, dT,-dPhysicalFocalLength );
    glTexCoord2f(0.0, 1.0); glVertex3d(dL, dB,-dPhysicalFocalLength );
    glTexCoord2f(1.0, 1.0); glVertex3d(dR, dB,-dPhysicalFocalLength );
    glTexCoord2f(1.0, 0.0); glVertex3d(dR, dT,-dPhysicalFocalLength );
	glEnd();
    glDisable(GL_TEXTURE_2D);

    glColor3d(1., 1., 1.); glLineWidth(.5);
	glBegin(GL_LINES);
	glVertex3d(0,   0, 0 );
    glVertex3d(dL, dT,-dPhysicalFocalLength );
	glEnd();
	glBegin(GL_LINES);
	glVertex3d(0,   0, 0 );
    glVertex3d(dR, dT,-dPhysicalFocalLength );
	glEnd();
    glBegin(GL_LINES);
	glVertex3d(0,   0, 0 );
    glVertex3d(dR, dB,-dPhysicalFocalLength );
	glEnd();
    glBegin(GL_LINES);
	glVertex3d(0,   0, 0 );
    glVertex3d(dL, dB,-dPhysicalFocalLength );
	glEnd();
    return;
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

    // render cameras
    for (unsigned int i = 0; i < cKinectCalib.views(); i++)
    {
        placeCameraInWorldCoordinate(i, RGB_CAMERA);
        //placeCameraInWorldCoordinate(i, IR_CAMERA );
        placeCameraInWorldCoordinate(i, IR_CAMERA, 2 );
    }

    glutSwapBuffers();
}
void reshape ( int nWidth_, int nHeight_ )
{
    // std::cout << "reshape()" << std::endl;
    setIntrinsics( _nScaleViewport, _nType );

    glMatrixMode ( GL_MODELVIEW );

    /* setup blending */
    glBlendFunc ( GL_SRC_ALPHA, GL_ONE );			// Set The Blending Function For Translucency
    glColor4f ( 1.0f, 1.0f, 1.0f, 0.5 );
    return;
}

void setIntrinsics(unsigned int nScaleViewport_, int nType_ )
{
    // set intrinsics
    double dWidth = cKinectCalib.imageResolution()(0) * nScaleViewport_;
    double dHeight= cKinectCalib.imageResolution()(1) * nScaleViewport_;
    glutReshapeWindow( int ( dWidth ), int ( dHeight ) );

    glMatrixMode ( GL_PROJECTION );

    Eigen::Matrix3d mK;
    if( IR_CAMERA == nType_ )
        mK = cKinectCalib.eiMatIRK();
    else if ( RGB_CAMERA == nType_ )
        mK = cKinectCalib.eiMatRGBK();
    else 
        THROW( " Unrecognized camera type.\n" );

    double u = mK(0,2);
    double v = mK(1,2);
    double f = ( mK(0,0) + mK(1,1) )/2.;
    //no need to times nScaleViewport_ factor, because v/f, -(dHeight -v)/f cancel the factor off.
    double dNear = _dNear;
    double dLeft, dRight, dBottom, dTop;
   //Two assumptions:
   //1. assuming the principle point is inside the image
   //2. assuming the x axis pointing right and y axis pointing upwards
	dTop    =              v  /f;
	dBottom = -( dHeight - v )/f;
	dLeft   =             -u  /f;
	dRight  = ( dWidth   - u )/f;

    glLoadIdentity(); //use the same style as page 130, opengl programming guide
    glFrustum( dLeft*dNear, dRight*dNear, dBottom*dNear, dTop*dNear, dNear, _dFar );
    glMatrixMode( GL_VIEWPORT );
    if ( nScaleViewport_ == 2)
        glViewport ( 0, -( GLsizei ) dHeight, ( GLsizei ) dWidth*nScaleViewport_, ( GLsizei ) dHeight*nScaleViewport_ );
    else if (nScaleViewport_ == 1)
        glViewport ( 0, 0, ( GLsizei ) dWidth, ( GLsizei ) dHeight );

    // set intrinsics end.

    return;
}

void setExtrinsics(unsigned int uNthView_, int nType_, int nMethod_ = 1)
{
    // set extrinsics
    glMatrixMode ( GL_MODELVIEW );

    Eigen::Vector3d vT;
    Eigen::Matrix3d mR;  

    if( IR_CAMERA == nType_ )
    {
        if( 1 == nMethod_ )
        {
            vT = cKinectCalib.eiVecIRT(uNthView_);
            mR = cKinectCalib.eiMatIRR(uNthView_); 
        }
        else if ( 2 == nMethod_ )
        {
            vT = cKinectCalib.eiMatRelativeRotation() *  cKinectCalib.eiVecRGBT(uNthView_) + cKinectCalib.eiVecRelativeTranslation();
            mR = cKinectCalib.eiMatRelativeRotation() *  cKinectCalib.eiMatRGBR(uNthView_); 
        }
    }
    else if ( RGB_CAMERA == nType_ )
    {
        vT = cKinectCalib.eiVecRGBT(uNthView_);
        mR = cKinectCalib.eiMatRGBR(uNthView_); 
    }


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
    Eigen::Matrix3d mK = cKinectCalib.eiMatK();
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
        
    setIntrinsics( _nScaleViewport, _nType );
    
    _uNthView++; _uNthView %= cKinectCalib.views(); 

    setExtrinsics( _uNthView, _nType );

    return;
}

void setPrevView()
{
    setIntrinsics( _nScaleViewport, _nType );
    
    _uNthView--; _uNthView %= cKinectCalib.views(); 

    setExtrinsics( _uNthView, _nType );

    return;
}

void processNormalKeys ( unsigned char key, int x, int y )
{
    if ( key == 27 )
    {
        exit ( 0 );
    }

    if ( key == '.' )
    {
        _bDepthView = false;
        setNextView();
        glutPostRedisplay();
    }
    if ( key == ',' )
    {
        _bDepthView = false;
        setPrevView();
        glutPostRedisplay();
    }
    if ( key == 'm' )
    {
        //switch between IR_CAMERA and RGB_CAMERA
        if ( IR_CAMERA == _nType )
            _nType = RGB_CAMERA;
        else
            _nType = IR_CAMERA;
        //set camera pose
        setIntrinsics( _nScaleViewport, _nType );
        setExtrinsics( _uNthView,       _nType );
        glutPostRedisplay();
    }
    if ( key == 'n' )
    {
        _nType = IR_CAMERA;
        //set camera pose
        setIntrinsics( _nScaleViewport, _nType );
        setExtrinsics( _uNthView,       _nType, 2 );
        glutPostRedisplay();
    }

    if ( key == 'i' )
    {
        //zoom in
        _dZoom += 0.2;
        glutPostRedisplay();
    }
    if ( key == 'k' )
    {
        //zoom out
        _dZoom -= 0.2;
        glutPostRedisplay();
    }
    if ( key == 's' )
    {
        if(_nScaleViewport == 1)
            _nScaleViewport = 2;
        else
            _nScaleViewport = 1;
        glutPostRedisplay();
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
       _nXMotion = nX_ - _nXLeftDown;
       _nYMotion = nY_ - _nYLeftDown;
       _dXAngle  = _dXLastAngle + _nXMotion;
       _dYAngle  = _dYLastAngle + _nYMotion;
    }
    else if ( _bRButtonDown == true )
    {
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
	PRINT( cFullPath );
	/*
    if ( argc > 1 )
    {
        cFullPath = boost::filesystem::system_complete ( boost::filesystem::path ( argv[1] ) );
    }
    else
    {
        std::cout << "\nusage:   ./Exe [path]" << std::endl;
        return 0;
    }
	*/
    try
    {
        cKinectCalib.mainFunc ( cFullPath );
        
        glutInit ( &argc, argv );
        glutInitDisplayMode ( GLUT_DOUBLE | GLUT_RGB );
        glutInitWindowSize ( cKinectCalib.imageResolution() ( 0 ), cKinectCalib.imageResolution() ( 1 ) );
        glutCreateWindow ( "CameraPose" );
        init();

        glutKeyboardFunc( processNormalKeys );
        glutMouseFunc   ( mouseClick );
        glutMotionFunc  ( mouseMotion );
        glutReshapeFunc ( reshape );
        glutDisplayFunc ( display );
        glutMouseWheelFunc(mouseWheel );

        glutMainLoop();
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
