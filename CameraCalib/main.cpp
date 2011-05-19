#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "calibrationthroughimages.hpp"
#include <btl/Utility/Converters.hpp>
#include <GL/freeglut.h>
//camera calibration from a sequence of images

using namespace btl; //for "<<" operator
using namespace utility;
using namespace Eigen;
using namespace cv;

shuda::CCalibrationThroughImages cC;
Eigen::Vector3d _eivCamera(10.,10.,10.);
Eigen::Vector3d _eivCenter(.0, .0,.0 );
Eigen::Vector3d _eivUp(.0, 1.0, 0.0);
Matrix4d _mGLMatrix;

double _dXAngle = 0;
double _dYAngle = 0;
double _dXLastAngle = 0;
double _dYLastAngle = 0;

double _dZoom = 1.;
unsigned int _nScaleViewport = 1;

bool _bDepthView = false;

unsigned int _uNthView = 0;
GLuint _uTexture;
int  _nXMotion = 0;
int  _nYMotion = 0;
int  _nXLeftDown, _nYLeftDown;
bool _bLButtonDown;


void setIntrinsics(unsigned int nScaleViewport_ = 1);
void setDepthIntrinsics(unsigned int nScaleViewport_ = 1 );
void renderCamera(unsigned int uView_);
void renderDepthCamera(unsigned int uView_);

GLvoid LoadTexture(const cv::Mat& img)
{
    glGenTextures ( 1, &_uTexture );

    glBindTexture ( GL_TEXTURE_2D, _uTexture );
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ); // cheap scaling when image bigger than texture
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ); // cheap scaling when image smalled than texture

    // 2d texture, level of detail 0 (normal), 3 components (red, green, blue), x size from image, y size from image,
    // border 0 (normal), rgb color data, unsigned byte data, and finally the data itself.
    glTexImage2D ( GL_TEXTURE_2D, 0, 3, img.cols, img.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, IplImage(img).imageData ); //???????????????????
    glTexEnvi ( GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_REPEAT );

    // 2d texture, 3 colors, width, height, RGB in that order, byte data, and the data.
    gluBuild2DMipmaps ( GL_TEXTURE_2D, 3, img.cols, img.rows,  GL_BGR, GL_UNSIGNED_BYTE, IplImage(img).imageData );
}



void init ( void )
{
    // only 0,1,2,4,5,6,8,9,10 are responsible for rotation
//    m_dT[ 0] = 1; m_dT[ 4] = 0; m_dT[ 8] = 0; m_dT[12] = -10; 
//    m_dT[ 1] = 0; m_dT[ 5] = 1; m_dT[ 9] = 0; m_dT[13] = -10; 
//    m_dT[ 2] = 0; m_dT[ 6] = 0; m_dT[10] = 1; m_dT[14] = -10; 
//    m_dT[ 3] = 0; m_dT[ 7] = 0; m_dT[11] = 0; m_dT[15] = 1;

    _mGLMatrix = Matrix4d::Identity();

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
    const std::vector<cv::Point3f>& vPts = cC.pattern(0);
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

void renderCameraWorld(unsigned int uView_)
{
    glPushMatrix();

    Eigen::Vector3d vT = cC.eiVecT(uView_);
    Eigen::Matrix3d mR = cC.eiMatR(uView_);  
    Eigen::Matrix4d mGLM = setOpenGLModelViewMatrix( mR, vT );
    mGLM = mGLM.inverse().eval();
    //double dMat[16]; setOpenGLModelViewMatrix( mGLM.inverse(), dMat ); //compute the inverse of mR\vT
    glMultMatrixd( mGLM.data() );
    renderCamera(uView_);
    glPopMatrix();
}

void renderDepthCameraWorld(unsigned int uView_)
{
    glPushMatrix();

    Eigen::Vector3d vT = cC.eiVecT(uView_);
    Eigen::Matrix3d mR = cC.eiMatR(uView_);  
    Eigen::Matrix4d mGLM = setOpenGLModelViewMatrix( mR, vT );
    mGLM = mGLM.inverse().eval();
    //double dMat[16]; setOpenGLModelViewMatrix( mGLM.inverse(), dMat ); //compute the inverse of mR\vT
    glMultMatrixd( mGLM.data() );
    renderDepthCamera(uView_);
    glPopMatrix();
}

void renderDepthCamera(unsigned int uView_)
{
 // render camera
    Eigen::Matrix3d mK = cC.eiMatDepthK();
    const double u = mK(0,2);
    const double v = mK(1,2);
    const double f = ( mK(0,0) + mK(1,1) )/2.;
    const double dW = cC.imageResolution()(0);
    const double dH = cC.imageResolution()(1);

    // Draw principle point
    double dPhysicalFocalLength = .03;
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
/*
    //draw principle axis
    glColor3d(0., 0., 1.);
	glPointSize(5);
	glBegin(GL_LINES);
	glVertex3d(0,0,0);
    glVertex3d(0,0,-1.5 * dPhysicalFocalLength);
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
*/
    //draw frame
    glEnable(GL_TEXTURE_2D); 
    glColor3d(1., 1., 1.);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0); glVertex3d(dL, dT,-dPhysicalFocalLength );
    glTexCoord2f(0.0, 1.0); glVertex3d(dL, dB,-dPhysicalFocalLength );
    glTexCoord2f(1.0, 1.0); glVertex3d(dR, dB,-dPhysicalFocalLength );
    glTexCoord2f(1.0, 0.0); glVertex3d(dR, dT,-dPhysicalFocalLength );
	glEnd();
    glDisable(GL_TEXTURE_2D);

    glColor3d(1., 1., 1.);//y
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
/*
    //render depth
    const Mat& cvDepth = cC.depth( uView_ ); 
    unsigned char* pDepth = (unsigned char*) cvDepth.data;
    Vector3d eivDirection;
    glBegin(GL_POINTS);
    PRINT ( cC.depthPathName(uView_) );
    for(int r = 0; r <  dH; r += 4 )
    for(int c = 0; c <  dW; c += 4 )
    {
        eivDirection(0) = c - u;
        eivDirection(1) = r - v;
        eivDirection(2) = -f;
        eivDirection.normalize();
        int nDepth =  int(*(pDepth+2));
        eivDirection *= nDepth/6.;
        glColor3d( (*pDepth)/255.,*(pDepth+1)/255.,*(pDepth+2)/255. );
        glVertex3dv( eivDirection.data() );
        pDepth += 3;
    }
    glEnd();
*/
    return;
}

void renderCamera(unsigned int uView_)
{
 // render camera
    Eigen::Matrix3d mK = cC.eiMatK();
    const double u = mK(0,2);
    const double v = mK(1,2);
    const double f = ( mK(0,0) + mK(1,1) )/2.;
    const double dW = cC.imageResolution()(0);
    const double dH = cC.imageResolution()(1);

    // Draw principle point
    double dPhysicalFocalLength = .03;
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
/*
    //draw principle axis
    glColor3d(0., 0., 1.);
	glPointSize(5);
	glBegin(GL_LINES);
	glVertex3d(0,0,0);
    glVertex3d(0,0,-1.5 * dPhysicalFocalLength);hao
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
*/
    //draw frame
    glEnable(GL_TEXTURE_2D); 
    glColor3d(1., 1., 1.);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0); glVertex3d(dL, dT,-dPhysicalFocalLength );
    glTexCoord2f(0.0, 1.0); glVertex3d(dL, dB,-dPhysicalFocalLength );
    glTexCoord2f(1.0, 1.0); glVertex3d(dR, dB,-dPhysicalFocalLength );
    glTexCoord2f(1.0, 0.0); glVertex3d(dR, dT,-dPhysicalFocalLength );
	glEnd();
    glDisable(GL_TEXTURE_2D);

    glColor3d(1., 1., 1.);//y
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
/*
    //render depth
    const Mat& cvDepth = cC.depth( uView_ ); 
    unsigned char* pDepth = (unsigned char*) cvDepth.data;
    Vector3d eivDirection;
    glBegin(GL_POINTS);
    PRINT ( cC.depthPathName(uView_) );
    for(int r = 0; r <  dH; r += 4 )
    for(int c = 0; c <  dW; c += 4 )
    {
        eivDirection(0) = c - u;
        eivDirection(1) = r - v;
        eivDirection(2) = -f;
        eivDirection.normalize();
        int nDepth =  int(*(pDepth+2));
        eivDirection *= nDepth/6.;
        glColor3d( (*pDepth)/255.,*(pDepth+1)/255.,*(pDepth+2)/255. );
        glVertex3dv( eivDirection.data() );
        pDepth += 3;
    }
    glEnd();
*/
    return;
}
void display ( void )
{
    glClearColor(0,0,0,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    glColor3f ( 1.0, 1.0, 1.0 );
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE); 
    glBindTexture(GL_TEXTURE_2D, _uTexture); 
    
    glMatrixMode ( GL_MODELVIEW );
    glLoadIdentity();
    glLoadMatrixd( _mGLMatrix.data() );

    gluLookAt ( _eivCamera(0), _eivCamera(1), _eivCamera(2),  _eivCenter(0), _eivCenter(1), _eivCenter(2), _eivUp(0), _eivUp(1), _eivUp(2) );
    glScaled( _dZoom, _dZoom, _dZoom );    
    glRotated ( _dYAngle, 0, 1 ,0 );

    glRotated ( _dXAngle, 1, 0 ,0 );
    renderAxis();
    renderPattern();
    for (unsigned int i = 0; i < cC.views(); i++)
    {
        renderCameraWorld(i);
        renderDepthCameraWorld(i);
    }


    glutSwapBuffers();
}
void reshape ( int nWidth_, int nHeight_ )
{
    // std::cout << "reshape()" << std::endl;
    if( _bDepthView )
        setDepthIntrinsics( _nScaleViewport );
    else
        setIntrinsics( _nScaleViewport );

    glMatrixMode ( GL_MODELVIEW );

    /* setup blending */
    glBlendFunc ( GL_SRC_ALPHA, GL_ONE );			// Set The Blending Function For Translucency
    glColor4f ( 1.0f, 1.0f, 1.0f, 0.5 );
}

void setDepthIntrinsics(unsigned int nScaleViewport_ )
{
    std::cout << "setDepthIntrinsics()" << std::endl;
    // set intrinsics
    double dWidth = cC.imageResolution()(0) * nScaleViewport_;
    double dHeight= cC.imageResolution()(1) * nScaleViewport_;
    glutReshapeWindow( int ( dWidth ), int ( dHeight ) );

    glMatrixMode ( GL_PROJECTION );

    Eigen::Matrix3d mK = cC.eiMatDepthK();
    double u = mK(0,2);
    double v = mK(1,2);
    double f = ( mK(0,0) + mK(1,1) )/2.;
    //no need to times nScaleViewport_ factor, because v/f, -(dHeight -v)/f cancel the factor off.
    double dNear = cC.near();
    double dLeft, dRight, dBottom, dTop;
   //Two assumptions:
   //1. assuming the principle point is inside the image
   //2. assuming the x axis pointing right and y axis pointing upwards
	dTop    =              v  /f;
	dBottom = -( dHeight - v )/f;
	dLeft   =             -u  /f;
	dRight  = ( dWidth   - u )/f;

    PRINT( f );
    PRINT( u );
    PRINT( v );
    glLoadIdentity(); //use the same style as page 130, opengl programming guide
    glFrustum( dLeft*dNear, dRight*dNear, dBottom*dNear, dTop*dNear, dNear, cC.far() );
    glMatrixMode( GL_VIEWPORT );
    if ( nScaleViewport_ == 2)
        glViewport ( 0, -( GLsizei ) dHeight, ( GLsizei ) dWidth*nScaleViewport_, ( GLsizei ) dHeight*nScaleViewport_ );
    else if (nScaleViewport_ == 1)
        glViewport ( 0, 0, ( GLsizei ) dWidth, ( GLsizei ) dHeight );



    // set intrinsics end.

    PRINT( u );
    PRINT( v );
    PRINT( f );
/*  PRINT( dWidth );
    PRINT( dHeight );
    PRINT( dTop );
    PRINT( dBottom );
    PRINT( dLeft );
    PRINT( dRight );
*/
    return;
}


void setIntrinsics(unsigned int nScaleViewport_ )
{
    std::cout << "setIntrinsics()" << std::endl;
    // set intrinsics
    double dWidth = cC.imageResolution()(0) * nScaleViewport_;
    double dHeight= cC.imageResolution()(1) * nScaleViewport_;
    glutReshapeWindow( int ( dWidth ), int ( dHeight ) );

    glMatrixMode ( GL_PROJECTION );

    Eigen::Matrix3d mK = cC.eiMatK();
    double u = mK(0,2);
    double v = mK(1,2);
    double f = ( mK(0,0) + mK(1,1) )/2.;
    //no need to times nScaleViewport_ factor, because v/f, -(dHeight -v)/f cancel the factor off.
    double dNear = cC.near();
    double dLeft, dRight, dBottom, dTop;
   //Two assumptions:
   //1. assuming the principle point is inside the image
   //2. assuming the x axis pointing right and y axis pointing upwards
	dTop    =              v  /f;
	dBottom = -( dHeight - v )/f;
	dLeft   =             -u  /f;
	dRight  = ( dWidth   - u )/f;

    glLoadIdentity(); //use the same style as page 130, opengl programming guide
    glFrustum( dLeft*dNear, dRight*dNear, dBottom*dNear, dTop*dNear, dNear, cC.far() );
    glMatrixMode( GL_VIEWPORT );
    if ( nScaleViewport_ == 2)
        glViewport ( 0, -( GLsizei ) dHeight, ( GLsizei ) dWidth*nScaleViewport_, ( GLsizei ) dHeight*nScaleViewport_ );
    else if (nScaleViewport_ == 1)
        glViewport ( 0, 0, ( GLsizei ) dWidth, ( GLsizei ) dHeight );

    // set intrinsics end.
/*
    PRINT( u );
    PRINT( v );
    PRINT( f );
    PRINT( dWidth );
    PRINT( dHeight );
    PRINT( dTop );
    PRINT( dBottom );
    PRINT( dLeft );
    PRINT( dRight );
*/
    return;
}

void setView(unsigned int uNthView_)
{
    // set extrinsics
    glMatrixMode ( GL_MODELVIEW );

    //load texture for rendering the image

    LoadTexture( cC.undistortedImg( uNthView_ ) ); 

    Eigen::Vector3d mT = cC.eiVecT(uNthView_);
    Eigen::Matrix3d mR = cC.eiMatR(uNthView_);  

    _mGLMatrix = setOpenGLModelViewMatrix( mR, mT );

    _eivCamera = Vector3d(0., 0., 0.);
    _eivCenter = Vector3d(0., 0.,-1.);
    _eivUp     = Vector3d(0., 1., 0.);

    _dXAngle = _dYAngle = 0;
    _dZoom = 1;

 /*   // 1. camera center
    _eivCamera = - mR.transpose() * mT;

    // 2. viewing vector
    Eigen::Matrix3d mK = cC.eiMatK();
    Eigen::Matrix3d mM = mK * mR;
    Eigen::Vector3d vV =  mM.row(2).transpose();
    _eivCenter = _eivCamera + vV;

    // 3. upper vector, that is the normal of row 1 of P
    _eivUp = - mM.row(1).transpose(); //negative sign because the y axis of the image plane is pointing downward. 
    _eivUp.normalize();

   */ 
    /*
    PRINT( cC.imagePathName(_uNthView) );
    PRINT( cC.cvMatR(_uNthView) );
    PRINT( cC.cvMatT(_uNthView) );

    PRINT( _eivCamera );
    PRINT( _eivCenter );
    PRINT( _eivUp );*/


    return;

}

void setDepthView(unsigned int uNthView_)
{
    // set extrinsics
    glMatrixMode ( GL_MODELVIEW );

    //load texture for rendering the image
    LoadTexture( cC.undistortedDepth( uNthView_) ); 

    Eigen::Vector3d mT = cC.eiVecT(uNthView_);
    Eigen::Matrix3d mR = cC.eiMatR(uNthView_);  

    _mGLMatrix = setOpenGLModelViewMatrix( mR, mT );

    _eivCamera = Vector3d(0., 0., 0.);
    _eivCenter = Vector3d(0., 0.,-1.);
    _eivUp     = Vector3d(0., 1., 0.);

    _dXAngle = _dYAngle = 0;
    _dZoom = 1;

 /*   // 1. camera center
    _eivCamera = - mR.transpose() * mT;

    // 2. viewing vector
    Eigen::Matrix3d mK = cC.eiMatK();
    Eigen::Matrix3d mM = mK * mR;
    Eigen::Vector3d vV =  mM.row(2).transpose();
    _eivCenter = _eivCamera + vV;

    // 3. upper vector, that is the normal of row 1 of P
    _eivUp = - mM.row(1).transpose(); //negative sign because the y axis of the image plane is pointing downward. 
    _eivUp.normalize();

   */ 
    /*
    PRINT( cC.imagePathName(_uNthView) );
    PRINT( cC.cvMatR(_uNthView) );
    PRINT( cC.cvMatT(_uNthView) );

    PRINT( _eivCamera );
    PRINT( _eivCenter );
    PRINT( _eivUp );*/


    return;

}


void setNextView()
{
        
    setIntrinsics( _nScaleViewport );
    
    _uNthView++; _uNthView %= cC.views(); 

    setView(_uNthView);

    return;
}

void setPrevView()
{
    setIntrinsics( _nScaleViewport );
    
    _uNthView--; _uNthView %= cC.views(); 

    setView(_uNthView);

    return;
}

void switchToDepthView()
{
        
    setDepthIntrinsics( _nScaleViewport );
    
    setDepthView(_uNthView);

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
        _bDepthView = !_bDepthView;
        if( _bDepthView )
            switchToDepthView();
        else
        {
            setIntrinsics( _nScaleViewport );
            setView(_uNthView);
        }
        glutPostRedisplay();
    }
    if ( key == 'i' )
    {
        //Vector3d eivZoom = _eivCenter - _eivCamera; eivZoom.normalize();
        //_eivCamera += 0.2 * eivZoom;
        _dZoom += 0.2;
        glutPostRedisplay();
    }
    if ( key == 'k' )
    {
        //Vector3d eivZoom = _eivCenter - _eivCamera; eivZoom.normalize();
        //_eivCamera -= 0.2 * eivZoom;
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

    if ( argc > 1 )
    {
        cFullPath = boost::filesystem::system_complete ( boost::filesystem::path ( argv[1] ) );
    }
    else
    {
        std::cout << "\nusage:   ./Exe [path]" << std::endl;
        return 0;
    }

    try
    {
        cC.main ( cFullPath );

        glutInit ( &argc, argv );
        glutInitDisplayMode ( GLUT_DOUBLE | GLUT_RGB );
        glutInitWindowSize ( cC.imageResolution() ( 0 ), cC.imageResolution() ( 1 ) );
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
