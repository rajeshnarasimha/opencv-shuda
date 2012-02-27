
#include <gl/freeglut.h>
#include <opencv2/core/core.hpp>
#include "boost/shared_ptr.hpp"
#include <opencv2/gpu/gpumat.hpp>
#include "Camera.h"
#include <string>
#include <Eigen/Core>
#include "CVUtil.hpp"

btl::kinect::SCamera::SCamera( tp_camera eT_ /*= CAMERA_RGB*/ )
	:_eType(eT_)
{
	importYML();
}

void btl::kinect::SCamera::generateMapXY4Undistort()
{
	cv::Mat_<float> cvmK(3,3); cvmK.setTo(0.f);
	cvmK.at<float>(0,0) = _fFx;cvmK.at<float>(1,1) = _fFy;cvmK.at<float>(2,2) = 1.f;
	cvmK.at<float>(0,2) = _u;cvmK.at<float>(1,2) = _v;
	cv::Mat_<float> cvmInvK = cvmK.inv(cv::DECOMP_SVD);
	btl::utility::map4UndistortImage<float> ( _sHeight, _sWidth, cvmK, cvmInvK, _cvmDistCoeffs, &_cvmMapX, &_cvmMapY );
	_cvgmMapX.upload(_cvmMapX);
	_cvgmMapY.upload(_cvmMapY);
}
void btl::kinect::SCamera::importYML()
{
	std::string strFileName;
	// create and open a character archive for output
#if __linux__
	strFileName = "/space/csxsl/src/opencv-shuda/Data/";
	//cv::FileStorage cFSRead( "/space/csxsl/src/opencv-shuda/Data/kinect_intrinsics.yml", cv::FileStorage::READ );
#else if _WIN32 || _WIN64
	strFileName = "C:\\csxsl\\src\\opencv-shuda\\Data\\";
	//cv::FileStorage cFSRead ( "C:\\csxsl\\src\\opencv-shuda\\Data\\kinect_intrinsics.yml", cv::FileStorage::READ );
#endif
	if( btl::kinect::SCamera::CAMERA_RGB ==_eType ) {strFileName += "CameraRGB.yml";}
	else if( btl::kinect::SCamera::CAMERA_IR ==_eType ) {strFileName += "CameraIR.yml";}

	cv::FileStorage cFSRead ( strFileName, cv::FileStorage::READ );
	cFSRead ["_fFx"] >> _fFx;
	cFSRead ["_fFy"] >> _fFy;
	cFSRead ["_u"] >> _u;
	cFSRead ["_v"] >> _v;
	cFSRead ["_sWidth"]  >> _sWidth;
	cFSRead ["_sHeight"] >> _sHeight;
	cFSRead ["_cvmDistCoeffs"] >> _cvmDistCoeffs;
	cFSRead.release();

	generateMapXY4Undistort();

	return;
}
void btl::kinect::SCamera::setIntrinsics ( unsigned int nScaleViewport_, const double dNear_, const double dFar_ )
{
//    glutReshapeWindow( int ( dWidth ), int ( dHeight ) );
    glMatrixMode ( GL_PROJECTION );

    double f = ( _fFx + _fFy ) / 2.;
    //no need to times nScaleViewport_ factor, because v/f, -(dHeight -v)/f cancel the factor off.
    double dLeft, dRight, dBottom, dTop;
    //Two assumptions:
    //1. assuming the principle point is inside the image
    //2. assuming the x axis pointing right and y axis pointing upwards
    dTop    =                _v   / f;
    dBottom = - ( _sHeight - _v ) / f;
    dLeft   =              - _u   / f;
    dRight  =   ( _sWidth  - _u ) / f;

    glLoadIdentity(); //use the same style as page 130, opengl programming guide
    glFrustum ( dLeft * dNear_, dRight * dNear_, dBottom * dNear_, dTop * dNear_, dNear_, dFar_ );
    glMatrixMode ( GL_VIEWPORT );

    if ( nScaleViewport_ == 2 )  { glViewport ( 0, - ( GLsizei ) _sHeight, ( GLsizei ) _sWidth * nScaleViewport_, ( GLsizei ) _sHeight * nScaleViewport_ ); }
    else if ( nScaleViewport_ == 1 )   { glViewport ( 0, 0, ( GLsizei ) _sWidth, ( GLsizei ) _sHeight ); }

    return;
}

void btl::kinect::SCamera::LoadTexture ( const cv::Mat& cvmImg_, GLuint* puTexture_ )
{
	glBindTexture ( GL_TEXTURE_2D, *puTexture_ );
	glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
	glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
	glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ); // cheap scaling when image bigger than texture
	glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ); // cheap scaling when image smalled than texture  
    // 2d texture, level of detail 0 (normal), 3 components (red, green, blue), x size from image, y size from image,
    // border 0 (normal), rgb color data, unsigned byte data, and finally the data itself.
    if( 3 == cvmImg_.channels())
        glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGB, cvmImg_.cols, cvmImg_.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, cvmImg_.data ); //???????????????????
    else if( 1 == cvmImg_.channels())
        glTexImage2D ( GL_TEXTURE_2D, 0, GL_LUMINANCE, cvmImg_.cols, cvmImg_.rows, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, cvmImg_.data );
    //glTexEnvi ( GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_REPEAT );
    // 2d texture, 3 colors, width, height, RGB in that order, byte data, and the data.
    //gluBuild2DMipmaps ( GL_TEXTURE_2D, GL_RGB, img.cols, img.rows,  GL_RGB, GL_UNSIGNED_BYTE, img.data );
    return;
}
void btl::kinect::SCamera::renderOnImage ( int nX_, int nY_ )
{
    const double f = ( _fFx + _fFy ) / 2.;

    // Draw principle point
    double dPhysicalFocalLength = .015;
    double dY =  _v - nY_;
    dY /= f;
    dY *= dPhysicalFocalLength;
    double dX = -_u + nX_;
    dX /= f;
    dX *= dPhysicalFocalLength;

    //draw principle point
    glVertex3d ( dX, dY, -dPhysicalFocalLength );
}
void btl::kinect::SCamera::renderCameraInGLLocal (const GLuint uTesture_, const cv::Mat& cvmImg_, float fPhysicalFocalLength_ /*= .02*/, bool bRenderTexture_/*=true*/ ) const 
{
	if(bRenderTexture_){
		glBindTexture(GL_TEXTURE_2D, uTesture_);

		if( 3 == cvmImg_.channels())
			glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, cvmImg_.cols,cvmImg_.rows, GL_RGB, GL_UNSIGNED_BYTE, cvmImg_.data);
		else if( 1 == cvmImg_.channels())
			glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, cvmImg_.cols,cvmImg_.rows, GL_LUMINANCE, GL_UNSIGNED_BYTE, cvmImg_.data);
	}

    const double f = ( _fFx + _fFy ) / 2.;

    // Draw principle point
    double dT =  _v;
    dT /= f;
    dT *= fPhysicalFocalLength_;
    double dB =  _v - _sHeight;
    dB /= f;
    dB *= fPhysicalFocalLength_;
    double dL = -_u;
    dL /= f;
    dL *= fPhysicalFocalLength_;
    double dR = -_u + _sWidth;
    dR /= f;
    dR *= fPhysicalFocalLength_;

    glPushAttrib ( GL_CURRENT_BIT );

    //draw principle point
    glColor3d ( 0., 0., 1. );
    glPointSize ( 5 );
    glBegin ( GL_POINTS );
    glVertex3d ( 0, 0, -fPhysicalFocalLength_ );
    glEnd();
	
    //draw principle axis
    glColor3d ( 0., 0., 1. );
    glLineWidth ( 1 );
    glBegin ( GL_LINES );
    glVertex3d ( 0, 0, 0 );
    glVertex3d ( 0, 0, -fPhysicalFocalLength_ );
    glEnd();

    //draw x axis in camera view
    glColor3d ( 1., 0., 0. ); //x
    glBegin ( GL_LINES );
    glVertex3d ( 0, 0, -fPhysicalFocalLength_ );
    glVertex3d ( dR, 0,-fPhysicalFocalLength_ );
    glEnd();

    //draw y axis in camera view
    glColor3d ( 0., 1., 0. ); //y
    glBegin ( GL_LINES );
    glVertex3d ( 0, 0, -fPhysicalFocalLength_ );
    glVertex3d ( 0, dT,-fPhysicalFocalLength_ );
    glEnd();

    glPopAttrib();

    //draw frame
    if ( bRenderTexture_ )
    {
        glEnable ( GL_TEXTURE_2D );
        glTexEnvf ( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
        glBindTexture ( GL_TEXTURE_2D, uTesture_ );

        glColor4f(1.f, 1.f, 1.f, 0.5f); glLineWidth(.5);
        glBegin ( GL_QUADS );
        glTexCoord2f ( 0.0, 0.0 );
        glVertex3d ( dL, dT, -fPhysicalFocalLength_ );
        glTexCoord2f ( 0.0, 1.0 );
        glVertex3d ( dL, dB, -fPhysicalFocalLength_ );
        glTexCoord2f ( 1.0, 1.0 );
        glVertex3d ( dR, dB, -fPhysicalFocalLength_ );
        glTexCoord2f ( 1.0, 0.0 );
        glVertex3d ( dR, dT, -fPhysicalFocalLength_ );
        glEnd();
        glDisable ( GL_TEXTURE_2D );
    }
/*
    //glColor3d(1., 1., 1.); glLineWidth(1.);
    glBegin ( GL_LINES );
    glVertex3d ( 0,   0, 0 );
    glVertex3d ( dL, dT, -dPhysicalFocalLength_ );
    glEnd();
    glBegin ( GL_LINES );
    glVertex3d ( 0,   0, 0 );
    glVertex3d ( dR, dT, -dPhysicalFocalLength_ );
    glEnd();
    glBegin ( GL_LINES );
    glVertex3d ( 0,   0, 0 );
    glVertex3d ( dR, dB, -dPhysicalFocalLength_ );
    glEnd();
    glBegin ( GL_LINES );
    glVertex3d ( 0,   0, 0 );
    glVertex3d ( dL, dB, -dPhysicalFocalLength_ );
    glEnd();
*/
    return;
}



