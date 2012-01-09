//display kinect depth in real-time
#include <GL/glew.h>
#include <iostream>
#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>
#include "Utility.hpp"
#include "VideoSourceKinect.hpp"
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

using namespace btl; //for "<<" operator
using namespace utility;
using namespace extra;
using namespace videosource;
using namespace Eigen;
//using namespace cv;

class CKinectView;

btl::extra::videosource::VideoSourceKinect _cVS;
btl::extra::videosource::CKinectView _cView(_cVS);

Eigen::Vector3d _eivCentroid(.0, .0, -1.0 );
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
GLuint _uTexture;

pcl::PointCloud<pcl::PointXYZ> _cloud;
pcl::PointCloud<pcl::Normal>   _cloudNormals;
std::vector<Eigen::Vector3d>   _vCloudNew;
std::vector<Eigen::Vector3d>   _vCloudNormalsNew;

pcl::PointCloud<pcl::PointXYZ> _cloudNoneZero;
std::vector<const unsigned char*> _vColors;
pcl::PointCloud<pcl::PointXYZ> _cloudPlane1;
pcl::PointCloud<pcl::PointXYZ> _cloudPlane2;
pcl::PointCloud<pcl::PointXYZ> _cloudPlane3;
pcl::PointCloud<pcl::PointXYZ> _cloudCylinder;

std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > _vpCloudCluster;

cv::Mat _cvColor( 480, 640, CV_8UC3 );


bool _bCaptureCurrentFrame = false;
GLuint _uDisk;
GLuint _uNormal;
bool _bRenderNormal = false;
bool _bEnableLighting = false;
double _dDepthFilterThreshold = 10;
GLUquadricObj *_pQObj;
int _nDensity = 2;
double _dSize = 0.2; // range from 0.05 to 1 by step 0.05

void processNormalKeys ( unsigned char key, int x, int y )
{
    switch( key )
    {
    case 27:
        exit ( 0 );
        break;
    case 'g':
        //zoom in
        glDisable     ( GL_BLEND );
        _dZoom += _dScale;
        glutPostRedisplay();
        break;
    case 'h':
        //zoom out
        glDisable     ( GL_BLEND );
        _dZoom -= _dScale;
        glutPostRedisplay();
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
        break;
    case 'l':
        _bEnableLighting = !_bEnableLighting;
        glutPostRedisplay();
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
        _cVS._eMethod = VideoSourceKinect::C1_CONTINUITY;
        PRINT( _cVS._eMethod );
        break;
    case '2':
        _cVS._eMethod = VideoSourceKinect::GAUSSIAN_C1;
        PRINT( _cVS._eMethod );
        break;
    case '3':
        _cVS._eMethod = VideoSourceKinect::DISPARIT_GAUSSIAN_C1;
        PRINT( _cVS._eMethod );
        break;
    case '4':
        _cVS._eMethod = VideoSourceKinect::RAW;
        PRINT( _cVS._eMethod );
        break;
    case '5':
        _cVS._eMethod = VideoSourceKinect::NEW_GAUSSIAN;
        PRINT( _cVS._eMethod );
        break;
	case '6':
		_cVS._eMethod = VideoSourceKinect::NEW_BILATERAL;
		PRINT( _cVS._eMethod );
		break;
	case '7':
		_cVS._eMethod = VideoSourceKinect::NEW_DEPTH;
		PRINT( _cVS._eMethod );
		break;
	case ']':
		_cVS._dSigmaSpace += 1;
		PRINT( _cVS._dSigmaSpace );
		break;
	case '[':
		_cVS._dSigmaSpace -= 1;
		PRINT( _cVS._dSigmaSpace );
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

//        _dZoom = _dZoom>0.
    }

    glutPostRedisplay();
}

void renderAxis()
{
    glDisable(GL_LIGHTING);

    glPushMatrix();
    float fAxisLength = 1.f;
    float fLengthWidth = 1;

    glLineWidth( fLengthWidth );
    // x axis
    glColor3f ( 1., .0, .0 );
    glBegin ( GL_LINES );

    glVertex3d ( .0, .0, .0 );
    Vector3d vXAxis;
    vXAxis << fAxisLength, .0, .0;
    glVertex3d ( vXAxis(0), vXAxis(1), vXAxis(2) );
    glEnd();
    // y axis
    glColor3f ( .0, 1., .0 );
    glBegin ( GL_LINES );
    glVertex3d ( .0, .0, .0 );
    Vector3d vYAxis;
    vYAxis << .0, fAxisLength, .0;
    glVertex3d ( vYAxis(0), vYAxis(1), vYAxis(2) );
    glEnd();
    // z axis
    glColor3f ( .0, .0, 1. );
    glBegin ( GL_LINES );
    glVertex3d ( .0, .0, .0 );
    Vector3d vZAxis;
    vZAxis << .0, .0, fAxisLength;
    glVertex3d ( vZAxis(0), vZAxis(1), vZAxis(2) );
    glEnd();
    glPopMatrix();
}

void render3DPts()
{
    if(_bCaptureCurrentFrame)
    {
        _cVS.setDepthFilterThreshold( _dDepthFilterThreshold );
        _cVS.getNextFrame();
		_cVS.centroidGL( &_eivCentroid );// get centroid of the depth map for display reasons
		_bCaptureCurrentFrame = false;
		std::cout << "capture done.\n" << std::flush;
	}
      
    const unsigned char* pColor;
    double x, y, z;
	 
    const std::vector< Eigen::Vector3d >& vPts=_cVS._vPts ;
    const std::vector< Eigen::Vector3d >& vNormals = _cVS._vNormals;
    const std::vector<const unsigned char*>&   vColors = _cVS._vColors;
    glPushMatrix();
// Generate the data
    for (size_t i = 0; i < vPts.size (); ++i)
    {
        if( _bEnableLighting )
            glEnable(GL_LIGHTING);
        else
            glDisable(GL_LIGHTING);
        if ( 1 != _nDensity && i % _nDensity != 1 ) // skip some points; when 1 == i, all dots wills drawn;
        {
            continue;
        }

        pColor = vColors[i];
        glColor3ubv( pColor );

        glPushMatrix();
        x =  vPts[i](0);
        y =  vPts[i](1);
        z =  vPts[i](2);
        glTranslated( x, y, z );

        double dNx, dNy, dNz;// in opengl default coordinate
        dNx = vNormals[i](0);
        dNy = vNormals[i](1);
        dNz = vNormals[i](2);

        if( fabs(dNx) + fabs(dNy) + fabs(dNz) < 0.00001 ) // normal is not computed
        {
            PRINT( dNz );
            continue;
        }

        double dA = atan2(dNx,dNz);
        double dxz= sqrt( dNx*dNx + dNz*dNz );
        double dB = atan2(dNy,dxz);

        glRotated(-dB*180 / M_PI,1,0,0 );
        glRotated( dA*180 / M_PI,0,1,0 );
        double dR = -z/0.5;
        glScaled( dR*_dSize, dR*_dSize, dR*_dSize );
        glCallList(_uDisk);
        if( _bRenderNormal )
        {
            glCallList(_uNormal);
        }
        glPopMatrix();
    }
    glPopMatrix();

    return;
}

void display ( void )
{
    glMatrixMode ( GL_MODELVIEW );
    glViewport (0, 0, _nWidth/2, _nHeight);
    glScissor  (0, 0, _nWidth/2, _nHeight);
    // after set the intrinsics and extrinsics
    // load the matrix to set camera pose
    glLoadIdentity();
    //glLoadMatrixd( _mGLMatrix.data() );
    glTranslated( _eivCentroid(0), _eivCentroid(1), _eivCentroid(2) ); // translate back to the original camera pose
    _dZoom = _dZoom < 0.1? 0.1: _dZoom;
    _dZoom = _dZoom > 10? 10: _dZoom;
    glScaled( _dZoom, _dZoom, _dZoom );                          // zoom in/out
    glRotated ( _dXAngle, 0, 1 ,0 );                             // rotate horizontally
    glRotated ( _dYAngle, 1, 0 ,0 );                             // rotate vertically
    glTranslated( -_eivCentroid(0),-_eivCentroid(1),-_eivCentroid(2)); // translate the world origin to align with object centroid
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // light position in 3d
    GLfloat light_position[] = { 3.0, 1.0, 1.0, 1.0 };
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);

    // render objects
    renderAxis();
    render3DPts();

    glBindTexture(GL_TEXTURE_2D, _uTexture);
    //glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, 640, 480, GL_RGBA, GL_UNSIGNED_BYTE, _cVS.cvRGB().data);
    //_cView.renderCamera( _uTexture, CCalibrateKinect::RGB_CAMERA );

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
    _mGLMatrix = Matrix4d::Identity();
    glClearColor ( 0.1f,0.1f,0.4f,1.0f );
    glClearDepth ( 1.0 );
    glDepthFunc  ( GL_LESS );
    glEnable     ( GL_DEPTH_TEST );
    glEnable 	 ( GL_SCISSOR_TEST );
    glEnable     ( GL_CULL_FACE );
//    glEnable     ( GL_BLEND );
//    glBlendFunc  ( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    glShadeModel ( GL_FLAT );

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    _cVS.getNextFrame();
    _uTexture = _cView.LoadTexture( _cVS.cvRGB() );

    _uDisk = glGenLists(1);
    GLUquadricObj *pQObj;
    _pQObj = gluNewQuadric();
    gluQuadricDrawStyle(_pQObj, GLU_FILL); //LINE); /* wireframe */
    gluQuadricNormals(_pQObj, GLU_SMOOTH);// FLAT);//
    glNewList(_uDisk, GL_COMPILE);
    gluDisk(_pQObj, 0.0, 0.01, 9, 1);
    glEndList();

    _uNormal = glGenLists(2);
    glNewList(_uNormal, GL_COMPILE);
    glDisable(GL_LIGHTING);
    glBegin(GL_LINES);
    glColor3d(1.,0.,0.);
    glVertex3d(0.,0.,0.);
    glVertex3d(0.,0.,0.016);
    glEnd();
    glEndList();

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
        // Fill in the cloud data
        _cloud.width  = 640;
        _cloud.height = 480;
        _cloud.points.resize (_cloud.width * _cloud.height);

        glutInit ( &argc, argv );
        glutInitDisplayMode ( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH );
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
        if ( std::string const* mi = boost::get_error_info< CErrorInfo > ( e ) )
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


