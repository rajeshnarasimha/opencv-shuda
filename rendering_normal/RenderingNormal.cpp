//display kinect depth in real-time
#define INFO

#include <GL/glew.h>
#include <iostream>
#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>
#include "Utility.hpp"

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
#include <opencv2/gpu/gpu.hpp>
#include "VideoSourceKinect.hpp"
#include "Model.h"

using namespace btl; //for "<<" operator
using namespace utility;
using namespace extra;
using namespace videosource;
using namespace Eigen;

class CKinectView;

btl::extra::videosource::VideoSourceKinect* _pVS;
btl::extra::videosource::CKinectView* _pView; 
btl::extra::CModel* _pModel;

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

bool _bCaptureCurrentFrame = false;
GLuint _uDisk;
GLuint _uNormal;
bool _bRenderNormal = false;
bool _bEnableLighting = false;
double _dDepthFilterThreshold = 10;
GLUquadricObj *_pQObj;
int _nDensity = 2;
double _dSize = 0.2; // range from 0.05 to 1 by step 0.05
unsigned int _uLevel = 3;
unsigned int _uPyrHeight = 4;
int _nColorIdx = 0;

enum tp_diplay {NORMAL_CLUSTRE, DISTANCE_CLUSTER};
tp_diplay _enumType = NORMAL_CLUSTRE;

void normalKeys ( unsigned char key, int x, int y )
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
		_pVS->_ePreFiltering = VideoSourceKinect::RAW;
		_pModel->_eNormalExtraction = CModel::_PCL;
		PRINTSTR(  "VideoSourceKinect::RAW" );
        break;
    case '2':
		_pVS->_ePreFiltering = VideoSourceKinect::RAW;
		_pModel->_eNormalExtraction = CModel::_FAST;
		PRINTSTR(  "VideoSourceKinect::RAW" );
        break;
    case '3':
		_pVS->_ePreFiltering = VideoSourceKinect::GAUSSIAN;
		PRINTSTR(  "VideoSourceKinect::GAUSSIAN" );
        break;
    case '4':
		_pVS->_ePreFiltering = VideoSourceKinect::GAUSSIAN_C1;
		PRINTSTR(  "VideoSourceKinect::GAUSSIAN_C1" );
        break;
	case '5':
		_pVS->_ePreFiltering = VideoSourceKinect::GAUSSIAN_C1_FILTERED_IN_DISPARTY;
		PRINTSTR(  "VideoSourceKinect::GAUSSIAN_C1_FILTERED_IN_DISPARTY" );
		break;
	case '6':
		_pVS->_ePreFiltering = VideoSourceKinect::BILATERAL_FILTERED_IN_DISPARTY;
		PRINTSTR(  "VideoSourceKinect::BILATERAL_FILTERED_IN_DISPARTY" );
		break;
	case '7':
		_pVS->_ePreFiltering = VideoSourceKinect::PYRAMID_BILATERAL_FILTERED_IN_DISPARTY;
		PRINTSTR(  "VideoSourceKinect::PYRAMID_BILATERAL_FILTERED_IN_DISPARTY" );
		break;
	case '8':
		_uLevel = ++_uLevel%_uPyrHeight;
		PRINT(_uLevel);
		break;
	case '9':
		_uPyrHeight++;
		PRINT(_uPyrHeight);
		break;
	case ']':
		_pVS->_dSigmaSpace += 1;
		PRINT( _pVS->_dSigmaSpace );
		break;
	case '[':
		_pVS->_dSigmaSpace -= 1;
		PRINT( _pVS->_dSigmaSpace );
		break;
	case '0'://reset camera location
		_dXAngle = 0.;
		_dYAngle = 0.;
		_dZoom = 1.;
		break;
    }
    return;
}
void specialKeys(int nKey_,int x, int y)
{
	switch( nKey_ )
	{
	case GLUT_KEY_F1: //display camera
		_nColorIdx++;
		PRINT(_nColorIdx);
		break;
	case GLUT_KEY_F2:
		_enumType = NORMAL_CLUSTRE == _enumType? DISTANCE_CLUSTER : NORMAL_CLUSTRE;
		if(NORMAL_CLUSTRE == _enumType)
		{
			PRINTSTR( "NORMAL_CLUSTRE" );
		}
		else
		{
			PRINTSTR( "DISTANCE_CLUSTER" );
		}
		break;
	}
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
void renderDisk(const Eigen::Vector3d& eivPt_, const Eigen::Vector3d& eivNl_, const unsigned char* pColor_,
	GLuint uDisk_, GLuint uNormal_, bool bRenderNormal_ )
{
	glColor3ubv( pColor_ );

	glPushMatrix();
	double x,y,z;
	x =  eivPt_(0);
	y =  eivPt_(1);
	z =  eivPt_(2);
	glTranslated( x, y, z );

	double dNx, dNy, dNz;// in opengl default coordinate
	dNx = eivNl_(0);
	dNy = eivNl_(1);
	dNz = eivNl_(2);

	if( fabs(dNx) + fabs(dNy) + fabs(dNz) < 0.00001 ) // normal is not computed
	{
		PRINT( dNz );
		return;
	}

	double dA = atan2(dNx,dNz);
	double dxz= sqrt( dNx*dNx + dNz*dNz );
	double dB = atan2(dNy,dxz);

	glRotated(-dB*180 / M_PI,1,0,0 );
	glRotated( dA*180 / M_PI,0,1,0 );
	double dR = -z/0.5;
	glScaled( dR*_dSize, dR*_dSize, dR*_dSize );
	glCallList(uDisk_);
	if( bRenderNormal_ )
	{
		glCallList(uNormal_);
	}
	glPopMatrix();
}
void render3DPts()
{
    if(_bCaptureCurrentFrame)
    {
        _pVS->_dThresholdDepth =_dDepthFilterThreshold;
		_pVS->_uPyrHeight = _uPyrHeight;
        //_pVS->getNextFrame();
		//_pModel->loadFrame();
		_pModel->loadPyramidAndDetectPlane();
		_pVS->centroidGL( &_eivCentroid );// get centroid of the depth map for display reasons
		_bCaptureCurrentFrame = false;
		std::cout << "capture done.\n" << std::flush;
	}
    
    const unsigned char* pColor;
    double x, y, z;
	if(_uLevel>=_pVS->_uPyrHeight)
	{
		PRINTSTR("CModel::pointCloud() uLevel_ is more than _uPyrHeight");
		_uLevel = 0;
	} 

	if( _bEnableLighting )
		glEnable(GL_LIGHTING);
	else
		glDisable(GL_LIGHTING);
	
	const std::vector< Eigen::Vector3d >* pvPtsPlane;
	const std::vector< Eigen::Vector3d >* pvNormalsPlane; 
	pvPtsPlane =&_pModel->_vvPyramidPts[_uPyrHeight-1];
	pvNormalsPlane =& _pModel->_vvPyramidNormals[_uPyrHeight-1];
	const std::vector< std::vector< unsigned int > >* pvvLabelPointIdx;
	if(NORMAL_CLUSTRE ==_enumType)
	{
		pvvLabelPointIdx = &_pModel->_vvLabelPointIdx;
	}
	else if(DISTANCE_CLUSTER ==_enumType)
	{
		pvvLabelPointIdx = &_pModel->_vvClusterPointIdx;
	}

	std::vector< std::vector< unsigned int > >::const_iterator cit = pvvLabelPointIdx->begin();
	for(int i=_nColorIdx; cit!=pvvLabelPointIdx->end(); cit++)
	{
		if(cit->size()>100)// the plane larger than 1500 pixels
		{
			const unsigned char* pColor = btl::utility::__aColors[i++%BTL_NUM_COLOR];
			std::vector< unsigned int >::const_iterator citIdx = cit->begin();
			for(; citIdx!=cit->end(); citIdx++)
			{
				renderDisk((*pvPtsPlane)[*citIdx], (*pvNormalsPlane)[*citIdx], pColor,_uDisk, _uNormal, _bRenderNormal); 
			}
		}
	}
	//render point cloud
    const std::vector< Eigen::Vector3d >& vPts=_pModel->_vvPyramidPts[_uLevel] ;
    const std::vector< Eigen::Vector3d >& vNormals = _pModel->_vvPyramidNormals[_uLevel];
    const std::vector<const unsigned char*>& vColors = _pModel->_vvPyramidColors[_uLevel];
    glPushMatrix();
    for (size_t i = 0; i < vPts.size (); ++i)
    {
        if ( 1 != _nDensity && i % _nDensity != 1 ) // skip some points; when 1 == i, all dots wills drawn;
        {
            continue;
        }
		renderDisk(vPts[i], vNormals[i], vColors[i],_uDisk, _uNormal, _bRenderNormal);
    }
    glPopMatrix();

    return;
}

void display ( void )
{
	_pVS->_dThresholdDepth =_dDepthFilterThreshold;
	_pVS->_uPyrHeight = _uPyrHeight;
	//_pVS->getNextFrame();
	//_pModel->loadFrame();
	_pModel->loadPyramidAndDetectPlane();
	_pVS->centroidGL( &_eivCentroid );// get centroid of the depth map for display reasons

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
    //glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, 640, 480, GL_RGBA, GL_UNSIGNED_BYTE, _pVS->cvRGB().data);
    //_pView->renderCamera( _uTexture, CCalibrateKinect::RGB_CAMERA );

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
    glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, 640, 480, GL_RGB, GL_UNSIGNED_BYTE, _pVS->cvRGB().data);
    _pView->renderCamera( _uTexture, CCalibrateKinect::RGB_CAMERA );

    glutSwapBuffers();
    glutPostRedisplay();

}

void reshape ( int nWidth_, int nHeight_ )
{
    //cout << "reshape() " << endl;
    _pView->setIntrinsics( 1, btl::extra::videosource::CCalibrateKinect::RGB_CAMERA, 0.01, 100 );

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
    glClearColor ( 0.1f,0.1f,0.4f,1.0f );
    glClearDepth ( 1.0 );
    glDepthFunc  ( GL_LESS );
    glEnable     ( GL_DEPTH_TEST );
    glEnable 	 ( GL_SCISSOR_TEST );
    glEnable     ( GL_CULL_FACE );
    glShadeModel ( GL_FLAT );

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    //_pModel->loadFrame();

	_pModel->loadPyramidAndDetectPlane();
    _uTexture = _pView->LoadTexture( _pVS->cvRGB() );

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
		_pVS = new btl::extra::videosource::VideoSourceKinect();
		_pView = new btl::extra::videosource::CKinectView(*_pVS);
		_pModel = new btl::extra::CModel(*_pVS);

        // Fill in the cloud data
        _cloud.width  = 640;
        _cloud.height = 480;
        _cloud.points.resize (_cloud.width * _cloud.height);

        glutInit ( &argc, argv );
        glutInitDisplayMode ( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH );
        glutInitWindowSize ( 1280, 480 );
        glutCreateWindow ( "CameraPose" );
        init();

        glutKeyboardFunc( normalKeys );
		glutSpecialFunc ( specialKeys );
        glutMouseFunc   ( mouseClick );
        glutMotionFunc  ( mouseMotion );

        glutReshapeFunc ( reshape );
        glutDisplayFunc ( display );
        glutMainLoop();
    }
	/*
    catch ( CError& e )
    {
        if ( std::string const* mi = boost::get_error_info< CErrorInfo > ( e ) )
        {
            std::cerr << "Error Info: " << *mi << std::endl;
        }
    }*/
	catch ( std::runtime_error& e )
	{
		PRINTSTR( e.what() );
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


