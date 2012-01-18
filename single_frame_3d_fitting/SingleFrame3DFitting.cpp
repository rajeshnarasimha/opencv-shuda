//display kinect depth in real-time
#include <iostream>
#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>
#include "Converters.hpp"
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

class CKinectView;

btl::extra::videosource::VideoSourceKinect _cVS;
btl::extra::videosource::CKinectView _cView(_cVS);

Eigen::Vector3d _eivCenter(.0, .0,-1.0 );
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

pcl::PointCloud<pcl::PointXYZ> _cloudNoneZero;
pcl::PointCloud<pcl::PointXYZ> _cloudPlane1;
pcl::PointCloud<pcl::PointXYZ> _cloudPlane2;
pcl::PointCloud<pcl::PointXYZ> _cloudPlane3;
pcl::PointCloud<pcl::PointXYZ> _cloudCylinder;

std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr > _vpCloudCluster;

cv::Mat _cvColor( 480, 640, CV_8UC3 );

pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

bool _bCaptureCurrentFrame = false;

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
        _dZoom += _dScale;
        glutPostRedisplay();
        break;
    case 'k':
        //zoom out
        glDisable     ( GL_BLEND );
        _dZoom -= _dScale;
        glutPostRedisplay();
        break;
    case 'c':
        //capture current frame the depth map and color
        _bCaptureCurrentFrame = true;
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
        // use the centroid as of the close by dot clouds as the centre of the 
        // origin. 
        inliers->indices.clear();
        _cloudNoneZero.clear();
        //for capturing the depth and color
        _cVS.cvRGB().copyTo( _cvColor );
        const double* pDepth = _cVS.alignedDepth();
        //convert depth map to PCL data
        for (size_t i = 0; i < _cloud.points.size (); ++i)
        {
            // convert to PCL data
            _cloud.points[i].x = *pDepth++;
            _cloud.points[i].y = *pDepth++;
            _cloud.points[i].z = *pDepth++;
            if( fabs(_cloud.points[i].z) > 0.0000001 )
            {
                pcl::PointXYZ point(_cloud.points[i].x,_cloud.points[i].y,_cloud.points[i].z);
                _cloudNoneZero.push_back(point);
            }
        }

        pcl::search::KdTree<pcl::PointXYZ>::Ptr pTree (new pcl::search::KdTree<pcl::PointXYZ>());
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;

        // Estimate point normals
        ne.setSearchMethod (pTree);
        ne.setInputCloud (_cloudNoneZero.makeShared());
        ne.setKSearch (30);
        ne.compute (_cloudNormals);

        pcl::ExtractIndices<pcl::Normal> cExtractNormals;
        ////////////////////////////
        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        //plane segmentation
        pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
        // Optional
        seg.setOptimizeCoefficients (true);
        // 1st plane
        seg.setModelType (pcl::SACMODEL_PLANE);
        seg.setMethodType (pcl::SAC_RANSAC);
        seg.setMaxIterations (1000);
        seg.setDistanceThreshold (0.01);

        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud (_cloudNoneZero.makeShared());
        seg.setInputNormals (_cloudNormals.makeShared());
        seg.segment (*inliers, *coefficients);
        // Create the filtering object
        pcl::ExtractIndices<pcl::PointXYZ> extract;

        // Extract the inliers
        extract.setInputCloud (_cloudNoneZero.makeShared());
        extract.setIndices (inliers);
        extract.setNegative (false);
        extract.filter (_cloudPlane1);
        extract.setNegative (true);
        extract.filter (_cloudNoneZero);

        cExtractNormals.setNegative (true);
        cExtractNormals.setInputCloud (_cloudNormals.makeShared());
        cExtractNormals.setIndices (inliers);
        cExtractNormals.filter (_cloudNormals);

        // Creating the KdTree object for the search method of the extraction
        pcl::search::KdTree<pcl::PointXYZ>::Ptr pTreePlane (new pcl::search::KdTree<pcl::PointXYZ>);
        pTreePlane->setInputCloud (_cloudPlane1.makeShared());

        std::vector<pcl::PointIndices> vClusterIndices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> cEC;
        cEC.setClusterTolerance (0.02); // 2cm
        cEC.setMinClusterSize (5000);
        cEC.setMaxClusterSize (2500000);
        cEC.setSearchMethod (pTreePlane);
        cEC.setInputCloud(_cloudPlane1.makeShared() );
        cEC.extract( vClusterIndices );

        _vpCloudCluster.clear();
        int j = 0;
        for (std::vector<pcl::PointIndices>::const_iterator cCluster_cit = vClusterIndices.begin (); cCluster_cit != vClusterIndices.end (); ++cCluster_cit)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr pCloudCluster (new pcl::PointCloud<pcl::PointXYZ>);
            for (std::vector<int>::const_iterator cit = cCluster_cit->indices.begin (); cit != cCluster_cit->indices.end (); cit++)
                pCloudCluster->points.push_back ( _cloudPlane1.points[*cit] ); 
            pCloudCluster->width = pCloudCluster->points.size ();
            pCloudCluster->height = 1;
            pCloudCluster->is_dense = true;

            _vpCloudCluster.push_back( pCloudCluster );
            j++;
        }

        PRINT( _vpCloudCluster.size() );

/*
        //2nd plane
        seg.setInputCloud (_cloudNoneZero.makeShared() );
        seg.segment (*inliers, *coefficients);

        extract.setInputCloud (_cloudNoneZero.makeShared());
        extract.setIndices (inliers);
        extract.setNegative (false);
        extract.filter (_cloudPlane2);

        extract.setNegative (true);
        extract.filter (_cloudNoneZero);

        cExtractNormals.setNegative (true);
        cExtractNormals.setInputCloud (_cloudNormals.makeShared());
        cExtractNormals.setIndices (inliers);
        cExtractNormals.filter (_cloudNormals);

        //3rd plane
        seg.setInputCloud (_cloudNoneZero.makeShared() );
        seg.segment (*inliers, *coefficients);

        // Extract the inliers
        extract.setInputCloud (_cloudNoneZero.makeShared());
        extract.setIndices (inliers);
        extract.setNegative (false);
        extract.filter (_cloudPlane3);
        extract.setNegative (true);
        extract.filter (_cloudNoneZero);

        cExtractNormals.setNegative (true);
        cExtractNormals.setInputCloud (_cloudNormals.makeShared());
        cExtractNormals.setIndices (inliers);
        cExtractNormals.filter (_cloudNormals);
*/
        /*
            //cylinder
                pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> cSegCylinder;
                // Create the segmentation object for cylinder segmentation and set all the parameters
                cSegCylinder.setOptimizeCoefficients (true);
                cSegCylinder.setModelType (pcl::SACMODEL_CYLINDER);
                cSegCylinder.setMethodType (pcl::SAC_RANSAC);
                cSegCylinder.setNormalDistanceWeight (0.1);
                cSegCylinder.setMaxIterations (10000);
                cSegCylinder.setDistanceThreshold (0.05);
                cSegCylinder.setRadiusLimits (0, 0.2);
                cSegCylinder.setInputCloud (_cloudNoneZero.makeShared());
                cSegCylinder.setInputNormals (_cloudNormals.makeShared());

                // Segment the largest planar component from the remaining cloud
                cSegCylinder.segment (*inliers, *coefficients);

                PRINT( *coefficients );
                PRINT( inliers->indices.size() );
                // Extract the inliers
                extract.setInputCloud (_cloudNoneZero.makeShared());
                extract.setIndices (inliers);
                extract.setNegative (false);
                extract.filter (_cloudCylinder);
                extract.setNegative (true);
                extract.filter (_cloudNoneZero);
        */
        _bCaptureCurrentFrame = false;
        std::cout << "capture done.\n" << std::flush;
    }
    //if( inliers )
    {
        glPushMatrix();
        glPointSize ( 1. );
        glBegin ( GL_POINTS );
        float fRed = 1.f;
        
        for (std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr >::const_iterator cit = _vpCloudCluster.begin(); cit!= _vpCloudCluster.end(); cit++ )
        {
            for( size_t i = 0; i < (*cit)->points.size(); ++i )
            {
                glColor3f( fRed, 0.f, 0.f );
                glVertex3d( (*cit)->points[i].x, -(*cit)->points[i].y, -(*cit)->points[i].z );


            }
            fRed -= 0.3;
        }
        /*
        for (size_t i = 0; i < _cloudPlane1.size (); ++i)
        {
            glColor3f ( 1.f, 0.f, 0.f );
            glVertex3d ( _cloudPlane1.points[i].x, -_cloudPlane1.points[i].y, -_cloudPlane1.points[i].z );
        }*/
        for (size_t i = 0; i < _cloudPlane2.size (); ++i)
        {
            glColor3f ( 0.f, 0.f, 1.f );
            glVertex3d ( _cloudPlane2.points[i].x, -_cloudPlane2.points[i].y, -_cloudPlane2.points[i].z );
        }
        for (size_t i = 0; i < _cloudPlane3.size (); ++i)
        {
            glColor3f ( 0.f, 1.f, 1.f );
            glVertex3d ( _cloudPlane3.points[i].x, -_cloudPlane3.points[i].y, -_cloudPlane3.points[i].z );
        }
        for (size_t i = 0; i < _cloudCylinder.size (); ++i)
        {
            glColor3f ( 0.f, 1.f, 0.f );
            glVertex3d ( _cloudCylinder.points[i].x, -_cloudCylinder.points[i].y, -_cloudCylinder.points[i].z );
        }
        glEnd();
        glPopMatrix();
    }
    
    const unsigned char* pColor = _cvColor.data;
    //convert depth map to PCL data
    glPushMatrix();
    glPointSize ( 1. );
    glBegin ( GL_POINTS );
    // Generate the data
    for (size_t i = 0; i < _cloud.points.size (); ++i)
    {
        if( abs(_cloud.points[i].z) > 0.0000001 )
        {
            glColor3ubv( pColor );
            glVertex3d ( _cloud.points[i].x, -_cloud.points[i].y, -_cloud.points[i].z );
        }
        pColor +=3;
    }
    glEnd();
    glPopMatrix();
}

void display ( void )
{
    _cVS.getNextFrame();
	_cVS.centroidGL(&_eivCenter);
    glMatrixMode ( GL_MODELVIEW );
    glViewport (0, 0, _nWidth/2, _nHeight);
    glScissor  (0, 0, _nWidth/2, _nHeight);
    // after set the intrinsics and extrinsics
    // load the matrix to set camera pose
    glLoadIdentity();
    //glLoadMatrixd( _mGLMatrix.data() );
    glTranslated( _eivCenter(0), _eivCenter(1), _eivCenter(2) ); // 5. translate back to the original camera pose
    _dZoom = _dZoom < 0.1? 0.1: _dZoom;
    _dZoom = _dZoom > 10? 10: _dZoom;
    glScaled( _dZoom, _dZoom, _dZoom );                          // 4. zoom in/out
    glRotated ( _dXAngle, 0, 1 ,0 );                             // 3. rotate horizontally
    glRotated ( _dYAngle, 1, 0 ,0 );                             // 2. rotate vertically
    glTranslated( -_eivCenter(0),-_eivCenter(1),-_eivCenter(2)); // 1. translate the world origin to align with object centroid
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // render objects
    renderAxis();
    render3DPts();
    glBindTexture(GL_TEXTURE_2D, _uTexture);
    //glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, 640, 480, GL_RGBA, GL_UNSIGNED_BYTE, _cVS.cvRGB().data);
    //_cView.renderCamera( _uTexture, CCalibrateKinect::RGB_CAMERA );

    glViewport (_nWidth/2, 0, _nWidth/2, _nHeight);
    glScissor  (_nWidth/2, 0, _nWidth/2, _nHeight);
    //gluLookAt ( _eivCamera(0), _eivCamera(1), _eivCamera(2),  _eivCenter(0), _eivCenter(1), _eivCenter(2), _eivUp(0), _eivUp(1), _eivUp(2) );
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
        // Fill in the cloud data
        _cloud.width  = 640;
        _cloud.height = 480;
        _cloud.points.resize (_cloud.width * _cloud.height);

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
        if ( std::string const* mi = boost::get_error_info< CErrorInfo > ( e ) )
        {
            std::cerr << "Error Info: " << *mi << std::endl;
        }
    }
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


