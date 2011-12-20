/**
* @file VideoSourceKinect.cpp
* @brief load of data from a kinect device 
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.0
* @date 2011-02-23
*/
#include "VideoSourceKinect.hpp"
#include "Converters.hpp"
#include <iostream>
#include <cassert>
#include <string>


#define CHECK_RC(rc, what)											            \
	if (rc != XN_STATUS_OK)											            \
	{																            \
		throw Exception(what + std::string(xnGetStatusString(rc)));\
	}


using namespace std;
using namespace btl;
using namespace utility;

namespace btl
{
namespace extra
{
namespace videosource
{

VideoSourceKinect::VideoSourceKinect ()
:CCalibrateKinect()
{
    _frameSize = Eigen::Vector2i ( KINECT_VIDEO_W, KINECT_VIDEO_H );

    cout << "  VideoSource_Linux: Opening Kinect..." << endl;

    XnStatus nRetVal = XN_STATUS_OK;
    //initialize OpenNI context
/*    
    nRetVal = _cContext.InitFromXmlFile("/space/csxsl/src/btl-shuda/Kinect.xml"); 
    CHECK_RC ( nRetVal, "Initialize context: " );
    nRetVal = _cContext.FindExistingNode(XN_NODE_TYPE_IR, _cIRGen); 
    CHECK_RC ( nRetVal, "Find existing node: " );
*/    
    nRetVal = _cContext.Init();
    CHECK_RC ( nRetVal, "Initialize context: " );

    //create a image generator
    nRetVal =  _cImgGen.Create ( _cContext );
    CHECK_RC ( nRetVal, "Create image generator: " );
    //create a depth generator
    nRetVal =  _cDepthGen.Create ( _cContext );
    CHECK_RC ( nRetVal, "Create depth generator: " );
    //start generating data
    nRetVal = _cContext.StartGeneratingAll();
    CHECK_RC ( nRetVal, "Start generating data: " );
	//register the depth generator with the image generator
    //nRetVal = _cDepthGen.GetAlternativeViewPointCap().SetViewPoint ( _cImgGen );
	//CHECK_RC ( nRetVal, "Getting and setting AlternativeViewPoint failed: " ); 

    _cvImage.create( _frameSize ( 1 ), _frameSize ( 0 ), CV_8UC3 );
	//_cvDepth.create( _frameSize ( 1 ), _frameSize ( 0 ), CV_8UC3 );
	_cvDepth.create( _frameSize ( 1 ), _frameSize ( 0 ), CV_16UC1 );
	_cvUndistImage.create( _frameSize ( 1 ), _frameSize ( 0 ), CV_8UC3 );
	_cvUndistDepth.create( _frameSize ( 1 ), _frameSize ( 0 ), CV_16UC1 );
	_cvUndistFilteredDepth.create( _frameSize ( 1 ), _frameSize ( 0 ), CV_16UC1 );
	_cvUndistImage.create( _frameSize ( 1 ), _frameSize ( 0 ), CV_8UC1 );

    //_pcvDepth = new cv::Mat( _frameSize ( 1 ), _frameSize ( 0 ), CV_8UC3 );
	
	//_pcvUndistImage =new cv::Mat( _frameSize ( 1 ), _frameSize ( 0 ), CV_8UC3 );
    //_pcvUndistDepth= new cv::Mat( _frameSize ( 1 ), _frameSize ( 0 ), CV_8UC3 );
    _eMethod = C1_CONTINUITY;
    _cvColor.create( 480, 640, CV_8UC3 );
	cout << " Done. " << endl;
}

VideoSourceKinect::~VideoSourceKinect()
{
    _cContext.Release();
}

void VideoSourceKinect::getNextFrame()
{
    //get next frame
    //set as _frame
	//cout << " getNextFrame() start."<< endl;

// timer on
//	_cT0 =  boost::posix_time::microsec_clock::local_time(); 

    XnStatus nRetVal = _cContext.WaitAndUpdateAll();
    CHECK_RC ( nRetVal, "UpdateData failed: " );
	// these two lines are required for getting a stable image and depth.
    _cImgGen.GetMetaData ( _cImgMD );
    _cDepthGen.GetMetaData( _cDepthMD );

    const XnRGB24Pixel* pRGBImg  = _cImgMD.RGB24Data();
	     unsigned char* pcvImage = _cvImage.data;
    const unsigned short* pDepth   = (unsigned short*)_cDepthMD.Data();
	      unsigned short* pcvDepth = (unsigned short*)_cvDepth.data;
    
    //XnStatus nRetVal = _cContext.WaitOneUpdateAll( _cIRGen );
    //CHECK_RC ( nRetVal, "UpdateData failed: " );

	for ( unsigned int y = 0; y < _cImgMD.YRes(); y++ )
    {
        for ( unsigned int x = 0; x < _cImgMD.XRes(); x++ )
        {
            // notice that OpenCV is use BGR order
            *pcvImage++ = uchar(pRGBImg->nRed);
            *pcvImage++ = uchar(pRGBImg->nGreen);
            *pcvImage++ = uchar(pRGBImg->nBlue);
			pRGBImg++;

			*pcvDepth++ = *pDepth++;
        }
    }

	// not fullly understand the lense distortion model used by OpenNI.
	undistortRGB( _cvImage, _cvUndistImage );
	undistortRGB( _cvDepth, _cvUndistDepth );
	cvtColor( _cvUndistImage, _cvmUndistBW, CV_RGB2GRAY );

    cv::Mat_<double> cvDisparity( _cvUndistDepth.rows, _cvUndistDepth.cols, CV_32F );
    cv::Mat_<double> cvFilterDisparity( _cvUndistDepth.rows, _cvUndistDepth.cols, CV_32F );
    cv::Mat_<double> cvThersholdDisparity( _cvUndistDepth.rows, _cvUndistDepth.cols, CV_32F );

    cv::Mat_<unsigned short> cvmFilter(_cvUndistDepth.rows, _cvUndistDepth.cols, CV_16U );
    double dDispThreshold;
    switch( _eMethod )
    {
        case RAW:
    	    // register the depth with rgb image
    	    registration( (const unsigned short*)_cvUndistDepth.data );
            break;
        case C1_CONTINUITY:
            btl::utility::filterDepth <unsigned short> ( _dThresholdDepth, (Mat_<unsigned short>)_cvUndistDepth, (Mat_<unsigned short>*)&_cvUndistFilteredDepth );
    	    // register the depth with rgb image
    	    registration( (const unsigned short*)_cvUndistFilteredDepth.data );
            break;
        case GAUSSIAN_C1:
        	// filter out depth noise
            cv::GaussianBlur(_cvUndistDepth, cvmFilter, cv::Size(3,3), 0, 0); // filter size has to be an odd number.
	        btl::utility::filterDepth <unsigned short> ( _dThresholdDepth, (Mat_<unsigned short>)cvmFilter, (Mat_<unsigned short>*)&_cvUndistFilteredDepth );
    	    // register the depth with rgb image
    	    registration( (const unsigned short*)_cvUndistFilteredDepth.data );
            break;
        case DISPARIT_GAUSSIAN_C1:
            convert2DisparityDomain< unsigned short, double >( _cvUndistDepth, &cvDisparity );
            // apply some bilateral gaussian filtering
            cv::GaussianBlur(cvDisparity, cvFilterDisparity, cv::Size(3,3), 0, 0);
            dDispThreshold = 1./600. - 1./(600.+_dThresholdDepth);
            btl::utility::filterDepth <double> ( dDispThreshold, (Mat_<double>)cvFilterDisparity, (Mat_<double>*)&cvThersholdDisparity );
    	    btl::utility::convert2DepthDomain< double, unsigned short >( cvThersholdDisparity,(Mat_<unsigned short>*)&_cvUndistFilteredDepth );
              // register the depth with rgb image
    	    registration( (const unsigned short*)_cvUndistFilteredDepth.data );
            break;
        case NEW:
            // filter out depth noise
            //cv::GaussianBlur(_cvUndistDepth, cvmFilter, cv::Size(5,5), 0, 0); // filter size has to be an odd number.
            registration( (const unsigned short*)_cvUndistDepth.data );
            normalEstimation<double, unsigned char>( registeredDepth(), _cvUndistImage.data, _cvColor.rows, _cvColor.cols, &_vColors, &_vPts, &_vNormals );
            break;
    }

// timer off
//	_cT1 =  boost::posix_time::microsec_clock::local_time(); 
// 	_cTDAll = _cT1 - _cT0 ;
//	PRINT( _cTDAll );

	//cout << " getNextFrame() ends."<< endl;
    return;
}



} //namespace videosource
} //namespace extra
} //namespace btl
