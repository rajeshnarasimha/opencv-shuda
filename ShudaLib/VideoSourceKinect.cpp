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
    _eMethod = NEW_BILATERAL; 
    
	_dSigmaSpace = 2;

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
	_vColors.clear();
	_vPts.clear();
	_vNormals.clear();
	
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

    cv::Mat cvDisparity( _cvUndistDepth.rows, _cvUndistDepth.cols, CV_32F );
    cv::Mat_<float> cvFilterDisparity( _cvUndistDepth.rows, _cvUndistDepth.cols, CV_32F );
    cv::Mat cvThersholdDisparity( _cvUndistDepth.rows, _cvUndistDepth.cols, CV_32F );

    cv::Mat_<unsigned short> cvmFilter(_cvUndistDepth.rows, _cvUndistDepth.cols, CV_16U );
    double dDispThreshold;

    switch( _eMethod )
    {
		case NONE: //default
			registration( (const unsigned short*)_cvUndistDepth.data );
			normalEstimationGL<double, unsigned char>( registeredDepth(), _cvUndistImage.data, _cvUndistImage.rows, _cvUndistImage.cols, &_vColors, &_vPts, &_vNormals );
			break;
        case RAW:
    	    // register the depth with rgb image
    	    registration( (const unsigned short*)_cvUndistDepth.data );
			normalEstimationGLPCL<double, unsigned char>( registeredDepth(), _cvUndistImage.data, _cvUndistImage.rows, _cvUndistImage.cols, &_vColors, &_vPts, &_vNormals );
            break;
        case C1_CONTINUITY:
            btl::utility::filterDepth <unsigned short> ( _dThresholdDepth, (cv::Mat_<unsigned short>)_cvUndistDepth, (cv::Mat_<unsigned short>*)&_cvUndistFilteredDepth );
    	    // register the depth with rgb image
    	    registration( (const unsigned short*)_cvUndistFilteredDepth.data );
			normalEstimationGLPCL<double, unsigned char>( registeredDepth(), _cvUndistImage.data, _cvUndistImage.rows, _cvUndistImage.cols, &_vColors, &_vPts, &_vNormals );
            break;
        case GAUSSIAN_C1:
        	// filter out depth noise
            cv::GaussianBlur(_cvUndistDepth, cvmFilter, cv::Size(0,0), _dSigmaSpace, _dSigmaSpace); // filter size has to be an odd number.
	        btl::utility::filterDepth <unsigned short> ( _dThresholdDepth, (cv::Mat_<unsigned short>)cvmFilter, (cv::Mat_<unsigned short>*)&_cvUndistFilteredDepth );
    	    // register the depth with rgb image
    	    registration( (const unsigned short*)_cvUndistFilteredDepth.data );
			normalEstimationGLPCL<double, unsigned char>( registeredDepth(), _cvUndistImage.data, _cvUndistImage.rows, _cvUndistImage.cols, &_vColors, &_vPts, &_vNormals );
            break;
        case DISPARIT_GAUSSIAN_C1:
            convert2DisparityDomain< unsigned short >( _cvUndistDepth, &(cv::Mat_<float>)cvDisparity );
            cv::GaussianBlur(cvDisparity, cvFilterDisparity, cv::Size(0,0), _dSigmaSpace, _dSigmaSpace);
            dDispThreshold = 1./600. - 1./(600.+_dThresholdDepth);
            btl::utility::filterDepth <float> ( dDispThreshold, ( cv::Mat_<float>)cvFilterDisparity, ( cv::Mat_<float>*)&cvThersholdDisparity );
    	    btl::utility::convert2DepthDomain< unsigned short >( cvThersholdDisparity,( cv::Mat_<unsigned short>*)&_cvUndistFilteredDepth );
              // register the depth with rgb image
    	    registration( (const unsigned short*)_cvUndistFilteredDepth.data );
			normalEstimationGLPCL<double, unsigned char>( registeredDepth(), _cvUndistImage.data, _cvUndistImage.rows, _cvUndistImage.cols, &_vColors, &_vPts, &_vNormals );
            break;
        case NEW_GAUSSIAN:
            // filter out depth noise
			// apply some bilateral gaussian filtering
            cv::GaussianBlur(_cvUndistDepth, cvmFilter, cv::Size(0,0), _dSigmaSpace, _dSigmaSpace); // filter size has to be an odd number.
            registration( (const unsigned short*)cvmFilter.data );
            normalEstimationGL<double, unsigned char>( registeredDepth(), _cvUndistImage.data, _cvUndistImage.rows, _cvUndistImage.cols, &_vColors, &_vPts, &_vNormals );
            break;
		case NEW_BILATERAL:
			// filter out depth noise
			// apply some bilateral gaussian filtering
			btl::utility::convert2DisparityDomain< unsigned short >( _cvUndistDepth, &(cv::Mat_<float>)cvDisparity );
			dDispThreshold = 1./600. - 1./(600.+_dThresholdDepth);
			cv::bilateralFilter(cvDisparity, cvThersholdDisparity,0, dDispThreshold, _dSigmaSpace); // filter size has to be an odd number.
			PRINT(_dThresholdDepth);
			PRINT(dDispThreshold);
			btl::utility::convert2DepthDomain< unsigned short >( cvThersholdDisparity,&(cv::Mat_<unsigned short>)_cvUndistFilteredDepth );
			registration( (const unsigned short*)_cvUndistFilteredDepth.data );
			normalEstimationGL<double, unsigned char>( registeredDepth(), _cvUndistImage.data, _cvUndistImage.rows, _cvUndistImage.cols, &_vColors, &_vPts, &_vNormals );
			break;
		case NEW_DEPTH:
			btl::utility::clearMat<float>(0,&_cvmDepthRGB);
			registration( (const unsigned short*)_cvUndistDepth.data ); //generate _cvmDepthRGBL1
		//bilateral filtering in disparity domain
			btl::utility::convert2DisparityDomain< float >( _cvmDepthRGB, &(cv::Mat_<float>)cvDisparity );
			dDispThreshold = 1./600 - 1./(600+_dThresholdDepth);
			PRINT(dDispThreshold);
			PRINT(_dSigmaSpace);
			cv::bilateralFilter(cvDisparity, cvThersholdDisparity,0, dDispThreshold, _dSigmaSpace); // filter size has to be an odd number.
			btl::utility::clearMat<float>(0,&_cvmDepthRGBL0);
			btl::utility::convert2DepthDomain< float >( cvThersholdDisparity,&(cv::Mat_<float>)_cvmDepthRGBL0 );
		//get normals L0
			//unprojectRGB ( _cvmDepthRGBL0, _pRGBWorldRGB );
			//estimate normal using fast method
			//normalEstimationGL<double, unsigned char>( registeredDepth(), _cvUndistImage.data, _cvUndistImage.rows, _cvUndistImage.cols, &_vColors, &_vPts, &_vNormals );
		//downsampling to get level 1
			btl::utility::clearMat<float>(0,&_cvmDepthRGBL1);
			btl::utility::downSampling<float>(_cvmDepthRGBL0,&_cvmDepthRGBL1);
		//get normals L1
			unprojectRGB ( _cvmDepthRGBL1, _pRGBWorldRGBL1, 1 );//float to double
			cv::pyrDown(_cvUndistImage,_cvmUndistDepthL1);
			normalEstimationGL<double, unsigned char>( _pRGBWorldRGBL1, _cvmUndistDepthL1.data, _cvmUndistDepthL1.rows, _cvmUndistDepthL1.cols, &_vColors, &_vPts, &_vNormals );
			//downsampling to get level 1
			/*
		//bilateral filtering in disparity domain
			btl::utility::convert2DisparityDomain< float >( _cvmDepthRGB, &(cv::Mat_<float>)cvDisparity );
			dDispThreshold = 1./600 - 1./(600+_dThresholdDepth);
			PRINT(dDispThreshold);
			PRINT(_dSigmaSpace);
			cv::bilateralFilter(cvDisparity, cvThersholdDisparity,0, dDispThreshold, _dSigmaSpace); // filter size has to be an odd number.
			btl::utility::clearMat<float>(0,&_cvmDepthRGBL0);
			btl::utility::convert2DepthDomain< float, float >( cvThersholdDisparity,&(cv::Mat_<float>)_cvmDepthRGBL0 );
		//downsampling to get level 2
			btl::utility::clearMat<float>(0,&_cvmDepthRGBL1);
			btl::utility::downSampling<float>(_cvmDepthRGBL1,&_cvmDepthRGBL2);
			unprojectRGB ( _cvmDepthRGBL1, _pRGBWorldRGBL1, 1 );//float to double
			cv::pyrDown(_cvUndistImage,_cvmUndistDepthL1);
			normalEstimationGL<double, unsigned char>( _pRGBWorldRGBL1, _cvmUndistDepthL1.data, _cvmUndistDepthL1.rows, _cvmUndistDepthL1.cols, &_vColors, &_vPts, &_vNormals );
			*/
			break;
    }

// timer off
//	_cT1 =  boost::posix_time::microsec_clock::local_time(); 
// 	_cTDAll = _cT1 - _cT0 ;
//	PRINT( _cTDAll );

	//cout << " getNextFrame() ends."<< endl;
    return;
}

void VideoSourceKinect::unprojectRGB ( const cv::Mat& cvmDepth_, double* pWorld_, int nLevel /*= 0*/ )
{
	double* pM = pWorld_ ;

	// initialize the Registered depth as NULLs
	int nN = cvmDepth_.rows*cvmDepth_.cols;
	for ( int i = 0; i < nN; i++ )
	{
		*pM++ = 0;
		*pM++ = 0;
		*pM++ = 0;
	}
	// pCamer format
	// 0 x (c) 1 y (r) 2 d
	//the pixel coordinate is defined w.r.t. camera reference, which is defined as x-left, y-downward and z-forward. It's
	//a right hand system. i.e. opencv-default reference system;
	//unit is meter
	//when rendering the point using opengl's camera reference which is defined as x-left, y-upward and z-backward. the
	//for example: glVertex3d ( Pt(0), -Pt(1), -Pt(2) ); i.e. opengl-default reference system
	int nScale = 1 << nLevel;

	CHECK( CV_32FC1 == cvmDepth_.type(), "VideoSourceKinect::unprojectRGB() cvmDepth_ must be CV_32FC1" );
	float *pDepth = (float*) cvmDepth_.data;
	
	for ( unsigned int r = 0; r < cvmDepth_.rows; r++ )
	for ( unsigned int c = 0; c < cvmDepth_.cols; c++ )
	{
		* ( pWorld_ + 2 ) = cvmDepth_.at<float>(r,c);//*pDepth++;
		* ( pWorld_ + 2 ) /= 1000.;
		//coordinate system is defined w.r.t. the camera plane which is 0.5 centimeters in front of the camera center
		* pWorld_		  = ( c*nScale - _uRGB ) / _dFxRGB * *( pWorld_ + 2 ); // + 0.0025;     //x by experience.
		* ( pWorld_ + 1 ) = ( r*nScale - _vRGB ) / _dFyRGB * *( pWorld_ + 2 ); // - 0.00499814; //y the value is esimated using CCalibrateKinectExtrinsics::calibDepth(

		pWorld_ += 3;
	}

	return;
}


} //namespace videosource
} //namespace extra
} //namespace btl
