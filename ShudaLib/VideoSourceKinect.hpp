#ifndef BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCEKINECT
#define BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCEKINECT
/**
* @file VideoSourceKinect.hpp
* @brief APIs for load of data from a kinect devices
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.1 
* 3-17 depth generator added
* @date 2011-03-17
*/

//turn on timer
#define TIMER 
//#define INFO

#include "calibratekinect.hpp"
#include <XnCppWrapper.h>
#include <opencv/highgui.h>
#include <opencv2/gpu/gpu.hpp>
namespace btl{
namespace extra{
namespace videosource{

#define KINECT_WIDTH 640
#define KINECT_HEIGHT 480


#define KINECT_WxH 307200
#define KINECT_WxH_L1 76800 //320*240
#define KINECT_WxH_L2 19200 //160*120
#define KINECT_WxH_L3 4800  // 80*60

#define KINECT_WxHx3 921600
#define KINECT_WxHx3_L1 230400 
#define KINECT_WxHx3_L2 57600

static unsigned int __aKinectWxH[4] = {KINECT_WxH,KINECT_WxH_L1,KINECT_WxH_L2,KINECT_WxH_L3};

//CCalibrateKinect is help to load camera parameters from 
class VideoSourceKinect : public CCalibrateKinect
{
public:
    VideoSourceKinect();
    virtual ~VideoSourceKinect();

    void getNextFrame();

    // 1. need to call getNextFrame() before hand
    // 2. RGB color channel (rather than BGR as used by cv::imread())
    const cv::Mat&            cvRGB()     const { return  _cvmUndistRGB; }
	const cv::Mat& 			  cvBW()      const { return  _cvmUndistBW; }
	const double*		alignedDepth()    const { return  _pRGBWorldRGB; }

	void cloneFrame(cv::Mat* pcvmRGB_, cv::Mat* pcvmDepth_);
	void clonePyramid(std::vector<cv::Mat>* pvcvmRGB_, std::vector<cv::Mat>* pvcvmDepth_);
	//opencv convention
	void centroid( Eigen::Vector3d* peivCentroid_ ) const 
	{
		(*peivCentroid_)(0) = _dXCentroid;
		(*peivCentroid_)(1) = _dYCentroid;
		(*peivCentroid_)(2) = _dZCentroid;
	}
	//opengl convention
	void centroidGL( Eigen::Vector3d* peivCentroid_ ) const 
	{
		(*peivCentroid_)(0) =   _dXCentroid;
		(*peivCentroid_)(1) = - _dYCentroid;
		(*peivCentroid_)(2) = - _dZCentroid;
	}

	// convert the depth map/ir camera to be aligned with the rgb camera
	void align( const cv::Mat& cvUndistortDepth_ );
	void unprojectIR( const unsigned short* pCamera_,const int& nN_, double* pWorld_ );
	void transformIR2RGB( const double* pIR_,const int& nN_, double* pRGB_ );
	void projectRGB ( double* pWorld_, const int& nN_, double* pRGBWorld_, cv::Mat* pDepthL1_ );
	void unprojectRGB ( const cv::Mat& cvmDepth_, double* pWorld_, int nLevel = 0 );
	//un-project individual depth
	void unprojectRGBGL ( const cv::Mat& cvmDepth_, const int& r,const int& c, double* pWorld_, int nLevel /*= 0*/ ); 

protected:
	//openni
    xn::Context        _cContext;
    xn::ImageGenerator _cImgGen;
    xn::ImageMetaData  _cImgMD;
    xn::DepthGenerator _cDepthGen;
    xn::DepthMetaData  _cDepthMD;
	//rgb
    cv::Mat       _cvmRGB;
	cv::Mat       _cvmUndistRGB;
	cv::gpu::GpuMat _cvgmRGB;
	cv::gpu::GpuMat _cvgmUndistRGB;
	//depth
    cv::Mat       _cvmDepth;
	cv::Mat 	  _cvmUndistDepth;
	//rgb pyramid
	std::vector< cv::Mat > _vcvmPyramidRGBs;

	//depth pyramid (need to be initially allocated in constructor)
	std::vector< cv::Mat > _vcvmPyramidDepths;
	cv::Mat		  _cvmAlignedDepthL0;//640*480
	//black and white
	cv::Mat 	  _cvmUndistBW;
	// temporary variables allocated in constructor and released in destructor
	unsigned short* _pPxDIR; //2D coordinate along with depth for ir image, column-major
	double*  _pIRWorld;      //XYZ w.r.t. IR camera reference system 
	double*  _pRGBWorld;     //XYZ w.r.t. RGB camera but indexed in IR image
	// refreshed for every frame
	//X,Y,Z coordinate of depth w.r.t. RGB camera reference system
	//in the format of the RGB image
	double*  _pRGBWorldRGB; //(need to be initially allocated in constructor)
	// the centroid of all depth point defined in RGB camera system
	// (opencv-default camera reference system convention)
	double _dXCentroid, _dYCentroid, _dZCentroid; 

public:
	//parameters
	double _dThresholdDepth; //threshold for filtering depth
	double _dSigmaSpace; //degree of blur for the bilateral filter
	double _dSigmaDisparity;
	unsigned int _uPyrHeight;//the height of pyramid
	enum {  RAW, GAUSSIAN, GAUSSIAN_C1, GAUSSIAN_C1_FILTERED_IN_DISPARTY, 
		BILATERAL_FILTERED_IN_DISPARTY, PYRAMID_BILATERAL_FILTERED_IN_DISPARTY } _ePreFiltering;
#ifdef TIMER
	//timer
	boost::posix_time::ptime _cT0, _cT1;
	boost::posix_time::time_duration _cTDAll;
#endif
};

} //namespace videosource
} //namespace extra
} //namespace btl

namespace btl{
namespace extra{

using videosource::VideoSourceKinect;

} //namespace extra
} //namespace btl


#endif //BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCEKINECT
