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

//#define INFO
#include <boost/shared_ptr.hpp>

#include "calibratekinect.hpp"
#include <XnCppWrapper.h>
#include <opencv/highgui.h>

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
	//type
	typedef boost::shared_ptr<VideoSourceKinect> tp_shared_ptr;
	enum tp_frame {  GPU_RAW, CPU_RAW, PYRAMID_BILATERAL_FILTERED_IN_DISPARTY } _ePreFiltering;
	//constructor
    VideoSourceKinect();
    virtual ~VideoSourceKinect();


	void getNextFrame(tp_frame ePreFiltering_);
	void getNextPyramid(const unsigned short& uPyrHeight_)
	{
		_uPyrHeight = uPyrHeight_>4? 4:uPyrHeight_;
		getNextFrame(PYRAMID_BILATERAL_FILTERED_IN_DISPARTY);
		return;
	}
    // 1. need to call getNextFrame() before hand
    // 2. RGB color channel (rather than BGR as used by cv::imread())
    //const cv::Mat&           cvRGB()     const { return  _vcvmPyrRGBs[0]; }
	//const double*		alignedDepth()    const { return  _pRGBWorldRGB; }

	void cloneRawFrame(cv::Mat* pcvmRGB_, cv::Mat* pcvmDepth_);
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
	void alignDepthWithRGB2( const cv::Mat& cvUndistortDepth_ , cv::Mat* pcvAligned_); //cv::Mat version
	void gpuAlignDepthWithRGB( const cv::gpu::GpuMat& cvUndistortDepth_ , cv::gpu::GpuMat* pcvAligned_);
	void unprojectIR ( const cv::Mat& cvmDepth_, cv::Mat* cvmIRWorld_);
	void transformIR2RGB ( const cv::Mat& cvmIRWorld, cv::Mat* pcvmRGBWorld );
	void projectRGB ( const cv::Mat& cvmRGBWorld_, cv::Mat* pcvAlignedRGB_ );
	void unprojectRGBGL ( const cv::Mat& cvmDepth_, int nLevel, cv::Mat* pcvmPts_ );
	void findRange(const cv::Mat& cvmMat_);
	void findRange(const cv::gpu::GpuMat& cvgmMat_);
	void fastNormalEstimationGL(const cv::Mat& cvmPts_, cv::Mat* pcvmNls_);
	void gpuFastNormalEstimationGL(const unsigned int& uLevel_, cv::gpu::GpuMat* pcvgmPts_, cv::gpu::GpuMat* pcvgmNls_ );

public:
	//openni
    xn::Context        _cContext;
    xn::ImageGenerator _cImgGen;
    xn::ImageMetaData  _cImgMD;
    xn::DepthGenerator _cDepthGen;
    xn::DepthMetaData  _cDepthMD;
	//rgb
    cv::Mat			_cvmRGB;
	cv::gpu::GpuMat _cvgmRGB;
	cv::Mat         _cvmUndistRGB;
	cv::gpu::GpuMat _cvgmUndistRGB;
	//depth
    cv::Mat         _cvmDepth;
	cv::gpu::GpuMat _cvgmDepth;
	cv::Mat         _cvmUndistDepth;
	cv::gpu::GpuMat _cvgmUndistDepth;
	//rgb pyramid
	std::vector< cv::Mat > _vcvmPyrRGBs;
	//depth pyramid (need to be initially allocated in constructor)
	std::vector< cv::Mat > _vcvmPyrDepths;
	//gpu
	std::vector< cv::gpu::GpuMat > _vcvgmPyrDepths;
	std::vector< cv::gpu::GpuMat > _vcvgmPyrDisparity;
	std::vector< cv::gpu::GpuMat > _vcvgmPyr32FC1Tmp;
	std::vector< cv::gpu::GpuMat > _vcvgmPyrRGBs;
	std::vector< cv::gpu::GpuMat > _vcvgmPyrPts;
	std::vector< cv::gpu::GpuMat > _vcvgmPyrNls;

	boost::shared_ptr<cv::Mat> _acvmShrPtrPyrPts[4]; //using pointer array is because the vector<cv::Mat> has problem when using it &vMat[0] in calling a function
	boost::shared_ptr<cv::Mat> _acvmShrPtrPyrNls[4]; //CV_32FC3 type
	
	//X,Y,Z coordinate of depth w.r.t. RGB camera reference system
	//in the format of the RGB image
	cv::Mat		     _cvmAlignedRawDepth;//640*480
	cv::gpu::GpuMat _cvgmAlignedRawDepth;
	// temporary variables allocated in constructor and released in destructor
	// refreshed for every frame
	cv::Mat _cvmIRWorld; //XYZ w.r.t. IR camera reference system
	cv::Mat _cvmRGBWorld;//XYZ w.r.t. RGB camera but indexed in IR image
	//temporary file but will be faster to be allocated only once.
	cv::gpu::GpuMat _cvgmIRWorld,_cvgmRGBWorld;
	cv::gpu::GpuMat _cvgm32FC1Tmp;

	// the centroid of all depth point defined in RGB camera system
	// (opencv-default camera reference system convention)
	double _dXCentroid, _dYCentroid, _dZCentroid; 

	//parameters
	float _fThresholdDepthInMeter; //threshold for filtering depth
	float _fSigmaSpace; //degree of blur for the bilateral filter
	float _fSigmaDisparity; 
	unsigned int _uPyrHeight;//the height of pyramid

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
