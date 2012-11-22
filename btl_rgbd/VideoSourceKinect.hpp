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

//#define INFO

namespace btl{
namespace kinect{





//CCalibrateKinect is help to load camera parameters from 
class VideoSourceKinect //: public CCalibrateKinect
{
public:
	//type
	typedef boost::shared_ptr<VideoSourceKinect> tp_shared_ptr;
	enum tp_frame {  CPU_PYRAMID_CV, GPU_PYRAMID_CV, CPU_PYRAMID_GL, GPU_PYRAMID_GL };

	//constructor
    VideoSourceKinect(ushort uResolution_, ushort uPyrHeight_, bool bUseNIRegistration_,float fCwX_, float fCwY_, float fCwZ_);
    virtual ~VideoSourceKinect();
	void initKinect();
	void initRecorder(std::string& strPath_, ushort nTimeInSecond_);
	void initPlayer(std::string& strPathFileName_,bool bRepeat_);
	bool isPlayStop(){ return VideoSourceKinect::_bIsPlayingStop; }

	virtual void getNextFrame(tp_frame eFrameType_);
	/*void getNextPyramid(const unsigned short& uPyrHeight_, tp_frame eFrameType_)
	{
		_uPyrHeight = uPyrHeight_>4? 4:uPyrHeight_;
		getNextFrame(eFrameType_);
		return;
	}*/
    // 1. need to call getNextFrame() before hand
    // 2. RGB color channel (rather than BGR as used by cv::imread())
	void setResolution(ushort uLevel_);
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

	void record(){
		if (_bRecordSequence){
			_pCyclicBuffer->Dump();
		}
		else{
			PRINTSTR("Record functionality is not enabled.");
		}
	}

protected:
	void importYML();
	// convert the depth map/ir camera to be aligned with the rgb camera
	void alignDepthWithRGB( const cv::Mat& cvUndistortDepth_ , cv::Mat* pcvAligned_); //cv::Mat version
	void gpuAlignDepthWithRGB( const cv::gpu::GpuMat& cvUndistortDepth_ , cv::gpu::GpuMat* pcvAligned_);
	void unprojectIR ( const cv::Mat& cvmDepth_, cv::Mat* cvmIRWorld_);
	void transformIR2RGB ( const cv::Mat& cvmIRWorld, cv::Mat* pcvmRGBWorld );
	void projectRGB ( const cv::Mat& cvmRGBWorld_, cv::Mat* pcvAlignedRGB_ );
	void unprojectRGB ( const cv::Mat& cvmDepth_, int nLevel, cv::Mat* pcvmPts_, btl::utility::tp_coordinate_convention eConvention_ = btl::utility::BTL_GL );
	void fastNormalEstimation(const cv::Mat& cvmPts_, cv::Mat* pcvmNls_);
	void gpuFastNormalEstimationGL(const unsigned int& uLevel_, cv::gpu::GpuMat* pcvgmPts_, cv::gpu::GpuMat* pcvgmNls_ );
	void buildPyramid(btl::utility::tp_coordinate_convention eConvention_ );
	void gpuBuildPyramidCVm( );
	void gpuBuildPyramidUseNICVm();
	//for debug
	void findRange(const cv::Mat& cvmMat_);
	void findRange(const cv::gpu::GpuMat& cvgmMat_);
	//in playing back mode, when a sequence is finished, this function is called
	static void playEndCallback(xn::ProductionNode& node, void* pCookie_){ 
		VideoSourceKinect::_bIsPlayingStop = true;
	}
public:
	//parameters
	float _fThresholdDepthInMeter; //threshold for filtering depth
	float _fSigmaSpace; //degree of blur for the bilateral filter
	float _fSigmaDisparity; 
	unsigned int _uPyrHeight;//the height of pyramid
	ushort _uResolution;//0 640x480; 1 320x240; 2 160x120 3 80x60
	//cameras
	boost::scoped_ptr<SCamera> _pRGBCamera;
	boost::scoped_ptr<SCamera> _pIRCamera;
	boost::scoped_ptr<CKeyFrame> _pFrame;
protected:
	//openni
    xn::Context        _cContext;
    xn::ImageGenerator _cImgGen;
    xn::ImageMetaData  _cImgMD;
    xn::DepthGenerator _cDepthGen;
    xn::DepthMetaData  _cDepthMD;
	xn::Player		   _cPlayer;
	//switch sequence record mode on/off
	bool _bRecordSequence;
	bool _bPlayerIsOn;

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
	//depth pyramid (need to be initially allocated in constructor)
	

	//X,Y,Z coordinate of depth w.r.t. RGB camera reference system
	//in the format of the RGB image
	//cv::Mat		     _cvmAlignedRawDepth;//640*480
	cv::gpu::GpuMat _cvgmAlignedRawDepth;
	// temporary variables allocated in constructor and released in destructor
	// refreshed for every frame
	//cv::Mat _cvmIRWorld; //XYZ w.r.t. IR camera reference system
	//cv::Mat _cvmRGBWorld;//XYZ w.r.t. RGB camera but indexed in IR image
	//temporary file but will be faster to be allocated only once.
	cv::gpu::GpuMat _cvgmIRWorld;
	cv::gpu::GpuMat _cvgmRGBWorld;
	cv::gpu::GpuMat _cvgm32FC1Tmp;

	// the centroid of all depth point defined in RGB camera system
	// (opencv-default camera reference system convention)
	double _dXCentroid, _dYCentroid, _dZCentroid; 
	// duplicated camera parameters for speed up the VideoSourceKinect::align() in . because Eigen and cv matrix class is very slow.
	// initialized in constructor after load of the _cCalibKinect.
	float _aR[9];	// Relative rotation transpose
	float _aRT[3]; // aRT =_aR * T, the relative translation

	// Create and initialize the cyclic buffer
	CCyclicBuffer::tp_scoped_ptr _pCyclicBuffer;
	XnMapOutputMode _sModeVGA; 
	// To count missed frames for recorder
	XnUInt64 _nLastDepthTime;
	XnUInt64 _nLastImageTime;
	XnUInt32 _nMissedDepthFrames;
	XnUInt32 _nMissedImageFrames;
	XnUInt32 _nDepthFrames;
	XnUInt32 _nImageFrames;
	//controlling flag
	bool _bUseNIRegistration;
	static bool _bIsPlayingStop;
	XnCallbackHandle _handle;

};//class VideoSourceKinect

} //namespace kinect
} //namespace btl



#endif //BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCEKINECT
