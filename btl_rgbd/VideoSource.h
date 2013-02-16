#ifndef BTL_VIDEOSOURCE
#define BTL_VIDEOSOURCE
/**
* @file VideoSource.h
* @brief APIs for load of data from a video file or a webcamera
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.1 
* @date 2013-02-08
*/

namespace btl{ namespace video
{
class VideoSource
{
public:
	//type
	typedef boost::shared_ptr<VideoSource> tp_shared_ptr;
	enum tp_mode { SIMPLE_CAPTURING = 1, RECORDING = 2, PLAYING_BACK = 3};
	enum tp_status { CONTINUE=01, PAUSE=02, MASK1 =07, START_RECORDING=010, STOP_RECORDING=020, CONTINUE_RECORDING=030, DUMP_RECORDING=040, MASK_RECORDER = 070 };
	//constructor
	VideoSource(const std::string& strCameraParam_, ushort uResolution_, ushort uPyrHeight_,const Eigen::Vector3f& eivCw_ );
	virtual ~VideoSource();
	virtual void init();
	void initRecorder(std::string& strPath_, ushort nTimeInSecond_);
	void initPlayer(std::string& strPathFileName_,bool bRepeat_);
	// 1. need to call getNextFrame() before hand
	// 2. RGB color channel (rather than BGR as used by cv::imread())
	virtual void getNextFrame(int* pnStatus_);

	void setDumpFileName( const std::string& strFileName_ ){_strDumpFileName = strFileName_;}
	//************************************
	// Method:    setSize
	// FullName:  btl::video::VideoSource::setSize
	// Access:    public 
	// Returns:   void
	// Qualifier:
	// Parameter: const float & fSize_ range [1.0, 0.0) is the original_image_size * fSize = new_image_size
	//************************************
	void setSize( const float& fScale_ ) { _fScale = fScale_; }

protected:

	void getNextFrameRecording( int* pnStatus_, float* pfTimeLeft_);
	void getNextFrameNormal(int* pnStatus_);

	void importYML();
	// convert the depth map/ir camera to be aligned with the rgb camera
	void gpuBuildPyramid();
	void gpuBuildPyramidCVm();
public:
	//parameters
	unsigned int _uPyrHeight;//the height of pyramid
	unsigned int _uResolution;
	unsigned int _uFrameIdx;
	float _fScale;
	std::string _strVideoFileName;

	//cameras
	btl::image::SCamera::tp_scoped_ptr _pCamera;
	boost::scoped_ptr<cv::VideoCapture> _pVideo;
	boost::scoped_ptr<btl::kinect::CKeyFrame> _pCurrFrame;
protected:
	//opencv
	//rgb
	cv::Mat			_cvmRGB;
	cv::gpu::GpuMat _cvgmRGB;
	cv::Mat         _cvmUndistRGB;
	cv::gpu::GpuMat _cvgmUndistRGB;

	// duplicated camera parameters for speed up the VideoSourceKinect::align() in . because Eigen and cv matrix class is very slow.
	// initialized in constructor after load of the _cCalibKinect.
	float _aR[9];	// Relative rotation transpose
	float _aRT[3]; // aRT =_aR * T, the relative translation

	//controlling flag
	static bool _bIsSequenceEnds;
	std::string _strDumpFileName;
	std::string _strCameraParam;
	int _nMode; 

	Eigen::Vector3f _eivCw;
};//class VideoSourceKinect

}//namespace video
}//namespace btl

#endif
