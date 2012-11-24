#ifndef BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCEKINECTSIMULATOR
#define BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCEKINECTSIMULATOR
/**
* @file VideoSourceKinect.hpp
* @brief APIs for load of data from a kinect devices
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.1 
* 3-17 depth generator added
* @date 2012-11-08
*/

//#define INFO

namespace btl{
namespace kinect{

//CCalibrateKinect is help to load camera parameters from 
class VideoSourceKinectSimulator: public VideoSourceKinect
{
public:
	//type
	typedef boost::shared_ptr<VideoSourceKinectSimulator> tp_shared_ptr;

	//constructor
    VideoSourceKinectSimulator(ushort uResolution_, ushort uPyrHeight_, bool bUseNIRegistration_,const Eigen::Vector3f& eivCw_);
    virtual ~VideoSourceKinectSimulator();

	virtual void getNextFrame(tp_frame eFrameType_, int* pnRecordingStatus_);
	void processZBuffer(const cv::Mat& cvmDepth_, cv::Mat* pcvmDepthImg_ ) const;
	void setSensorDepthRange() const;
	void exportRawDepth() const;
	void captureScreen();
private:
	float _fNear, _fFar; // Near and Far clipping plane.

};

} //namespace kinect
} //namespace btl



#endif //BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCEKINECT
