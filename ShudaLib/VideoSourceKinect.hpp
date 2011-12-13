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
#include <stdexcept>
#include "calibratekinect.hpp"
#include <XnCppWrapper.h>
#include <opencv/highgui.h>
using namespace xn;

namespace btl{
namespace extra{
namespace videosource{

#define KINECT_VIDEO_W 640
#define KINECT_VIDEO_H 480

struct Frame
{
 	cv::Mat       _cvImage;
	cv::Mat 	  _cvmBW;
    cv::Mat       _cvDepth;
};

//CCalibrateKinect is help to load camera parameters from 
class VideoSourceKinect : public CCalibrateKinect
{

public:

    VideoSourceKinect();

    virtual ~VideoSourceKinect();

    void getNextFrame();

    // 1. need to call getNextFrame() before hand
    // 2. RGB color channel (rather than BGR as used by cv::imread())
    const cv::Mat&            cvRGB()     const { return  _cvUndistImage; }
    const cv::Mat*            cvRGBPtr()  const { return &_cvUndistImage; }
    const cv::Mat&            cvDepth()   const { return  _cvUndistFilteredDepth; }
    const cv::Mat*            cvDepthPtr()const { return &_cvUndistFilteredDepth; }
		  cv::Mat*            cvDepthPtr()      { return &_cvUndistFilteredDepth; }
	const cv::Mat& 			  cvBW()      const { return  _cvmUndistBW; }
	
    struct Exception : public std::runtime_error
    {
       Exception(const std::string& str) : std::runtime_error(str) {}
    };


protected:
    Eigen::Vector2i _frameSize;
    Context        _cContext;
    ImageGenerator _cImgGen;
    ImageMetaData  _cImgMD;

    DepthGenerator _cDepthGen;
    DepthMetaData  _cDepthMD;

    cv::Mat       _cvImage;
    cv::Mat       _cvDepth;
	cv::Mat       _cvUndistImage;
	cv::Mat 	  _cvUndistDepth;
	cv::Mat 	  _cvUndistFilteredDepth;
	cv::Mat 	  _cvmUndistBW;

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
