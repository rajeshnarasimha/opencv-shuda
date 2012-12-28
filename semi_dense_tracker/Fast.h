#ifndef FAST_SHUDA
#define FAST_SHUDA

namespace btl
{
namespace image
{

class CFast
{
public:
	enum
	{
		LOCATION_ROW = 0,
		RESPONSE_ROW,
		ROWS_COUNT
	};

	// all features have same size
	static const int FEATURE_SIZE = 7;

	explicit CFast(int threshold, bool nonmaxSupression = true, double keypointsRatio = 0.05);

	//! finds the keypoints using FAST detector
	//! supports only CV_8UC1 images
	void operator ()(const cv::gpu::GpuMat& cvgmImage_, const cv::gpu::GpuMat& cvgmMask_, cv::gpu::GpuMat* pcvgmKeyPoints_);
	void operator ()(const cv::gpu::GpuMat& cvgmImage_, const cv::gpu::GpuMat& cvgmMask_, std::vector<cv::KeyPoint>* pvKeyPoints_);
	//! download keypoints from device to host memory
	static void downloadKeypoints(const cv::gpu::GpuMat& cvgmKeyPoints_, std::vector<cv::KeyPoint>* pvKeyPoints_);
	//! convert keypoints to KeyPoint vector
	static void convertKeypoints(const cv::Mat& cvmKeyPoints_, std::vector<cv::KeyPoint>* pvKeyPoints_);
	//! release temporary buffer's memory
	void release();

	bool _bNonMaxSupression;

	int _nThreshold;

	//! max keypoints = _dKeyPointsRatio * img.size().area()
	double _dKeyPointsRatio;

	//! find keypoints and compute it's response if _bNonMaxSupression is true
	//! return count of detected keypoints
	int calcKeyPointsLocation(const cv::gpu::GpuMat& cvgmImage_, const cv::gpu::GpuMat& cvgmMask_);

	//! get final array of keypoints
	//! performs nonmax supression if needed
	//! return final count of keypoints
	int getKeyPoints(cv::gpu::GpuMat* pcvgmKeyPoints_);

private:
	unsigned int _uCount;
	cv::gpu::GpuMat _cvgmKeyPointLocation;
	cv::gpu::GpuMat _cvgmScore;
	cv::gpu::GpuMat _cvgmdKeyPoints;
};

}//namespace image
}//namespace btl

#endif
