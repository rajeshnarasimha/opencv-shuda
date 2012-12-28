#ifndef ORB_SHUDA
#define ORB_SHUDA


namespace btl
{
namespace image
{
////////////////////////////////// ORB //////////////////////////////////////////

class CORB
{
public:
	enum
	{
		X_ROW = 0,
		Y_ROW,
		RESPONSE_ROW,
		ANGLE_ROW,
		OCTAVE_ROW,
		SIZE_ROW,
		ROWS_COUNT
	};

	enum
	{
		DEFAULT_FAST_THRESHOLD = 20
	};

	//! Constructor
	explicit CORB(int nFeatures = 500, float scaleFactor = 1.2f, int nLevels = 8, int edgeThreshold = 31,
		int firstLevel = 0, int WTA_K = 2, int scoreType = 0, int patchSize = 31);

	//! Compute the ORB features on an image
	//! image - the image to compute the features (supports only CV_8UC1 images)
	//! mask - the mask to apply
	//! keypoints - the resulting keypoints
	void operator()(const cv::gpu::GpuMat& cvgmImage_, const cv::gpu::GpuMat& cvgmMask_, std::vector<cv::KeyPoint>* pvKeypoints_);
	void operator()(const cv::gpu::GpuMat& cvgmImage_, const cv::gpu::GpuMat& cvgmMask_, cv::gpu::GpuMat* pcvgmKeypoints_);

	//! Compute the ORB features and descriptors on an image
	//! image - the image to compute the features (supports only CV_8UC1 images)
	//! mask - the mask to apply
	//! keypoints - the resulting keypoints
	//! descriptors - descriptors array
	void operator()(const cv::gpu::GpuMat& cvgmImage_, const cv::gpu::GpuMat& cvgmMask_, std::vector<cv::KeyPoint>* pvKeypoints_, cv::gpu::GpuMat* pcvgmDescriptors_);
	void operator()(const cv::gpu::GpuMat& cvgmImage_, const cv::gpu::GpuMat& cvgmMask_, cv::gpu::GpuMat* pcvgmKeypoints, cv::gpu::GpuMat* pcvgmDescriptors_);
	//! download keypoints from device to host memory
	static void downloadKeyPoints(const cv::gpu::GpuMat &cvgmKeypoints_, std::vector<cv::KeyPoint>* pvKeypoints_);
	//! convert keypoints to KeyPoint vector
	static void convertKeyPoints(const cv::Mat &cvmKeypoints_, std::vector<cv::KeyPoint>* pvKeypoints_);
	//! returns the descriptor size in bytes
	inline int descriptorSize() const { return kBytes; }

	inline void setFastParams(int nThreshold_, bool bNonMaxSupression = true)
	{
		_fastDetector._nThreshold = nThreshold_;
		_fastDetector._bNonMaxSupression = bNonMaxSupression;
	}

	//! release temporary buffer's memory
	void release();

	//! if true, image will be blurred before descriptors calculation
	bool _bBlurForDescriptor;

private:
	enum { kBytes = 32 };

	void buildScalePyramids(const cv::gpu::GpuMat& image, const cv::gpu::GpuMat& mask);
	//calculate the fast corner as key points and compute the angle
	void computeKeyPointsPyramid();

	void computeDescriptors(cv::gpu::GpuMat* pcvgmDescriptors_);
	//convert the location of keypoints in various scales into the first scale
	void mergeKeyPoints(cv::gpu::GpuMat* pcvgmKeyPoints_);
	int _nFeatures;
	float _fScaleFactor;
	int _nLevels;
	int _nEdgeThreshold;
	int _nFirstLevel;
	int _nWTA_K;
	int _nScoreType;
	int _nPatchSize;

	// The number of desired features per scale
	std::vector<size_t> _vFeaturesPerLevel;

	// Points to compute BRIEF descriptors from
	cv::gpu::GpuMat _cvgmPattern;

	std::vector<cv::gpu::GpuMat> _vcvgmImagePyr;
	std::vector<cv::gpu::GpuMat> _vcvgmMaskPyr;

	cv::gpu::GpuMat _cvgmBuf;

	std::vector<cv::gpu::GpuMat> _vcvgmKeyPointsPyr;
	std::vector<int> _vKeyPointsCount;

	btl::image::CFast _fastDetector;

	cv::Ptr<cv::gpu::FilterEngine_GPU> _pBlurFilter;

	cv::gpu::GpuMat _cvgmKeypoints;
};

}//namespace image
}//namespace btl

#endif