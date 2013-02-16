#ifndef SIMPLE_TRACKER_FREAK_BTL
#define SIMPLE_TRACKER_FREAK_BTL


namespace btl{	namespace image	{
using namespace btl::image::semidense;

class CTrackerSimpleFreak: public CSemiDenseTracker{
public:
	//type definition
	typedef boost::scoped_ptr<CTrackerSimpleFreak> tp_scoped_ptr;
	typedef boost::shared_ptr<CTrackerSimpleFreak> tp_shared_ptr;

	CTrackerSimpleFreak(unsigned int uPyrHeight_);

	int _nFrameIdx;
	unsigned int _uPyrHeight;

	//virtual bool initialize( boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrBW[4] );
	bool initialize( boost::shared_ptr<cv::Mat> _acvmShrPtrPyrBW[4] );
	bool initialize( boost::shared_ptr<cv::gpu::GpuMat> acgvmShrPtrPyrBW_[4], const cv::Mat& cvmMaskCurr_ );
	//virtual void track(boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrBW[4] );
	void track(boost::shared_ptr<cv::Mat> _acvmShrPtrPyrBW[4] );
	void track( const Eigen::Matrix3f& eimHomoInit_, boost::shared_ptr<cv::gpu::GpuMat> acvgmShrPtrPyrBW_[4], const cv::Mat& cvmMaskCurr_, Eigen::Matrix3f* peimHomo_ );

	void displayCandidates( cv::Mat& cvmColorFrame_ );
	virtual void display(cv::Mat& cvmColorFrame_);
	cv::Mat calcHomography(const cv::Mat& cvmMaskCurr_, const cv::Mat& cvmMaskPrev_);
	void extractHomography(const cv::gpu::GpuMat& cvgmBuffer_,Eigen::Matrix3f* peimDeltaHomo_);


	boost::scoped_ptr<cv::SURF> _pSurf;
	boost::scoped_ptr<cv::FREAK> _pFreak;
	std::vector<cv::KeyPoint> _vKeypoints1;
	std::vector<cv::KeyPoint> _vKeypointsPrev;
	std::vector<cv::KeyPoint> _vKeypointsCurr;
	cv::Mat _cvmDescriptor1;
	cv::Mat _cvmDescriptorPrev;
	cv::Mat _cvmDescriptorCurr;

	cv::BruteForceMatcher<cv::HammingLUT> _cMatcher;  
	std::vector<cv::DMatch> _vMatches;

	unsigned short _usTotal;

	//full frame alignment
	boost::shared_ptr<cv::gpu::GpuMat> _acgvmShrPtrPyrBWPrev[4];
	cv::Mat _cvmMaskPrev;


};//class CSemiDenseTracker

}//image
}//btl

#endif