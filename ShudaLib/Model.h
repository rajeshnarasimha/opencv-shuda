#ifndef BTL_EXTRA_MODEL
#define BTL_EXTRA_MODEL

namespace btl
{
namespace extra
{
using namespace btl::utility;

class CVolume
{

}

class CModel
{
//type
public:

	typedef boost::shared_ptr<CModel> tp_shared_ptr;
private:
	//normal histogram type
	typedef std::pair< std::vector< unsigned int >, Eigen::Vector3d > tp_normal_hist_bin;
	//distance histogram type
	typedef std::pair< double,unsigned int >						  tp_pair_hist_element; 
	typedef std::pair< std::vector< tp_pair_hist_element >, double >  tp_pair_hist_bin;
	typedef std::vector< tp_pair_hist_bin >							  tp_hist;
	//mergeable flag for distance clustering
	enum tp_flag { EMPTY, NO_MERGE, MERGE_WITH_LEFT, MERGE_WITH_RIGHT, MERGE_WITH_BOTH };//methods
public:
	CModel( VideoSourceKinect& cKinect_ );
	~CModel(void);
	void detectPlaneFromCurrentFrame(const short uPyrLevel_);
protected:
	void clusterNormal(const unsigned short& uPyrLevel_,cv::Mat* pcvmLabel_,std::vector< std::vector< unsigned int > >* pvvLabelPointIdx_);

	void normalHistogram( const cv::Mat& cvmNls_, int nSamples_, std::vector< tp_normal_hist_bin >* pvNormalHistogram_);
	void distanceHistogram( const cv::Mat& cvmNls_, const cv::Mat& cvmPts_, const unsigned int& nSamples, const std::vector< unsigned int >& vIdx_, tp_hist* pvDistHist );
	void calcMergeFlag( const tp_hist& vDistHist, const double& dSampleStep, std::vector< tp_flag >* vMergeFlags );
	void mergeBins( const std::vector< tp_flag >& vMergeFlags_, const tp_hist& vDistHist_, const std::vector< unsigned int >& vLabelPointIdx_, short* pLabel_, cv::Mat* pcvmLabel_ );
	void clusterDistance( const unsigned short uPyrLevel_, const std::vector< std::vector<unsigned int> >& vvNormalClusterPtIdx_, cv::Mat* cvmDistanceClusters_ );
//data
public:
	//clusters
	boost::shared_ptr<cv::Mat>   _acvmShrPtrNormalClusters[4];
	boost::shared_ptr<cv::Mat> _acvmShrPtrDistanceClusters[4];
private:
	std::vector< std::vector< unsigned int > > _vvLabelPointIdx;
	//the minmum area of a cluster
	unsigned short _usMinArea;
	//video source
	btl::extra::videosource::VideoSourceKinect& _cKinect;
#ifdef TIMER
	//timer
	boost::posix_time::ptime _cT0, _cT1;
	boost::posix_time::time_duration _cTDAll;
	float _fFPS;//frame per second
#endif
};




}//extra
}//btl
#endif

