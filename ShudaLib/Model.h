#ifndef BTL_EXTRA_MODEL
#define BTL_EXTRA_MODEL

namespace btl
{
namespace extra
{
using namespace btl::utility;

class CModel
{
public:
	//type
	typedef boost::shared_ptr<CModel> tp_shared_ptr;
private:
	//normal histogram type
	typedef std::pair< std::vector< unsigned int >, Eigen::Vector3d > tp_normal_hist_bin;
	//distance histogram type
	typedef std::pair< double,unsigned int >						  tp_pair_hist_element; 
	typedef std::pair< std::vector< tp_pair_hist_element >, double >  tp_pair_hist_bin;
	typedef std::vector< tp_pair_hist_bin >							  tp_hist;

	enum tp_flag { EMPTY, NO_MERGE, MERGE_WITH_LEFT, MERGE_WITH_RIGHT, MERGE_WITH_BOTH };
public:
	CModel( VideoSourceKinect& cKinect_ );
	~CModel(void);
	//loaders
	void storeCurrentFrame(); //load depth and rgb from video source and convert it to point cloud data
	void loadPyramid(); //load pyramid
	void convert2PointCloudModelGL(const cv::Mat& cvmDepth_,const cv::Mat& cvmRGB_, unsigned int uLevel_, std::vector<const unsigned char*>* vColor_, 
		std::vector<Eigen::Vector3d>* vPt_, std::vector<Eigen::Vector3d>* vNormal_, 
		std::vector< int >* pvX_=NULL, std::vector< int >* pvY_=NULL);
	void detectPlanePCL(unsigned int uLevel_,std::vector<int>* pvXIdx_, std::vector<int>* pvYIdx_);
	void detectPlaneFromCurrentFrame(const short uPyrLevel_);
	void loadPyramidAndDetectPlanePCL();
	//extract a plane from depth map and convert to GL convention point cloud
	void extractPlaneGL(unsigned int uLevel_, const std::vector<int>& vX_, const std::vector<int>& vY_, std::vector<Eigen::Vector3d>* pvPlane_);
	void clusterNormal(const unsigned short& uPyrLevel_,cv::Mat* pcvmLabel_,std::vector< std::vector< unsigned int > >* pvvLabelPointIdx_);
protected:
	void normalHistogram( const cv::Mat& cvmNls_, int nSamples_, std::vector< tp_normal_hist_bin >* pvNormalHistogram_);
	void distanceHistogram( const cv::Mat& cvmNls_, const cv::Mat& cvmPts_, const unsigned int& nSamples, const std::vector< unsigned int >& vIdx_, tp_hist* pvDistHist );
	void calcMergeFlag( const tp_hist& vDistHist, const double& dSampleStep, std::vector< tp_flag >* vMergeFlags );
	void mergeBins( const std::vector< tp_flag >& vMergeFlags_, const tp_hist& vDistHist_, const std::vector< unsigned int >& vLabelPointIdx_, short* pLabel_, cv::Mat* pcvmLabel_ );
	void clusterDistance( const unsigned short uPyrLevel_, const std::vector< std::vector<unsigned int> >& vvNormalClusterPtIdx_, cv::Mat* cvmDistanceClusters_ );
public:
	//global model
	std::vector< Eigen::Vector3d > _vGlobalPts;
	std::vector< Eigen::Vector3d > _vGlobalNormals;
	std::vector<const unsigned char*> _vGlobalColors;
	Eigen::Vector3d	_eivGlobalCentrod;

	//pyramid model (GL-convention)r
	std::vector< std::vector< Eigen::Vector3d > > _vvPyramidPts; 
	std::vector< std::vector< Eigen::Vector3d > >_vvPyramidNormals;
	std::vector< std::vector<const unsigned char*> > _vvPyramidColors;
	std::vector< std::vector<int> > _vvX;//x index for non zero PyramidPts, Normals and Colors
	std::vector< std::vector<int> > _vvY;//y index
	std::vector< std::vector< unsigned int > > _vvLabelPointIdx;//normal idx associated with each cluster label
	std::vector< std::vector< unsigned int > > _vvClusterPointIdx;
	std::vector< Eigen::Vector3d > _vLabelAvgNormals;
	//planes
	std::vector< Eigen::Vector3d > _veivPlane;
	//clusters
	boost::shared_ptr<cv::Mat>   _acvmShrPtrNormalClusters[4];
	boost::shared_ptr<cv::Mat> _acvmShrPtrDistanceClusters[4];
private:
	//frame data
	Eigen::Vector3d _eivCentroid;
	//frame pyramid
	std::vector< cv::Mat > _vcvmPyramidRGBs;
	std::vector< cv::Mat > _vcvmPyramidDepths;
	//the minmum area of a cluster
	unsigned short _usMinArea;
	//frame index
	unsigned int _uCurrent;
	//video source
	btl::extra::videosource::VideoSourceKinect& _cKinect;
#ifdef TIMER
	//timer
	boost::posix_time::ptime _cT0, _cT1;
	boost::posix_time::time_duration _cTDAll;
	float _fFPS;//frame per second
#endif
public:
	//control
	int _nKNearest; // for normal extraction using PCL
	enum {  _FAST, _PCL } _eNormalExtraction;
};

}//extra
}//btl
#endif

