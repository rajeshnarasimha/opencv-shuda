#ifndef BTL_HISTOGRAM
#define BTL_HISTOGRAM

namespace btl { namespace utility {

struct SNormalHist{
	//normal histogram type
	typedef std::pair< std::vector< unsigned int >, Eigen::Vector3d > tp_normal_hist_bin;

	tp_normal_hist_bin** _ppNormalHistogram;//contains extra bins
	std::vector<ushort> _vBins;//store all effective bins
	float _fBinSize;
	unsigned short _usSamplesAzimuth;
	unsigned short _usSamplesElevationZ;
	unsigned short _usWidth;
	unsigned short _usLevel;
	unsigned short _usTotal;
	boost::scoped_ptr<cv::gpu::GpuMat> _acvgmScpPtrBinIdx[4];
	boost::scoped_ptr<cv::Mat> _acvmScpPtrBinIdx[4];

	//cpu histogram
	std::vector< btl::utility::SNormalHist::tp_normal_hist_bin > _vNormalHistogram;

	~SNormalHist(){delete _ppNormalHistogram;}
	void init(const unsigned short usSamples_);
	void getNeighbourIdxCylinder(const ushort& usIdx_, std::vector< ushort >* pNeighbours_ );
	void gpuNormalHistogram( const cv::gpu::GpuMat& cvgmNls_, const cv::Mat& cvmNls_, const ushort usPryLevel_,btl::utility::tp_coordinate_convention eCon_);
	void normalHistogram( const cv::Mat& cvmNls_, int nSamples_, btl::utility::tp_coordinate_convention eCon_);
private:
	void clear( const unsigned short usPyrLevel_ );

};

struct SDistanceHist{
	//distance histogram type
	typedef std::pair< double,unsigned int >						  tp_pair_hist_element; 
	typedef std::pair< std::vector< tp_pair_hist_element >, double >  tp_pair_hist_bin;
	typedef std::vector< tp_pair_hist_bin >							  tp_dist_hist;
private:
	enum tp_flag { EMPTY, NO_MERGE, MERGE_WITH_LEFT, MERGE_WITH_RIGHT, MERGE_WITH_BOTH };
public:
	void init(const unsigned short usSamples_);
	void distanceHistogram( const cv::Mat& cvmNls_, const cv::Mat& cvmPts_, const std::vector< unsigned int >& vIdx_ );
	void calcMergeFlag();
	void mergeDistanceBins( const std::vector< unsigned int >& vLabelPointIdx_, short* pLabel_, cv::Mat* pcvmLabel_ );

	boost::scoped_ptr<tp_dist_hist> _pvDistHist;
	std::vector< tp_flag > _vMergeFlags;

	unsigned short _uSamples;
	double _dLow;
	double _dHigh;
	double _dSampleStep; 
	double _dMergeDistance;
	unsigned short _usMinArea;
};

}//utility
}//btl
#endif