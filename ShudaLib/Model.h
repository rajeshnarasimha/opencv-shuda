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
	CModel( VideoSourceKinect& cKinect_ );
	~CModel(void);
	//loaders
	void loadFrame(); //load depth and rgb from video source and convert it to point cloud data
	void loadPyramid(); //load pyramid
	void convert2PointCloudModel(const cv::Mat& cvmDepth_,const cv::Mat& cvmRGB_, std::vector<const unsigned char*>* vColor_, std::vector<Eigen::Vector3d>* vPt_, std::vector<Eigen::Vector3d>* vNormal_,int nLevel_=0);

public:
	//global model
	std::vector< Eigen::Vector3d > _vGlobalPts;
	std::vector< Eigen::Vector3d > _vGlobalNormals;
	std::vector<const unsigned char*> _vGlobalColors;
	Eigen::Vector3d	_eivGlobalCentrod;

	//pyramid model
	std::vector< std::vector< Eigen::Vector3d > > _vvPyramidPts;
	std::vector< std::vector< Eigen::Vector3d > >_vvPyramidNormals;
	std::vector< std::vector<const unsigned char*> > _vvPyramidColors;

private:
	//frame data
	Eigen::Vector3d _eivCentroid;
	//frame pyramid
	std::vector< cv::Mat > _vcvmPyramidRGBs;
	std::vector< cv::Mat > _vcvmPyramidDepths;

	//frame index
	unsigned int _uCurrent;
	// refreshed for every frame
	//X,Y,Z coordinate of depth w.r.t. RGB camera reference system
	//in the format of the RGB image
	double* _pPointL0;
	
	//video source
	btl::extra::videosource::VideoSourceKinect& _cKinect;
public:
	//control
	int _nKNearest; // for normal extraction using PCL
	enum {  _FAST, _PCL } _eNormalExtraction;
};

}//extra
}//btl
#endif

