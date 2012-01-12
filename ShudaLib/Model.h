#ifndef BTL_EXTRA_MODEL
#define BTL_EXTRA_MODEL


namespace btl
{
namespace extra
{

class CModel
{
public:
	CModel( VideoSourceKinect& cKinect_ );
	~CModel(void);

	void loadFrame();
public:
	//global model
	std::vector< Eigen::Vector3d > _vGlobalPts;
	std::vector< Eigen::Vector3d > _vGlobalNormals;
	std::vector<const unsigned char*> _vGlobalColors;
	//frame model
	std::vector< Eigen::Vector3d > _vPts;
	std::vector< Eigen::Vector3d > _vNormals;
	std::vector<const unsigned char*> _vColors;

private:
	//raw data
	std::vector< cv::Mat > _vcvmRGBs;
	std::vector< cv::Mat > _vcvmDepths;
	std::vector< Eigen::Vector3d > _veivCentroids;
	unsigned int _uMaxFrames;
	//frame data
	cv::Mat _cvmRGB;
	cv::Mat _cvmDepth;
	Eigen::Vector3d _eivCentroid;
	unsigned int _uCurrent;
	// refreshed for every frame
	//X,Y,Z coordinate of depth w.r.t. RGB camera reference system
	//in the format of the RGB image
	double*  _pPointL0; //(need to be initially allocated in constructor)
	double*  _pPointL1;
	double*  _pPointL2;
	double*  _pPointL3;
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

