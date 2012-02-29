#ifndef BTL_EXTRA_MODEL
#define BTL_EXTRA_MODEL

namespace btl{ namespace geometry
{
#define VOLUME_RESOL 128  
#define VOLUME_LEVEL 16384 //VOLUME_RESOL * VOLUME_RESOL
#define VOXEL_TOTAL 2097152//VOLUME_LEVEL * VOLUME_RESOL
class CModel
{
//type
public:
	typedef boost::shared_ptr<CModel> tp_shared_ptr;
private:
	//methods
public:
	CModel();
	~CModel();
	void gpuIntegrate( btl::kinect::CKeyFrame& cFrame_, unsigned short usPyrLevel_ );
	void gpuRenderVoxel();
public:
	//data
	Eigen::Vector3d _eivAvgNormal;
	double _dAvgPosition;
	std::vector<unsigned int> _vVolumIdx;
//volume data
	//the center of the volume defines the origin of the world coordinate
	//and follows the right-hand cv-convention
	//physical size of the volume
	float _dScale;
	//host
	cv::Mat _cvmYZxXVolContent; //x*y,z,CV_32FC1,x-first
	//device
	cv::gpu::GpuMat _cvgmYZxXVolContent;
	cv::gpu::GpuMat _cvgmYZxZVolCenters;
};




}//geometry
}//btl
#endif

