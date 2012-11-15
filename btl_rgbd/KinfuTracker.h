#ifndef BTL_GEOMETRY_MODEL
#define BTL_GEOMETRY_MODEL

namespace btl{ namespace geometry
{

class CCubicGrids
{
//type
public:
	typedef boost::shared_ptr<CCubicGrids> tp_shared_ptr;
	enum {_X = 1, _Y = 2, _Z = 3};

	enum
	{ 
		DEFAULT_OCCUPIED_VOXEL_BUFFER_SIZE = 2 * 1000 * 1000      
	};

private:
	void releaseVBOPBO();//methods
public:
	CCubicGrids(ushort _usResolution,float fVolumeSizeM_);
	~CCubicGrids();
	void gpuRenderVoxelInWorldCVGL();
	void gpuCreateVBO(btl::gl_util::CGLUtil::tp_ptr pGL_);
	void gpuIntegrateFrameIntoVolumeCVCV(const btl::kinect::CKeyFrame& cFrame_);
	void gpuRaycast(btl::kinect::CKeyFrame* pVirtualFrame_, std::string& strPathFileName_=std::string("")) const;
	void reset();
	void gpuExportVolume(const std::string& strPath_,ushort usNo_, ushort usV_, ushort usAxis_) const;

	
	void gpuMarchingCubes();
	void gpuGetOccupiedVoxels();
	void exportYML(const std::string& strPath_, const unsigned int uNo_ = 0 ) const;
	void importYML(const std::string& strPath_) ;
public:

	//data
	Eigen::Vector3d _eivAvgNormal;
	double _dAvgPosition;
	std::vector<unsigned int> _vVolumIdx;
//volume data
	//the center of the volume defines the origin of the world coordinate
	//and follows the right-hand cv-convention
	//physical size of the volume
	float _fVolumeSizeM;//in meter
	float _fVoxelSizeM; //in meter
	unsigned int _uResolution;
	unsigned int _uVolumeLevel;
	unsigned int _uVolumeTotal;
	//truncated distance in meter
	//must be larger than 2*voxelsize 
	float _fTruncateDistanceM;
	//host
	cv::Mat _cvmYZxXVolContent; //y*z,x,CV_32FC1,x-first
	//device
	cv::gpu::GpuMat _cvgmYZxXVolContentCV;
	//render context
	btl::gl_util::CGLUtil::tp_ptr _pGL;
	GLuint _uVBO;
	cudaGraphicsResource* _pResourceVBO;
	GLuint _uPBO;
	cudaGraphicsResource* _pResourcePBO;
	GLuint _uTexture;


	/** \brief Temporary buffer used by marching cubes (first row stores occuped voxes id, second number of vetexes, third poits offsets */
	cv::gpu::GpuMat/*pcl::gpu::DeviceArray2D<int>*/ _cvgmOccupiedVoxelsBuffer;
};




}//geometry
}//btl
#endif

