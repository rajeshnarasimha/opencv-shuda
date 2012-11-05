#ifndef BTL_GEOMETRY_MODEL
#define BTL_GEOMETRY_MODEL

namespace btl{ namespace geometry
{

class CKinfuTracker
{
//type
public:
	typedef boost::shared_ptr<CKinfuTracker> tp_shared_ptr;
	enum {_X = 1, _Y = 2, _Z = 3};

private:
	void releaseVBOPBO();//methods
public:
	CKinfuTracker(ushort _usResolution);
	~CKinfuTracker();
	void gpuRenderVoxelInWorldCVGL();
	void gpuCreateVBO(btl::gl_util::CGLUtil::tp_ptr pGL_);
	void gpuIntegrateFrameIntoVolumeCVCV(const btl::kinect::CKeyFrame& cFrame_);
	void gpuRaycast(btl::kinect::CKeyFrame* pVirtualFrame_, std::string& strPathFileName_=std::string("")) const;
	void reset();
	void gpuExportVolume(const std::string& strPath_,ushort usNo_, ushort usV_, ushort usAxis_) const;

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
};




}//geometry
}//btl
#endif

