#ifndef BTL_CAMERA
#define BTL_CAMERA

namespace btl{ namespace image {

struct SCamera
{
	//type
	typedef boost::shared_ptr<SCamera> tp_shared_ptr;
	typedef boost::shared_ptr<SCamera> tp_scoped_ptr;
	typedef SCamera* tp_ptr;
	enum tp_camera {CAMERA_RGB, CAMERA_IR};

	//constructor
	//************************************
	// Method:    SCamera
	// FullName:  btl::image::SCamera::SCamera
	// Access:    public 
	// Returns:   na
	// Qualifier: 
	// Parameter: const std::string & strCamParam_: the yml file stores the camera internal parameters
	// Parameter: ushort uResolution_: the resolution level, where 0 is the original 1 is by half, 2 is the half of haly so on
	//************************************
	SCamera(const std::string& strCamParam_/*btl::kinect::SCamera::tp_camera eT_ = CAMERA_RGB*/,ushort uResolution_ = 0);//0 480x640
	//methods

	//rendering
	void LoadTexture ( const cv::Mat& cvmImg_, GLuint* puTesture_ );
	void setGLProjectionMatrix ( unsigned int nScaleViewport_, const double dNear_, const double dFar_ );
	//void renderCameraInGLLocal (const GLuint uTesture_, const cv::Mat& cvmImg_, float fPhysicalFocalLength_ = .02f, bool bRenderTexture_=true ) const;
	void renderCameraInGLLocal (const GLuint uTesture_, float fPhysicalFocalLength_ = .02f, bool bRenderTexture_=true ) const;
	void renderOnImage( int nX_, int nY_ );
	void importYML(const std::string& strCamParam_);
	void generateMapXY4Undistort();

	Eigen::Matrix3f getK(){ 
		Eigen::Matrix3f eimK;
		eimK << _fFx, 0.f , _u,
			    0.f,  _fFy, _v,
				0.f,  0.f , 1.f;
		return eimK;
	}

	//camera parameters
	ushort _uResolution;
	float _fFx, _fFy, _u, _v; //_dFxIR, _dFyIR IR camera focal length
	unsigned short _sWidth, _sHeight;
	cv::Mat _cvmDistCoeffs;
	//rendering
	//GLuint _uTexture;
	//cv::Mat        _cvmMapX; //for undistortion
	//cv::Mat		 _cvmMapY; //useless just for calling cv::remap
	cv::gpu::GpuMat  _cvgmMapX;
	cv::gpu::GpuMat  _cvgmMapY;
	//type
private:
	tp_camera _eType;
};

}//image
}//btl
#endif