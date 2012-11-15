#ifndef BTL_CAMERA
#define BTL_CAMERA

namespace btl{ namespace kinect {

struct SCamera
{
	//type
	typedef boost::shared_ptr<SCamera> tp_shared_ptr;
	typedef SCamera* tp_ptr;
	enum tp_camera {CAMERA_RGB, CAMERA_IR};

	//constructor
	SCamera(btl::kinect::SCamera::tp_camera eT_ = CAMERA_RGB,ushort uResolution_ = 0);//0 480x640
	//methods

	//rendering
	void LoadTexture ( const cv::Mat& cvmImg_, GLuint* puTesture_ );
	void setGLProjectionMatrix ( unsigned int nScaleViewport_, const double dNear_, const double dFar_ );
	void renderCameraInGLLocal (const GLuint uTesture_, const cv::Mat& cvmImg_, float fPhysicalFocalLength_ = .02f, bool bRenderTexture_=true ) const;
	void renderOnImage( int nX_, int nY_ );
	void importYML();
	void generateMapXY4Undistort();

	//camera parameters
	ushort _uResolution;
	float _fFx, _fFy, _u, _v; //_dFxIR, _dFyIR IR camera focal length
	unsigned short _sWidth, _sHeight;
	cv::Mat _cvmDistCoeffs;
	//rendering
	//GLuint _uTexture;
	//cv::Mat          _cvmMapX; //for undistortion
	//cv::Mat			 _cvmMapY; //useless just for calling cv::remap
	cv::gpu::GpuMat  _cvgmMapX;
	cv::gpu::GpuMat  _cvgmMapY;
	//type
	tp_camera _eType;
};

}//kinect
}//btl
#endif