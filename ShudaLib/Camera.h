#ifndef BTL_CAMERA
#define BTL_CAMERA

namespace btl{ namespace kinect {

struct SCamera
{
	//type
	typedef boost::shared_ptr<SCamera> tp_shared_ptr;
	enum tp_camera {CAMERA_RGB, CAMERA_IR};

	//constructor
	SCamera(btl::kinect::SCamera::tp_camera eT_ = CAMERA_RGB);
	//methods

	//rendering
	void LoadTexture(const cv::Mat& img);
	void setIntrinsics ( unsigned int nScaleViewport_, const double dNear_, const double dFar_ );
	void renderCamera (const cv::Mat& cvmRGB_, double dPhysicalFocalLength_ = .02, bool bRenderTexture_=true ) const;
	void renderOnImage( int nX_, int nY_ );
	void importYML();
	//data
	float _fFx, _fFy, _u, _v; //_dFxIR, _dFyIR IR camera focal length
	unsigned short _sWidth, _sHeight;
	GLuint _uTexture;
	tp_camera _eType;
};

}//kinect
}//btl
#endif