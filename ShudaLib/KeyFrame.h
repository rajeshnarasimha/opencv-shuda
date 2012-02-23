#ifndef BTL_KEYFRAME
#define BTL_KEYFRAME

namespace btl { namespace kinect {

class CKeyFrame {
public:
	typedef boost::shared_ptr< CKeyFrame > tp_shared_ptr;
	typedef CKeyFrame* tp_ptr;

    CKeyFrame( btl::kinect::SCamera::tp_ptr pRGBCamera_ );
    ~CKeyFrame() {}
	// detect the correspondences 
	void detectConnectionFromCurrentToReference ( CKeyFrame& sReferenceKF_, const short sLevel_ );
	//calculate the R and T relative to Reference Frame.
	void calcRT ( const CKeyFrame& sReferenceKF_, const unsigned short sLevel_ );
	//accumulate the relative R T to the global RT
	void applyRelativePose( const CKeyFrame& sReferenceKF_ ) {
		_eimR = _eimR*sReferenceKF_._eimR;
		_eivT = _eimR*sReferenceKF_._eivT + _eivT;
	}
	// set the opengl modelview matrix to align with the current view
	void setView(Eigen::Matrix4d* pModelViewGL_) const {
		if (_eConvention == btl::utility::BTL_CV) {
			*pModelViewGL_ = btl::utility::setModelViewGLfromRTCV ( _eimR, _eivT );
			return;
		}else if(btl::utility::BTL_GL == _eConvention ){
			pModelViewGL_->setIdentity();
		}
	}
	// render the camera location in the GL world
	void renderCamera( bool bRenderCamera_, const unsigned short uLevel_=0) const;
	// render the depth in the GL world 
	void render3DPts(const unsigned short _uLevel) const;

	// copy the content to another keyframe at 
	void copyTo( CKeyFrame* pKF_, const short sLevel_ );
	void copyTo( CKeyFrame* pKF_ );

private:
    void selectInlier ( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, const std::vector< int >& vVoterIdx_,
						Eigen::MatrixXd* peimXInlier_, Eigen::MatrixXd* peimYInlier_ );

	int voting ( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, 
		const Eigen::Matrix3d& eimR_, const Eigen::Vector3d& eivV_, const double& dThreshold, std::vector< int >* pvVoterIdx_ ); 

    void select5Rand ( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, boost::variate_generator< boost::mt19937&, boost::uniform_real<> >& dice_, 
						Eigen::MatrixXd* eimXTmp_, Eigen::MatrixXd* eimYTmp_, std::vector< int >* pvIdx_ = NULL );
public:
	btl::kinect::SCamera::tp_ptr _pRGBCamera;

	boost::shared_ptr<cv::Mat> _acvmShrPtrPyrPts[4]; //using pointer array is because the vector<cv::Mat> has problem when using it &vMat[0] in calling a function
	boost::shared_ptr<cv::Mat> _acvmShrPtrPyrNls[4]; //CV_32FC3 type
	boost::shared_ptr<cv::Mat> _acvmShrPtrPyrRGBs[4];
	boost::shared_ptr<cv::Mat> _acvmShrPtrPyrBWs[4];

	boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrPts[4]; //using pointer array is because the vector<cv::Mat> has problem when using it &vMat[0] in calling a function
	boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrNls[4]; //CV_32FC3 type
	boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrRGBs[4];
	boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrBWs[4];

	Eigen::Matrix3d _eimR; //R & T is the relative pose w.r.t. the coordinate defined by the previous camera system.
	Eigen::Vector3d _eivT; //R & T is defined using CV convention
	//render context
	btl::gl_util::CGLUtil::tp_ptr _pGL;
	bool _bIsReferenceFrame;
	btl::utility::tp_coordinate_convention _eConvention;
private:
	std::vector<cv::KeyPoint> _vKeyPoints;
	std::vector<cv::DMatch> _vMatches;

	cv::gpu::GpuMat _cvgmKeyPoints;
	cv::gpu::GpuMat _cvgmDescriptors;

};//end of class

}//utility
}//btl

#endif
