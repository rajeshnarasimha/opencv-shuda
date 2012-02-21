#ifndef BTL_KEYFRAME
#define BTL_KEYFRAME

namespace btl { namespace kinect {

class CKeyFrame {
public:
	typedef boost::shared_ptr< CKeyFrame > tp_shared_ptr;
	btl::kinect::SCamera& _sRGB;
    cv::Mat _cvmRGB;
    cv::Mat _cvmBW;
	cv::Mat _cvmPt;
	cv::Mat _cvmNl;
    float* _pPts;
	float* _pNls;
	std::vector<cv::KeyPoint> _vKeyPoints;
	std::vector<cv::DMatch> _vMatches;

	cv::gpu::GpuMat _cvgmKeyPoints;
	cv::gpu::GpuMat _cvgmDescriptors;
	cv::gpu::GpuMat _cvgmBW;

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

	bool _bIsReferenceFrame;

    CKeyFrame( btl::kinect::SCamera& sRGB_);

    ~CKeyFrame() {
        delete [] _pPts;
    }
    void assign ( const cv::Mat& rgb_, const float* pD_ );
	//detect surf features in the current frame
    void detectCorners();
	void detectCorners(const short sLevel_);
	//detect matches between current frame and reference frame
    void detectCorrespondences ( const CKeyFrame& sReferenceKF_ );
	//calculate the R and T relative to Reference Frame.
    void calcRT ( const CKeyFrame& sReferenceKF_ );
	//accumulate the relative R T to the global RT
	void applyRelativePose( const CKeyFrame& sReferenceKF_ ) {
		_eimR = _eimR*sReferenceKF_._eimR;
		_eivT = _eimR*sReferenceKF_._eivT + _eivT;
	}

	// set the opengl modelview matrix to align with the current view
	void setView(Eigen::Matrix4d* pModelViewGL_) const {
		*pModelViewGL_ = btl::utility::setModelViewGLfromRTCV ( _eimR, _eivT );
		return;
	}
	// render the camera location in the GL world
	void renderCamera( bool bRenderCamera_ ) const;
	// render the depth in the GL world 
	void renderDepth() const;
	// copy the content to another keyframe at 
	void copyTo( CKeyFrame* pKF_, const short sLevel_ );
	void copyTo( CKeyFrame* pKF_ );
	// detect the correspondences 
	void detectConnectionFromCurrentToReference ( CKeyFrame& sReferenceKF_, const short sLevel_ );

private:
    void selectInlier ( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, const std::vector< int >& vVoterIdx_,
						Eigen::MatrixXd* peimXInlier_, Eigen::MatrixXd* peimYInlier_ );

	int voting ( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, 
		const Eigen::Matrix3d& eimR_, const Eigen::Vector3d& eivV_, const double& dThreshold, std::vector< int >* pvVoterIdx_ ); 

    void select5Rand ( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, boost::variate_generator< boost::mt19937&, boost::uniform_real<> >& dice_, 
						Eigen::MatrixXd* eimXTmp_, Eigen::MatrixXd* eimYTmp_, std::vector< int >* pvIdx_ = NULL );

};//end of class

}//utility
}//btl

#endif
