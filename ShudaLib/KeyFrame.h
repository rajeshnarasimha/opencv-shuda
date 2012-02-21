#ifndef BTL_KEYFRAME
#define BTL_KEYFRAME

namespace btl { namespace kinect {

class CKeyFrame {
public:
	typedef boost::shared_ptr< CKeyFrame > tp_shared_ptr;
	boost::shared_ptr<btl::kinect::CKinectView> _pView;
    cv::Mat _cvmRGB;
    cv::Mat _cvmBW;
    float* _pPts;

    std::vector<cv::KeyPoint> _vKeyPoints;
	std::vector<cv::DMatch> _vMatches;

	cv::gpu::GpuMat _cvgmKeyPoints;
	cv::gpu::GpuMat _cvgmDescriptors;
	cv::gpu::GpuMat _cvgmBW;

	Eigen::Matrix3d _eimR; //R & T is the relative pose w.r.t. the coordinate defined by the previous camera system.
    Eigen::Vector3d _eivT; //R & T is defined using CV convention

	bool _bIsReferenceFrame;

    CKeyFrame( btl::kinect::CCalibrateKinect& cCK_);

    ~CKeyFrame() {
        delete [] _pPts;
    }
    void assign ( const cv::Mat& rgb_, const float* pD_ );
	//detect surf features in the current frame
    void detectCorners();
	//detect matches between current frame and reference frame
    void detectCorrespondences ( const CKeyFrame& sReferenceKF_ );
	//
    void calcRT ( const CKeyFrame& sReferenceKF_ );

	void applyRelativePose( const CKeyFrame& sReferenceKF_ ) {
		_eimR = _eimR*sReferenceKF_._eimR;
		_eivT = _eimR*sReferenceKF_._eivT + _eivT;
	}

	void renderCamera( bool bRenderCamera_ ) const;

	void setView(Eigen::Matrix4d* pModelViewGL) const {
    	*pModelViewGL = btl::utility::setModelViewGLfromRTCV ( _eimR, _eivT );
		return;
	}

    void renderDepth() const;

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
