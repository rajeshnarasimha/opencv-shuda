#ifndef BTL_KEYFRAME
#define BTL_KEYFRAME

namespace btl { namespace kinect {
class CKeyFrame {
	//type
public:
	typedef boost::shared_ptr< CKeyFrame > tp_shared_ptr;
	typedef CKeyFrame* tp_ptr;
	enum tp_cluster { NORMAL_CLUSTRE, DISTANCE_CLUSTER};

public:
    CKeyFrame( btl::kinect::SCamera::tp_ptr pRGBCamera_ );
    ~CKeyFrame() {}
	// detect the correspondences 
	void detectConnectionFromCurrentToReference ( CKeyFrame& sReferenceKF_, const short sLevel_ );
	//calculate the R and T relative to Reference Frame.
	double calcRT ( const CKeyFrame& sReferenceKF_, const unsigned short sLevel_ , unsigned short* pInliers_);
	//accumulate the relative R T to the global RT
	void applyRelativePose( const CKeyFrame& sReferenceKF_ ) {
		_eivT = _eimR*sReferenceKF_._eivT + _eivT;//order matters 
		_eimR = _eimR*sReferenceKF_._eimR;
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
	void renderCameraInGLWorld( bool bRenderCamera_, bool bBW_, bool bRenderDepth_, const double& dSize_,const unsigned short uLevel_ );
	// render the depth in the GL world 
	void render3DPtsInGLLocal(const unsigned short _uLevel) const;
	void renderPlanesInGLLocal(const unsigned short _uLevel) const;
	void gpuRender3DPtsCVInLocalGL(const unsigned short _uLevel) const;

	// copy the content to another keyframe at 
	void copyTo( CKeyFrame* pKF_, const short sLevel_ );
	void copyTo( CKeyFrame* pKF_ );

	void detectPlane (const short uPyrLevel_);
	void gpuDetectPlane (const short uPyrLevel_);

private:
	//surf keyframe matching
    void selectInlier ( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, const std::vector< int >& vVoterIdx_,
						Eigen::MatrixXd* peimXInlier_, Eigen::MatrixXd* peimYInlier_ );
	int voting ( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, 
		const Eigen::Matrix3d& eimR_, const Eigen::Vector3d& eivV_, const double& dThreshold, std::vector< int >* pvVoterIdx_ ); 
    void select5Rand ( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, boost::variate_generator< boost::mt19937&, boost::uniform_real<> >& dice_, 
						Eigen::MatrixXd* eimXTmp_, Eigen::MatrixXd* eimYTmp_, std::vector< int >* pvIdx_ = NULL );
	//for plane detection
	//for normal cluster
	void clusterNormal(const unsigned short& uPyrLevel_,cv::Mat* pcvmLabel_,std::vector< std::vector< unsigned int > >* pvvLabelPointIdx_);
	void gpuClusterNormal(const unsigned short uPyrLevel_,cv::Mat* pcvmLabel_,std::vector< std::vector< unsigned int > >* pvvLabelPointIdx_);
	//for distance cluster
	void clusterDistance( const unsigned short uPyrLevel_, const std::vector< std::vector<unsigned int> >& vvNormalClusterPtIdx_, cv::Mat* cvmDistanceClusters_ );

public:
	btl::kinect::SCamera::tp_ptr _pRGBCamera;
	//host
	boost::shared_ptr<cv::Mat> _acvmShrPtrPyrPts[4]; //using pointer array is because the vector<cv::Mat> has problem when using it &vMat[0] in calling a function
	boost::shared_ptr<cv::Mat> _acvmShrPtrPyrNls[4]; //CV_32FC3 type
	boost::shared_ptr<cv::Mat> _acvmShrPtrPyrRGBs[4];
	boost::shared_ptr<cv::Mat> _acvmShrPtrPyrBWs[4];
	//device
	boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrPts[4]; //using pointer array is because the vector<cv::Mat> has problem when using it &vMat[0] in calling a function
	boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrNls[4]; //CV_32FC3 type
	boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrRGBs[4];
	boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrBWs[4];
	static boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrAA[4];//for rendering
	//clusters
	boost::shared_ptr<cv::Mat>   _acvmShrPtrNormalClusters[4];
	boost::shared_ptr<cv::Mat> _acvmShrPtrDistanceClusters[4];
	static boost::shared_ptr<cv::Mat> _acvmShrPtrAA[4];//for rendering
		
	//pose
	Eigen::Matrix3d _eimR; //R & T is the relative pose w.r.t. the coordinate defined by the previous camera system.
	Eigen::Vector3d _eivT; //R & T is defined using CV convention
	//render context
	btl::gl_util::CGLUtil::tp_ptr _pGL;
	bool _bIsReferenceFrame;
	bool _bRenderPlane;
	bool _bGPURender;
	GLuint _uTexture;

	btl::utility::tp_coordinate_convention _eConvention;
	tp_cluster _eClusterType;
	static btl::utility::SNormalHist _sNormalHist;
	static btl::utility::SDistanceHist _sDistanceHist;
private:
	//for surf matching
	//host
	std::vector<cv::KeyPoint> _vKeyPoints;
	std::vector<cv::DMatch> _vMatches;
	//device
	cv::gpu::GpuMat _cvgmKeyPoints;
	cv::gpu::GpuMat _cvgmDescriptors;
	//for plane detection
	std::vector< std::vector< unsigned int > > _vvLabelPointIdx;
	//the minmum area of a cluster
	unsigned short _usMinArea;
	
};//end of class



}//utility
}//btl

#endif
