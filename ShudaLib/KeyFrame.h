#ifndef BTL_KEYFRAME
#define BTL_KEYFRAME

namespace btl { namespace kinect {

struct SPlaneCorrespondence{
	//constructor
	SPlaneCorrespondence(float fCur_,float fRef_, unsigned int _uMatchIdx)
	:_fCur(fCur_),_fRef(fRef_),_uMatchIdx(_uMatchIdx){}
	//matcher
	bool operator<( const SPlaneCorrespondence &sPC_ ) const {
		return _fCur < sPC_._fCur;
	}
	//data
	float _fCur;
	float _fRef;
	unsigned int _uMatchIdx;

};

class CKeyFrame {
	//type
public:
	typedef boost::shared_ptr< CKeyFrame > tp_shared_ptr;
	typedef CKeyFrame* tp_ptr;
	enum tp_cluster { NORMAL_CLUSTER, DISTANCE_CLUSTER};

public:
    CKeyFrame( btl::kinect::SCamera::tp_ptr pRGBCamera_ );
	CKeyFrame(CKeyFrame::tp_ptr pFrame_);
    ~CKeyFrame() {}
	// detect the correspondences 
	void extractSurfFeatures ();
	//calculate the R and T relative to Reference Frame.
	double calcRT ( const CKeyFrame& sReferenceKF_, const unsigned short sLevel_ , const double dDistanceThreshold_, unsigned short* pInliers_);
	void gpuICP(const CKeyFrame* pRefFrameWorld_,bool bUseReferenceRTAsInitial);

	//accumulate the relative R T to the global RT
	void applyRelativePose( const CKeyFrame& sReferenceKF_ ); 
	void associatePlanes(btl::kinect::CKeyFrame& sReferenceFrame_,const ushort usLevel_);
	bool isMovedwrtReferencInRadiusM(const CKeyFrame* const pRefFrame_, double dRotAngleThreshold_, double dTranslationThreshold_);


	// set the opengl modelview matrix to align with the current view
	void setView(Eigen::Matrix4d* pModelViewGL_) const {
		if (_eConvention == btl::utility::BTL_CV) {
			*pModelViewGL_ = btl::utility::setModelViewGLfromRTCV ( _eimRw, _eivTw );
			return;
		}else if(btl::utility::BTL_GL == _eConvention ){
			pModelViewGL_->setIdentity();
		}
	}
	void setView2(double* aModelViewGL_) const {
		if (_eConvention == btl::utility::BTL_CV) {
			Eigen::Matrix4d eimTmp;
			eimTmp = btl::utility::setModelViewGLfromRTCV ( _eimRw, _eivTw );
			eimTmp.transposeInPlace();
			memcpy(aModelViewGL_,eimTmp.data(),sizeof(double)*16);
			return;
		}else if(btl::utility::BTL_GL == _eConvention ){
			cv::Mat cvmTmp(4,4,CV_64FC1,aModelViewGL_);
			cv::setIdentity(cvmTmp);
		}
	}
	// render the camera location in the GL world
	void renderCameraInWorldCVGL( btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_,bool bRenderCamera_, bool bBW_, bool bRenderDepth_, const double& dSize_,const unsigned short uLevel_ );
	void renderCameraInWorldCVGL2( btl::gl_util::CGLUtil::tp_ptr pGL_, bool bRenderCameraTexture_, bool bBW_, const double& dPhysicalFocalLength_,const unsigned short usPyrLevel_);
	// render the depth in the GL world 
	void render3DPtsInLocalGL(btl::gl_util::CGLUtil::tp_ptr pGL_, const unsigned short uLevel_,const bool bRenderPlane_) const;
	void render3DPtsInWorldCVCV(btl::gl_util::CGLUtil::tp_ptr pGL_,const ushort usPyrLevel_,int nColorIdx_, bool bRenderPlanes_);
	void renderPlanesInLocalGL(btl::gl_util::CGLUtil::tp_ptr pGL_, const unsigned short uLevel_) const;
	void renderPlaneObjsInLocalCVGL(btl::gl_util::CGLUtil::tp_ptr pGL_,const unsigned short uLevel_) const;
	void renderASinglePlaneObjInLocalCVGL(const float*const pPt_, const float*const pNl_, const std::vector<unsigned int>& vIdx_, const unsigned char* pColor_) const;
	void renderASinglePlaneObjInWorldCVCV(const float*const pPt_, const float*const pNl_, const std::vector<unsigned int>& vIdx_, const unsigned char* pColor_) const;
	void gpuRender3DPtsInLocalCVGL(btl::gl_util::CGLUtil::tp_ptr pGL_,const ushort usColorIdx_, const unsigned short uLevel_, const bool bRenderPlane_) const;

	inline void loadGLMVIn() const{	glMultMatrixd ( _eimGLMVInv.data() );}

	// copy the content to another keyframe at 
	void copyTo( CKeyFrame* pKF_, const short sLevel_ );
	void copyTo( CKeyFrame* pKF_ );

	void detectPlane (const short uPyrLevel_);
	void gpuDetectPlane (const short uPyrLevel_);
	void transformPlaneObjsToWorldCVCV(const ushort usPyrLevel_);
	void gpuTransformToWorldCVCV(const ushort usPyrLevel_);
	void updateMVInv();
	void constructPyramid(const float fSigmaSpace_, const float fSigmaDisparity_);
	void setRT(double dXA_, double dYA_, double dZA_, double dXC_,double dYC_,double dZC_);

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
	void gpuClusterNormal(const unsigned short uPyrLevel_,cv::Mat* pcvmLabel_,btl::geometry::tp_plane_obj_list* pvPlaneObjs_);
	void allocate();
	void establishPlaneCorrespondences( const CKeyFrame& sReferenceKF_);

public:
	btl::kinect::SCamera::tp_ptr _pRGBCamera;
	//host
	boost::shared_ptr<cv::Mat> _acvmPyrDepths[4];
	boost::shared_ptr<cv::Mat> _acvmShrPtrPyrPts[4]; //using pointer array is because the vector<cv::Mat> has problem when using it &vMat[0] in calling a function
	boost::shared_ptr<cv::Mat> _acvmShrPtrPyrNls[4]; //CV_32FC3 type
	boost::shared_ptr<cv::Mat> _acvmShrPtrPyrRGBs[4];
	boost::shared_ptr<cv::Mat> _acvmShrPtrPyrBWs[4];
	//device
	boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrDepths[4];
	boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrPts[4]; //using pointer array is because the vector<cv::Mat> has problem when using it &vMat[0] in calling a function
	boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrNls[4]; //CV_32FC3 type
	boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrRGBs[4];
	boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrBWs[4];

	static boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrDisparity[4];
	static boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyr32FC1Tmp[4];


	static boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrAA[4];//for rendering
	//clusters
	boost::shared_ptr<cv::Mat>   _acvmShrPtrNormalClusters[4];
	boost::shared_ptr<cv::Mat> _acvmShrPtrDistanceClusters[4];
	static boost::shared_ptr<cv::Mat> _acvmShrPtrAA[4];//for rendering
		
	//pose
	//R & T is the relative pose w.r.t. the coordinate defined in previous camera system.
	//R & T is defined using CV convention
	//R & T X_curr = R* X_prev + T;
	//after applying void applyRelativePose() R, T -> R_w, T_w
	//X_c = R_w * X_w + T_w 
	//where _w defined in world reference system
	//      _c defined in camera reference system (local reference system) 
	Eigen::Matrix3d _eimRw; 
	Eigen::Vector3d _eivTw; 
	//GL ModelView Matrix
	Eigen::Matrix4d _eimGLMVInv;
	//render context
	//btl::gl_util::CGLUtil::tp_ptr _pGL;
	bool _bIsReferenceFrame;
	bool _bRenderPlane;
	bool _bGPURender;
	GLuint _uTexture;
	unsigned short _nColorIdx;

	btl::utility::tp_coordinate_convention _eConvention;
	tp_cluster _eClusterType;
	static btl::utility::SNormalHist _sNormalHist;
	static btl::utility::SDistanceHist _sDistanceHist;
	btl::geometry::tp_plane_obj_list _vPlaneObjsNormal;
	btl::geometry::tp_plane_obj_list _vPlaneObjsDistanceNormal[4];
	static boost::shared_ptr<cv::gpu::SURF_GPU> _pSurf;
private:
	//for surf matching
	//host
	std::vector<cv::KeyPoint> _vKeyPoints;
	std::vector<cv::DMatch> _vMatches;
	//device
	cv::gpu::GpuMat _cvgmKeyPoints;
	cv::gpu::GpuMat _cvgmDescriptors;
	//plane correspondences
	std::vector<SPlaneCorrespondence> _vPlaneCorrespondences;
	
};//end of class



}//utility
}//btl

#endif
