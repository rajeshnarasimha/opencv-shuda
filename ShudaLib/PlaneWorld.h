#ifndef BTL_GEOMETRY_PLANEWORLD
#define BTL_GEOMETRY_PLANEWORLD

namespace btl{ namespace geometry
{

struct CSinglePlaneSingleViewInWorld{
	typedef boost::shared_ptr<CSinglePlaneSingleViewInWorld> tp_shared_ptr;
	typedef CSinglePlaneSingleViewInWorld*                   tp_ptr;
private:
	typedef std::vector<unsigned int>						 tp_idx_vector;
public:
	CSinglePlaneSingleViewInWorld(const btl::geometry::SPlaneObj& sPlaneObj_, ushort usPyrLevel_, btl::kinect::CKeyFrame::tp_ptr pFrame_, ushort usPlaneIdx_);
	void renderInWorldCVGL(btl::gl_util::CGLUtil::tp_ptr pGL_, const uchar* pColor_, const ushort usPyrLevel_ ) const;
	bool identical(const Eigen::Vector3d& eivNormal_, const double dPosition_, const ushort usPyrLevel_) const;
	//data
	Eigen::Vector3d _aeivAvgNormal[4];
	double _adAvgPosition[4];
	boost::shared_ptr<tp_idx_vector> _avIdx[4];
	btl::kinect::CKeyFrame::tp_ptr _pFrame;
	ushort _usPlaneIdx;
};

class CMultiPlanesSingleViewInWorld{
public:
	typedef boost::shared_ptr<CMultiPlanesSingleViewInWorld>	   tp_shared_ptr;
	typedef CMultiPlanesSingleViewInWorld*						   tp_ptr;
	typedef std::vector<CSinglePlaneSingleViewInWorld::tp_ptr>	   tp_ptr_spsv_vec;

	CMultiPlanesSingleViewInWorld(btl::kinect::CKeyFrame::tp_ptr pFrame_);
	void renderAllPlanesInSingleViewWorldCVGL(btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_, const ushort usPyrLevel_) const;
	
	btl::kinect::CKeyFrame::tp_ptr _pFrame;
	tp_ptr_spsv_vec _vPtrSPSV;
};

class CSinglePlaneMultiViewsInWorld{
public:
	typedef std::vector<CSinglePlaneSingleViewInWorld::tp_shared_ptr> tp_shr_spsv_vec;
	typedef boost::shared_ptr<CSinglePlaneMultiViewsInWorld>          tp_shared_ptr;


	CSinglePlaneMultiViewsInWorld( const btl::geometry::SPlaneObj& sPlaneObj_, const ushort usPyrLevel_, btl::kinect::CKeyFrame::tp_ptr pFrame_, ushort usPlaneIdx_ );

	void integrateFrameIntoPlanesWorldCVCV( btl::kinect::CKeyFrame::tp_ptr pFrame_, btl::geometry::tp_plane_obj_list& lPlanes_, const ushort usPyrLevel_, CMultiPlanesSingleViewInWorld::tp_ptr pMPSV_);
	void renderSinglePlaneInAllViewsWorldGL(btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_,const ushort usPyrLevel_ ) const;
	void renderSinglePlaneInSingleViewWorldGL(btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_,const ushort usView_, const ushort usPyrLevel_ = 3) const;
	bool identical( const Eigen::Vector3d& eivNormal_, const double dPosition_, const ushort usPyrLevel_ ) const;

	//data
	tp_shr_spsv_vec _vShrPtrSPSV; // store all the points in multiple frames, each element in the vector contains the points from single frame
	ushort _usPlaneIdx;//the index of the plane;
	Eigen::Vector3d _aeivAvgNormal[4];
	double _adAvgPosition[4];
	//render context
};

class CMultiPlanesMultiViewsInWorld{
public:
	typedef boost::shared_ptr<CMultiPlanesMultiViewsInWorld> tp_shared_ptr;
private:
	typedef std::vector<CSinglePlaneMultiViewsInWorld::tp_shared_ptr> tp_shr_spmv_vec;
	typedef std::vector<CMultiPlanesSingleViewInWorld::tp_shared_ptr> tp_shr_mpsv_vec;
	typedef std::vector<CSinglePlaneSingleViewInWorld::tp_ptr>		  tp_ptr_spsv_vec;
	typedef std::vector<btl::kinect::CKeyFrame::tp_shared_ptr>		  tp_shr_kfrm_vec;
public:
	CMultiPlanesMultiViewsInWorld(btl::kinect::CKeyFrame::tp_ptr pFrame_ );
	void integrateFrameIntoPlanesWorldCVCV( btl::kinect::CKeyFrame::tp_ptr pFrame_ );
	void renderAllPlanesInGivenViewWorldCVGL( btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_, const ushort usPyrLevel_, const ushort usView_ );
	void renderGivenPlaneInGivenViewWorldCVGL( btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_, const ushort usPyrLevel_, const ushort usView_, const ushort usPlane_ );
	void fuse(CMultiPlanesSingleViewInWorld::tp_ptr pMPSV_, const ushort usPyrLevel_);
	void renderGivenPlaneInAllViewWorldCVGL( btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_, const ushort usPyrLevel_, const ushort usPlane_ );
	void renderAllCamrea(btl::gl_util::CGLUtil::tp_ptr pGL_,bool bBW_, bool bRenderDepth_, float fSize_ );
	//data
	tp_shr_spmv_vec _vShrPtrSPMV; //shared pointer of CSinglePlaneMultiViewsInWorld
	tp_shr_mpsv_vec _vShrPtrMPSV; //shared pointer of CMultiPlanesSingleViewInWorld
	tp_shr_kfrm_vec _vShrPtrKFrs; //shared pointer of CKeyFrame
};

}//geometry
}//btl

#endif