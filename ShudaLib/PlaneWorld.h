#ifndef BTL_GEOMETRY_PLANEWORLD
#define BTL_GEOMETRY_PLANEWORLD

namespace btl{ namespace geometry
{

struct SSinglePlaneSingleViewInWorld{
	typedef boost::shared_ptr<SSinglePlaneSingleViewInWorld> tp_shared_ptr;
	typedef std::vector<unsigned int> tp_idx_vector;

	SSinglePlaneSingleViewInWorld(const btl::geometry::SPlaneObj& sPlaneObj_, ushort usPyrLevel_, btl::kinect::CKeyFrame::tp_ptr pFrame_ );
	void renderInWorldCVGL(btl::gl_util::CGLUtil::tp_ptr pGL_, const uchar* pColor_, const ushort usPyrLevel_ ) const;
	bool identical(const Eigen::Vector3d& eivNormal_, const double dPosition_, const ushort usPyrLevel_) const;
	//data
	Eigen::Vector3d _aeivAvgNormal[4];
	double _adAvgPosition[4];
	boost::shared_ptr<tp_idx_vector> _avIdx[4];
	btl::kinect::CKeyFrame::tp_ptr _pFrame;
};

class CSinglePlaneMultiViewsInWorld{
public:
	typedef std::vector<SSinglePlaneSingleViewInWorld::tp_shared_ptr> tp_shr_spsv_vec;
	typedef boost::shared_ptr<CSinglePlaneMultiViewsInWorld>          tp_shared_ptr;

	CSinglePlaneMultiViewsInWorld( const btl::geometry::SPlaneObj& sPlaneObj_, const ushort usPyrLevel_, btl::kinect::CKeyFrame::tp_ptr pFrame_ );

	void integrateFrameIntoPlanesWorldCVCV( btl::kinect::CKeyFrame::tp_ptr pFrame_, btl::geometry::tp_plane_obj_list& lPlanes_, const ushort usPyrLevel_);
	void renderPlaneInAllViewsWorldGL(btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_,const ushort usPyrLevel_ ) const;
	void renderPlaneInSingleViewWorldGL(btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_,const ushort usView_, const ushort usPyrLevel_ = 3) const;
	bool identical( const Eigen::Vector3d& eivNormal_, const double dPosition_, const ushort usPyrLevel_ ) const;

	//data
	tp_shr_spsv_vec _vShrPtrSPSV; // store all the points in multiple frames, each element in the vector contains the points from single frame
	ushort _usIdx;//the index of the plane;
	Eigen::Vector3d _aeivAvgNormal[4];
	double _adAvgPosition[4];
	//render context
};

class CMultiPlanesMultiViewsInWorld{
public:
	typedef boost::shared_ptr<CMultiPlanesMultiViewsInWorld> tp_shared_ptr;
	typedef std::vector<CSinglePlaneMultiViewsInWorld::tp_shared_ptr> tp_shr_spmv_vec;

	CMultiPlanesMultiViewsInWorld(btl::kinect::CKeyFrame::tp_ptr pFrame_ );
	void integrateFrameIntoPlanesWorldCVCV( btl::kinect::CKeyFrame::tp_ptr pFrame_ );
	void renderASinglePlaneInWorldGL(btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_, const ushort usPyrLevel_ ) const;
	void renderAllPlanesInSingleViewWorldGL(btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_, const ushort usPyrLevel_, const ushort usView_ ) const;
	//data
	tp_shr_spmv_vec _vShrPtrSPMV; //shared pointer of CSinglePlaneMultiViewsInWorld
	std::vector<btl::kinect::CKeyFrame::tp_shared_ptr> _vShrPtrKeyFrames;
};

}//geometry
}//btl

#endif