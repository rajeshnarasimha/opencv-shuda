#ifndef BTL_GEOMETRY_PLANEOBJ
#define BTL_GEOMETRY_PLANEOBJ

namespace btl{ namespace geometry
{

typedef struct SPlaneObj{
	SPlaneObj():_eivAvgNormal(0,0,0),_dAvgPosition(0){
		_vIdx.reserve(100);
		_bCorrespondetFound = false;
	}
	SPlaneObj(Eigen::Vector3d eivNormal_, double dPos_)
		:_eivAvgNormal(eivNormal_),_dAvgPosition(dPos_){
		_vIdx.reserve(100);
		_bCorrespondetFound = false;
	}
	void reset(){
		_eivAvgNormal.setZero();
		_dAvgPosition=0;
		_vIdx.clear();
		_vIdx.reserve(100);
	}
	bool identical( const SPlaneObj& sPlane_) const;
	inline SPlaneObj& operator= ( const SPlaneObj& sPlane_ ){
		_vIdx				= sPlane_._vIdx;
		_eivAvgNormal		= sPlane_._eivAvgNormal;
		_dAvgPosition	    = sPlane_._dAvgPosition;
		_bCorrespondetFound = sPlane_._bCorrespondetFound;
		_uIdx				= sPlane_._uIdx;
		return *this;
	}
	//data
	Eigen::Vector3d _eivAvgNormal;
	double _dAvgPosition;
	std::vector<unsigned int> _vIdx;
	unsigned int _uIdx;
	bool _bCorrespondetFound;
} tp_plane_obj;

typedef std::list<tp_plane_obj> tp_plane_obj_list;
void mergePlaneObj(btl::geometry::tp_plane_obj_list* plPlanes_, cv::Mat* pcvmDistanceClusters_ );
void transformPlaneIntoWorldCVCV(tp_plane_obj& sPlane_, const Eigen::Matrix3d& eimRw_, const Eigen::Vector3d& eivTw_);
void transformPlaneIntoLocalCVCV(tp_plane_obj& sPlane_, const Eigen::Matrix3d& eimRW_, const Eigen::Vector3d& eivTw_);
void separateIntoDisconnectedRegions(cv::Mat* pcvmLabels_);

}//geometry
}//btl

#endif