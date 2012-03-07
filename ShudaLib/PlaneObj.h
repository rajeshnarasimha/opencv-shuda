#ifndef BTL_GEOMETRY_PLANEOBJ
#define BTL_GEOMETRY_PLANEOBJ

namespace btl{ namespace geometry
{

typedef struct SPlaneObj{
	SPlaneObj():_eivAvgNormal(0,0,0),_dAvgPosition(0){
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
		_usIdx				= sPlane_._usIdx;
		return *this;
	}
	//data
	Eigen::Vector3d _eivAvgNormal;
	double _dAvgPosition;
	std::vector<unsigned int> _vIdx;
	unsigned short _usIdx;
	bool _bCorrespondetFound;
} tp_plane_obj;

typedef std::list<tp_plane_obj> tp_plane_obj_list;
void mergePlaneObj(btl::geometry::tp_plane_obj_list& lPlanes_ );
void transformPlaneIntoWorldCVCV(tp_plane_obj& sPlane_, const Eigen::Matrix3d& eimRw_, const Eigen::Vector3d& eivTw_);
void transformPlaneIntoLocalCVCV(tp_plane_obj& sPlane_, const Eigen::Matrix3d& eimRW_, const Eigen::Vector3d& eivTw_);

}//geometry
}//btl

#endif