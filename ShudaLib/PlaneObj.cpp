#include <vector>
#include <list>
#include <Eigen/Core>
#include "PlaneObj.h"

bool btl::geometry::SPlaneObj::identical(const SPlaneObj& sPlane_ ) const{
	double dCos = _eivAvgNormal.dot(sPlane_._eivAvgNormal);
	double dDif = fabs(_dAvgPosition - sPlane_._dAvgPosition );
	if(dCos > 0.80 && dDif < 0.2 ) return true; 
	else return false;
}

void btl::geometry::mergePlaneObj(btl::geometry::tp_plane_obj_list& lPlanes_ ){
	for (btl::geometry::tp_plane_obj_list::iterator itMerging = lPlanes_.begin(); itMerging!=lPlanes_.end(); itMerging++ ){
		for (btl::geometry::tp_plane_obj_list::iterator itTesting = itMerging; itTesting!=lPlanes_.end(); itTesting++ ){
			if( itTesting!= itMerging && itMerging->identical( *itTesting ) ){
				itTesting->_usIdx=itMerging->_usIdx;
				int nMerging = itMerging->_vIdx.size();
				int nTesting = itTesting->_vIdx.size();
				itMerging->_dAvgPosition = (nMerging* itMerging->_dAvgPosition + nTesting* itTesting->_dAvgPosition)/(nMerging+nTesting);
				itMerging->_eivAvgNormal = (nMerging* itMerging->_eivAvgNormal + nTesting* itTesting->_eivAvgNormal);
				itMerging->_eivAvgNormal.normalize();
				itMerging->_vIdx.insert(itMerging->_vIdx.end(),itTesting->_vIdx.begin(),itTesting->_vIdx.end());
				lPlanes_.erase(itTesting);
				break;
			}//if the planes are identical
		}//for each plane in refererce frame
	}//for each plane in the list
}//mergePlaneObj() 
void btl::geometry::transformPlaneIntoWorldCVCV(tp_plane_obj& sPlane_, const Eigen::Matrix3d& eimRw_, const Eigen::Vector3d& eivTw_){
	Eigen::Vector3d eivAvgNormalNew = eimRw_.transpose()*sPlane_._eivAvgNormal;
	Eigen::Vector3d eivPt = sPlane_._dAvgPosition*sPlane_._eivAvgNormal;
	//transform the virtual pt into world
	Eigen::Vector3d eivPtw = eimRw_.transpose()*(eivPt-eivTw_);
	sPlane_._dAvgPosition= fabs(eivPtw.dot(eivAvgNormalNew));
	sPlane_._eivAvgNormal= eivAvgNormalNew;
}//transformPlaneIntoWorldCVCV()
void btl::geometry::transformPlaneIntoLocalCVCV(tp_plane_obj& sPlane_, const Eigen::Matrix3d& eimRw_, const Eigen::Vector3d& eivTw_){
	Eigen::Vector3d eivPtw = sPlane_._dAvgPosition*sPlane_._eivAvgNormal;
	Eigen::Vector3d eivPt  = eimRw_*eivPtw + eivTw_;
	sPlane_._eivAvgNormal = eimRw_ * sPlane_._eivAvgNormal;
	sPlane_._dAvgPosition = fabs(eivPt.dot(sPlane_._eivAvgNormal));
}//transformPlaneIntoLocalCVCV()