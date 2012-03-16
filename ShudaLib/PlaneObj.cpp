#include <vector>
#include <list>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include "PlaneObj.h"

bool btl::geometry::SPlaneObj::identical(const SPlaneObj& sPlane_ ) const{
	double dCos = _eivAvgNormal.dot(sPlane_._eivAvgNormal);
	double dDif = fabs(_dAvgPosition - sPlane_._dAvgPosition );
	if(dCos > 0.85 && dDif < 0.05 ) return true; 
	else return false;
}

void btl::geometry::mergePlaneObj(btl::geometry::tp_plane_obj_list& lPlanes_, cv::Mat* pcvmDistanceClusters_ ){
	btl::geometry::tp_plane_obj_list::iterator itErase;
	bool bErase = false;
	float* pLabel = (float*)pcvmDistanceClusters_->data;
	for (btl::geometry::tp_plane_obj_list::iterator itMerging = lPlanes_.begin(); itMerging!=lPlanes_.end(); itMerging++ ){
		for (btl::geometry::tp_plane_obj_list::iterator itTesting = itMerging; itTesting!=lPlanes_.end(); itTesting++ ){
			if (bErase) {
				lPlanes_.erase(itErase);
				bErase = false;
			}
			if( itTesting!= itMerging && itMerging->identical( *itTesting ) ){
				//merge itMenging and itTesting
				int nMerging = itMerging->_vIdx.size();
				int nTesting = itTesting->_vIdx.size();
				itMerging->_dAvgPosition = (nMerging* itMerging->_dAvgPosition + nTesting* itTesting->_dAvgPosition)/(nMerging+nTesting);
				itMerging->_eivAvgNormal = (nMerging* itMerging->_eivAvgNormal + nTesting* itTesting->_eivAvgNormal);
				itMerging->_eivAvgNormal.normalize();
				itMerging->_vIdx.insert(itMerging->_vIdx.end(),itTesting->_vIdx.begin(),itTesting->_vIdx.end());
				float fMergingLabel = pLabel[*itMerging->_vIdx.begin()];
				for( std::vector<unsigned int>::const_iterator citIdx = itTesting->_vIdx.begin(); citIdx != itTesting->_vIdx.end(); citIdx++ ){
					pLabel[*citIdx] = fMergingLabel;
				}//for each pixel being merged
				//set up erase flag to erase itTesting
				itErase = itTesting;
				bErase = true;
			}//if the planes are identical
		}//for each plane in reference frame
		if (bErase) {
			lPlanes_.erase(itErase);
			bErase = false;
		}
	}//for each plane in the list
}//mergePlaneObj() 
void btl::geometry::transformPlaneIntoWorldCVCV(tp_plane_obj& sPlane_, const Eigen::Matrix3d& eimRw_, const Eigen::Vector3d& eivTw_){
	//http://stackoverflow.com/questions/2096474/given-a-surface-normal-find-rotation-for-3d-plane
	// nw = Rw' * nc
	// dw = Tw' * nc + dc 
	// where nc is the plane normal in camera coordinates and nw is the plane normal in world 
	// and dc is the d coefficient in camera and dw is the d coefficient in world
	sPlane_._dAvgPosition = eivTw_.dot(sPlane_._eivAvgNormal)+sPlane_._dAvgPosition; //1
	sPlane_._eivAvgNormal = eimRw_.transpose()*sPlane_._eivAvgNormal; //2 order is important
}//transformPlaneIntoWorldCVCV()
void btl::geometry::transformPlaneIntoLocalCVCV(tp_plane_obj& sPlane_, const Eigen::Matrix3d& eimRw_, const Eigen::Vector3d& eivTw_){
	Eigen::Vector3d eivPtw = sPlane_._dAvgPosition*sPlane_._eivAvgNormal;
	Eigen::Vector3d eivPt  = eimRw_*eivPtw + eivTw_;
	sPlane_._eivAvgNormal = eimRw_ * sPlane_._eivAvgNormal;
	sPlane_._dAvgPosition = fabs(eivPt.dot(sPlane_._eivAvgNormal));
}//transformPlaneIntoLocalCVCV()