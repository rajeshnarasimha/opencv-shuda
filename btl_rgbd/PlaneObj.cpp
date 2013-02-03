#include <vector>
#include <list>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "PlaneObj.h"

bool btl::geometry::SPlaneObj::identical(const SPlaneObj& sPlane_ ) const{
	double dCos = _eivAvgNormal.dot(sPlane_._eivAvgNormal);
	double dDif = fabs(_dAvgPosition - sPlane_._dAvgPosition );
	if(dCos > 0.85 && dDif < 0.05 ) return true; 
	else return false;
}
void btl::geometry::separateIntoDisconnectedRegions(cv::Mat* pcvmLabels_){
	//spacial continuity constraint
	float *pLabel = (float*) pcvmLabels_->data;
	ushort usNewLabel = 50000;
	for (int r =0; r<pcvmLabels_->rows;r++){
		for (int c=0; c<pcvmLabels_->cols; c++,pLabel++){
			if( *pLabel>0 && *pLabel < 50000){
				cv::floodFill(*pcvmLabels_,cv::Point(c,r), usNewLabel, NULL, 0.5,0.5 );
				usNewLabel++;
			}//if pLabel is not floodfilled 
		}//for each col
	}//for each row	
}
void btl::geometry::mergePlaneObj(btl::geometry::tp_plane_obj_list* plPlanes_, cv::Mat* pcvmDistanceClusters_ ){
	btl::geometry::tp_plane_obj_list::iterator itErase;
	bool bErase = false;
	float* pLabel = (float*)pcvmDistanceClusters_->data;
	for (btl::geometry::tp_plane_obj_list::iterator itMerging = plPlanes_->begin(); itMerging!=plPlanes_->end(); itMerging++ ){
		for (btl::geometry::tp_plane_obj_list::iterator itTesting = itMerging; itTesting!=plPlanes_->end(); itTesting++ ){
			if (bErase) {
				plPlanes_->erase(itErase);
				bErase = false;
			}
			if( itTesting!= itMerging && itMerging->identical( *itTesting ) ){
				//merge itMenging and itTesting
				int nMerging = (int)itMerging->_vIdx.size();
				int nTesting = (int)itTesting->_vIdx.size();
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
			plPlanes_->erase(itErase);
			bErase = false;
		}
	}//for each plane in the list
}//mergePlaneObj() 
