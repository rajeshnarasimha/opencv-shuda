#define INFO
#include <boost/scoped_ptr.hpp>
#include <vector>

#include <opencv2/gpu/gpu.hpp>
#include <Eigen/Core>
#include "OtherUtil.hpp"
#include "Kinect.h"
#include "PlaneObj.h"
#include "Histogram.h"
#include "cuda/CudaLib.h"
#include <math.h>
#include "Utility.hpp"

void btl::utility::SNormalHist::init(const unsigned short usSamples_) 
{
	for(int i=0; i<4; i++){
		int nRows = KINECT_HEIGHT>>i; 
		int nCols = KINECT_WIDTH>>i;
		//host
		_acvgmScpPtrBinIdx[i].reset(new cv::gpu::GpuMat(nRows,nCols,CV_16SC1));
		_acvmScpPtrBinIdx[i] .reset(new cv::Mat(nRows,nCols,CV_16SC1));
	}
	//asuming cv-convention
	const unsigned short usSamplesElevationZ = 1<<usSamples_; //2^usSamples
	const unsigned short usSamplesAzimuthX = usSamplesElevationZ<<1;   //usSamplesElevationZ*2
	const unsigned short usSamplesAzimuthY = usSamplesElevationZ<<1;   //usSamplesElevationZ*2
	const unsigned short usWidth = usSamplesAzimuthX;				    //
	const unsigned short usLevel = usSamplesAzimuthX<<(usSamples_+1);	//usSamplesAzimuthX*usSamplesAzimuthX
	const unsigned short usTotal = usLevel<<(usSamples_);  //usSamplesAzimuthX*usSamplesAzimuthY*usSamplesElevationZ

	const float fSqrt3_2 = sqrt(3.f)/2.f; // longest radius of cube
	float fSize = 1.f/usSamplesElevationZ;
	float fExtra= fSize/4;
	fSize = (fExtra + 1.f)/usSamplesElevationZ;
	float fCx,fCy,fCz;
	fCx=fCy=-fSize/2.f-(fExtra + 1.f); //from -1,-1

	//return values
	_usSamplesAzimuth = usSamplesAzimuthX;
	_usSamplesElevationZ = usSamplesElevationZ;
	_usWidth = usWidth;
	_usLevel = usLevel;
	_usTotal = usTotal;
	_fBinSize = fSize;

	_ppNormalHistogram = new tp_normal_hist_bin*[usTotal];
	unsigned short usIdx=0;
	unsigned short usBinCounter=0;
	_vBins.clear();
	fCz=fSize/2.f; // cv-convention
	for(unsigned short z =0; z < usSamplesElevationZ; z++){
		fCz -= fSize; //because of cv-convention
		fCy = -fSize/2.f-(fExtra + 1.f);
		for(unsigned short y =0; y < usSamplesAzimuthY; y++){
			fCy += fSize;
			fCx = -fSize/2.f-(fExtra + 1.f);
			for(unsigned short x =0; x < usSamplesAzimuthX; x++){
				fCx += fSize;
				//if( fabs(1.-sqrt(fCx*fCx+fCz*fCz+fCz*fCz)) < fSize*fSqrt3_2 ) {// the unit surface goes through that bin
				_ppNormalHistogram[usIdx]=new tp_normal_hist_bin(std::vector<unsigned int>(),Eigen::Vector3d(0,0,0)); usBinCounter++;
				_vBins.push_back(usIdx);
				//}else{
				//	_ppNormalHistogram[usIdx]=NULL;
				//}
				usIdx++;
			}
		}
	}
	PRINT(usBinCounter);
	return;
}
void btl::utility::SNormalHist::getNeighbourIdxCylinder(const ushort& usIdx_, std::vector< ushort >* pNeighbours_ )
{
	pNeighbours_->clear();
	pNeighbours_->push_back(usIdx_);
	return;
}
void btl::utility::SNormalHist::gpuNormalHistogram( const cv::gpu::GpuMat& cvgmNls_, const cv::Mat& cvmNls_, const ushort usPryLevel_,btl::utility::tp_coordinate_convention eCon_) {
	clear(usPryLevel_);
	_usMinArea = 30;
	//gpu calc hist idx
	btl::device::cudaNormalHistogramCV(cvgmNls_,
		_usSamplesAzimuth,
		_usSamplesElevationZ,
		_usWidth,
		_usLevel,
		_fBinSize,&*_acvgmScpPtrBinIdx[usPryLevel_]);
	//download to host
	_acvgmScpPtrBinIdx[usPryLevel_]->download(*_acvmScpPtrBinIdx[usPryLevel_]);
	const short* pIdx = (const short*)_acvmScpPtrBinIdx[usPryLevel_]->data;
	const float* pNl = (const float*) cvmNls_.data;
	int nCounter=0;int nCounter2 =0;int nCounter3 =0;

	for (unsigned int nIdx = 0; nIdx<btl::kinect::__aKinectWxH[usPryLevel_];nIdx++,pNl+=3 ){
		if(pIdx[nIdx]>0 ){
			nCounter++;
			if(pIdx[nIdx]>_usTotal||pIdx[nIdx]<0) nCounter3++;
			if( _ppNormalHistogram[pIdx[nIdx]]){
				nCounter2++;
				_ppNormalHistogram[pIdx[nIdx]]->first.push_back(nIdx);
				_ppNormalHistogram[pIdx[nIdx]]->second+=Eigen::Vector3d(pNl[0],pNl[1],pNl[2]);
			}
		}
	}
	//calculate the average normal
	for(std::vector<ushort>::const_iterator cit_vBins=_vBins.begin();cit_vBins<_vBins.end();cit_vBins++){
		if(_ppNormalHistogram[*cit_vBins] && _ppNormalHistogram[*cit_vBins]->first.size() > 0)
			_ppNormalHistogram[*cit_vBins]->second.normalize();
	}
}
void btl::utility::SNormalHist::clear( const unsigned short usPyrLevel_ )
{
	//clear histogram
	_acvgmScpPtrBinIdx[usPyrLevel_]->setTo(-1);
	for(std::vector<ushort>::const_iterator cit_vBins=_vBins.begin();cit_vBins<_vBins.end();cit_vBins++){
		if(_ppNormalHistogram[*cit_vBins]){
			_ppNormalHistogram[*cit_vBins]->first.clear();
			_ppNormalHistogram[*cit_vBins]->second.setZero();
		}
	}
}
void btl::utility::SNormalHist::clusterNormalHist(const cv::Mat& cvmNls_, const double dCosThreshold_, cv::Mat* pcvmLabel_, btl::geometry::tp_plane_obj_list* pvPlaneObjs_){
	//re-cluster the normals
	pvPlaneObjs_->clear();
	pcvmLabel_->setTo(-1);
	short nLabel =0;
	for(std::vector<ushort>::const_iterator cit_vBins=_vBins.begin();cit_vBins!=_vBins.end();cit_vBins++){
		if( !_ppNormalHistogram[*cit_vBins] || _ppNormalHistogram[*cit_vBins]->first.size() < _usMinArea ) continue;
		//get neighborhood of a sampling bin
		std::vector<unsigned short> vNeighourhood; 
		getNeighbourIdxCylinder(*cit_vBins,&vNeighourhood);
		//traverse the neighborhood and cluster the 
		btl::geometry::tp_plane_obj sPlane;
		std::vector<unsigned int> vLabelNormalIdx; Eigen::Vector3d eivAvgNl;
		for( std::vector<unsigned short>::const_iterator cit_vNeighbourhood=vNeighourhood.begin();cit_vNeighbourhood!=vNeighourhood.end();cit_vNeighbourhood++) {
			createNormalCluster(cvmNls_,_ppNormalHistogram[*cit_vNeighbourhood]->first,_ppNormalHistogram[*cit_vNeighbourhood]->second,dCosThreshold_,nLabel,
				/*out*/ pcvmLabel_,&sPlane._vIdx,&sPlane._eivAvgNormal);
		}
		nLabel++;
		pvPlaneObjs_->push_back(sPlane);
	}
}

void btl::utility::SNormalHist::createNormalCluster( const cv::Mat& cvmNls_,const std::vector< unsigned int >& vNlIdx_, 
	const Eigen::Vector3d& eivClusterCenter_, const double& dCosThreshold_, const short& sLabel_, cv::Mat* pcvmLabel_, std::vector< unsigned int >* pvNlIdx_, Eigen::Vector3d* pAvgNl_ ){
	//the pvLabel_ must be same length as vNormal_ 
	//with each element assigned with a NEGATIVE value
	const float* pNl = (const float*)cvmNls_.data; 
	short* pLabel = (short*) pcvmLabel_->data;
	pAvgNl_->setZero();
	for( std::vector< unsigned int >::const_iterator cit_vNlIdx_ = vNlIdx_.begin();	cit_vNlIdx_!= vNlIdx_.end(); cit_vNlIdx_++ ){
		int nOffset = (*cit_vNlIdx_)*3;
		if( pLabel[*cit_vNlIdx_]<0 && btl::utility::isNormalSimilar< float >(pNl+nOffset,eivClusterCenter_,dCosThreshold_) ) {
			pLabel[*cit_vNlIdx_] = sLabel_;
			pvNlIdx_->push_back(*cit_vNlIdx_);
			*pAvgNl_+=Eigen::Vector3d((pNl+nOffset)[0],(pNl+nOffset)[1],(pNl+nOffset)[2]);
		}//if
	}
	pAvgNl_->normalize();
	return;
}
void btl::utility::SNormalHist::gpuClusterNormal(const cv::gpu::GpuMat& cvgmNls,const cv::Mat& cvmNls,const unsigned short uPyrLevel_,cv::Mat* pcvmLabel_,btl::geometry::tp_plane_obj_list* pvPlaneObjs_){
	//define constants
	const double dCosThreshold = std::cos(M_PI_4/4);
	_usMinArea = 30;
	//make a histogram on the top pyramid
	gpuNormalHistogram(cvgmNls,cvmNls,uPyrLevel_,btl::utility::BTL_CV);
	clusterNormalHist(cvmNls,dCosThreshold,pcvmLabel_,pvPlaneObjs_);
	return;
}

void btl::utility::SDistanceHist::distanceHistogram(const cv::Mat& cvmPts_, const std::vector<unsigned int>& vPts_, const Eigen::Vector3d& eivAvgNl_ ){
	//collecting distance histogram for the current normal cluster
	_pvDistHist->clear();
	_pvDistHist->resize(_uSamples,tp_pair_hist_bin(std::vector<tp_pair_hist_element>(), 0.) );
	const float*const pPt = (float*) cvmPts_.data;
	//collect the distance histogram
	for(std::vector< unsigned int >::const_iterator cit_vPointIdx = vPts_.begin(); cit_vPointIdx!=vPts_.end(); cit_vPointIdx++){
		unsigned int uOffset = (*cit_vPointIdx)*3;
		double dDist = fabs(pPt[uOffset]*eivAvgNl_(0) + pPt[uOffset+1]*eivAvgNl_(1)+ pPt[uOffset+2]*eivAvgNl_(2));
		ushort nBin = (ushort)floor( (dDist -_dLow)/ _dSampleStep );
		if( nBin < _uSamples ){
			(*_pvDistHist)[nBin].first.push_back(tp_pair_hist_element(dDist,*cit_vPointIdx));
			(*_pvDistHist)[nBin].second += dDist;
		}
	}
	//calc avg distance 
	for(std::vector< tp_pair_hist_bin >::iterator cit_vDistHist = _pvDistHist->begin();	cit_vDistHist != _pvDistHist->end(); cit_vDistHist++ )	{
		unsigned int uBinSize = cit_vDistHist->first.size();
		if( uBinSize==0 ) continue;
		//calculate avg distance
		cit_vDistHist->second /= uBinSize;
	}//for each distance bin
	return;
}

void btl::utility::SDistanceHist::calcMergeFlag(){
	_vMergeFlags.resize(_uSamples, SDistanceHist::EMPTY);
	//merge the bins whose distance is similar
	std::vector< tp_flag >::iterator it_vMergeFlags = _vMergeFlags.begin()+1; 
	std::vector< tp_flag >::iterator it_prev;
	std::vector< tp_pair_hist_bin >::const_iterator cit_prev;
	std::vector< tp_pair_hist_bin >::const_iterator cit_endm1 = _pvDistHist->end() - 1;

	for(std::vector< tp_pair_hist_bin >::const_iterator cit_vDistHist = _pvDistHist->begin() + 1;
		cit_vDistHist != cit_endm1; cit_vDistHist++,it_vMergeFlags++ ) {
			unsigned int uBinSize = cit_vDistHist->first.size();
			if(0==uBinSize) continue;
			*it_vMergeFlags = NO_MERGE;
			cit_prev = cit_vDistHist -1;
			it_prev  = it_vMergeFlags-1;
			if( EMPTY == *it_prev ) continue;

			if( fabs(cit_prev->second - cit_vDistHist->second) < _dMergeDistance ){ //avg distance smaller than the sample step.
				//previou bin
				if     (NO_MERGE       ==*it_prev){	*it_prev = MERGE_WITH_RIGHT;}
				else if(MERGE_WITH_LEFT==*it_prev){ *it_prev = MERGE_WITH_BOTH; }
				//current bin
				*it_vMergeFlags = MERGE_WITH_LEFT;
			}//if mergable
	}//for each bin
}

void btl::utility::SDistanceHist::init( const unsigned short usSamples_ ){
	_uSamples=usSamples_;
	_dLow  =  0; //negative doesnot make sense
	_dHigh =  3;
	_dSampleStep = ( _dHigh - _dLow )/_uSamples; 
	_pvDistHist.reset(new tp_dist_hist);
	_vMergeFlags.resize(_uSamples, SDistanceHist::EMPTY); 
	//==0 no merging, ==1 merge with left, ==2 merge with right, ==3 merging with both
	_usMinArea = 10;
}

void btl::utility::SDistanceHist::clusterDistanceHist( const cv::Mat& cvmPts_, const cv::Mat& cvmNls_, const unsigned short usPyrLevel_, const btl::geometry::tp_plane_obj_list& vInPlaneObjs_, cv::Mat* pcvmDistanceClusters_, btl::geometry::tp_plane_obj_list* pOutPlaneObjs_ )
{
	_usMinArea = 30;
	pOutPlaneObjs_->clear();
	pcvmDistanceClusters_->setTo(-1);
	//construct the label mat
	short sLabel = 0;
	for(btl::geometry::tp_plane_obj_list::const_iterator cit_vPlaneObj = vInPlaneObjs_.begin(); cit_vPlaneObj!=vInPlaneObjs_.end(); cit_vPlaneObj++){
		//collect 
		distanceHistogram( cvmPts_, cit_vPlaneObj->_vIdx, cit_vPlaneObj->_eivAvgNormal );
		calcMergeFlag(); // EMPTY/NO_MERGE/MERGE_WITH_LEFT/MERGE_WITH_BOTH/MERGE_WITH_RIGHT 
		//cluster
		mergeDistanceBins( cvmNls_, &sLabel, pcvmDistanceClusters_, pOutPlaneObjs_ );
		sLabel++;
	}//for each plane object clustered by normals
}

void btl::utility::SDistanceHist::mergeDistanceBins( const cv::Mat& cvmNls_, short* pLabel_, cv::Mat* pcvmLabel_, btl::geometry::tp_plane_obj_list* pPlaneObjs ){
	//
	std::vector< tp_flag >::const_iterator cit_vMergeFlags = _vMergeFlags.begin();
	std::vector< tp_pair_hist_bin >::const_iterator cit_endm1 = _pvDistHist->end() - 1;
	short* pDistanceLabel = (short*) pcvmLabel_->data;
	const float* pNls= (const float*)cvmNls_.data;
	btl::geometry::tp_plane_obj sPlane;sPlane._eivAvgNormal.setZero();sPlane._dAvgPosition=0;
	for(std::vector< tp_pair_hist_bin >::const_iterator cit_vDistHist = _pvDistHist->begin() + 1; cit_vDistHist != cit_endm1; cit_vDistHist++,cit_vMergeFlags++ ){
		if(EMPTY==*cit_vMergeFlags) continue;
		if(NO_MERGE==*cit_vMergeFlags||MERGE_WITH_RIGHT==*cit_vMergeFlags||MERGE_WITH_BOTH==*cit_vMergeFlags||MERGE_WITH_LEFT==*cit_vMergeFlags){
			if(cit_vDistHist->first.size()>_usMinArea){
				for( std::vector<tp_pair_hist_element>::const_iterator cit_vPair = cit_vDistHist->first.begin();cit_vPair != cit_vDistHist->first.end(); cit_vPair++ ){
					pDistanceLabel[cit_vPair->second] = *pLabel_;//labeling
					sPlane._vIdx.push_back(cit_vPair->second);//store pts
					sPlane._dAvgPosition += cit_vPair->first;//accumulate distance
					unsigned int nOffset = cit_vPair->second*3;
					sPlane._eivAvgNormal += Eigen::Vector3d(pNls[nOffset],pNls[nOffset+1],pNls[nOffset+2]);//accumulate normals
				}//for each distance bin 
			}//if large enough
		}//if mergable
		if(sPlane._vIdx.size()>0 && (NO_MERGE==*cit_vMergeFlags||MERGE_WITH_LEFT==*cit_vMergeFlags)){
			//calc the average 
			sPlane._eivAvgNormal.normalize();
			sPlane._dAvgPosition/=sPlane._vIdx.size();
			sPlane._usIdx = *pLabel_;
			//store plane objects
			pPlaneObjs->push_back(sPlane);
			//reset plane objects
			sPlane._dAvgPosition = 0;
			sPlane._eivAvgNormal.setZero();
			sPlane._vIdx.clear();
			//increase the labeling
			(*pLabel_)++;
		}//if merging ends
	}//for
	return;
}
