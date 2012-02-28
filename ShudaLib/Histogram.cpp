#define INFO
#include <boost/scoped_ptr.hpp>
#include <vector>

#include <opencv2/gpu/gpu.hpp>
#include <Eigen/Core>
#include "OtherUtil.hpp"
#include "Kinect.h"
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
	//gpu calc hist idx
	btl::cuda_util::cudaNormalHistogram(cvgmNls_,
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
void btl::utility::SNormalHist::normalHistogram( const cv::Mat& cvmNls_, int nSamples_, btl::utility::tp_coordinate_convention eCon_) {
	//clear and re-initialize pvvIdx_
	int nSampleAzimuth = nSamples_<<2; //nSamples*4
	_vNormalHistogram.clear();
	_vNormalHistogram.resize(nSamples_*nSampleAzimuth,btl::utility::SNormalHist::tp_normal_hist_bin(std::vector<unsigned int>(),Eigen::Vector3d(0,0,0)));
	const double dS = M_PI_2/nSamples_;//sampling step
	int r,c,rc;
	const float* pNl = (const float*) cvmNls_.data;
	for(unsigned int i =0; i< cvmNls_.total(); i++, pNl+=3)	{
		if( pNl[2]>0 || fabs(pNl[0])+fabs(pNl[1])+fabs(pNl[2])<0.0001 ) {continue;}
		btl::utility::normalVotes<float>(pNl,dS,&r,&c,eCon_);
		rc = r*nSampleAzimuth+c;
		if(rc<0||rc>_vNormalHistogram.size()){continue;}
		_vNormalHistogram[rc].first.push_back(i);
		_vNormalHistogram[rc].second += Eigen::Vector3d(pNl[0],pNl[1],pNl[2]);
	}
	//average the 
	for(std::vector<btl::utility::SNormalHist::tp_normal_hist_bin>::iterator it_vNormalHist = _vNormalHistogram.begin();
		it_vNormalHist!=_vNormalHistogram.end(); it_vNormalHist++) {
			if(it_vNormalHist->first.size()>0) {
				it_vNormalHist->second.normalize();
			}
	}

	return;
}

void btl::utility::SDistanceHist::distanceHistogram( const cv::Mat& cvmNls_, const cv::Mat& cvmPts_,  
	const std::vector< unsigned int >& vIdx_ )
{
	_pvDistHist->clear();
	_pvDistHist->resize(_uSamples,tp_pair_hist_bin(std::vector<tp_pair_hist_element>(), 0.) );
	const float*const pPt = (float*) cvmPts_.data;
	const float*const pNl = (float*) cvmNls_.data;
	//collect the distance histogram
	for(std::vector< unsigned int >::const_iterator cit_vPointIdx = vIdx_.begin(); cit_vPointIdx!=vIdx_.end(); cit_vPointIdx++){
		unsigned int uOffset = (*cit_vPointIdx)*3;
		double dDist = pPt[uOffset]*pNl[uOffset] + pPt[uOffset+1]*pNl[uOffset+1] + pPt[uOffset+2]*pNl[uOffset+2];
		ushort nBin = (ushort)floor( fabs(dDist -_dLow)/ _dSampleStep );
		if( nBin >= 0 && nBin < _uSamples){
			(*_pvDistHist)[nBin].first.push_back(tp_pair_hist_element(dDist,*cit_vPointIdx));
			(*_pvDistHist)[nBin].second += dDist;
		}
	}

	//calc the avg distance for each bin 
	//construct a list for sorting
	for(std::vector< tp_pair_hist_bin >::iterator cit_vDistHist = _pvDistHist->begin();
		cit_vDistHist != _pvDistHist->end(); cit_vDistHist++ )	{
		unsigned int uBinSize = cit_vDistHist->first.size();
		if( uBinSize==0 ) continue;
		//calculate avg distance
		cit_vDistHist->second /= uBinSize;
	}
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

void btl::utility::SDistanceHist::mergeDistanceBins( const std::vector< unsigned int >& vLabelPointIdx_, short* pLabel_, cv::Mat* pcvmLabel_ ){
	std::vector< tp_flag >::const_iterator cit_vMergeFlags = _vMergeFlags.begin();
	std::vector< tp_pair_hist_bin >::const_iterator cit_endm1 = _pvDistHist->end() - 1;
	short* pDistanceLabel = (short*) pcvmLabel_->data;
	for(std::vector< tp_pair_hist_bin >::const_iterator cit_vDistHist = _pvDistHist->begin() + 1;
		cit_vDistHist != cit_endm1; cit_vDistHist++,cit_vMergeFlags++ )	{
			if(EMPTY==*cit_vMergeFlags) continue;
			if(NO_MERGE==*cit_vMergeFlags||MERGE_WITH_RIGHT==*cit_vMergeFlags||MERGE_WITH_BOTH==*cit_vMergeFlags||MERGE_WITH_LEFT==*cit_vMergeFlags){
					if(cit_vDistHist->first.size()>_usMinArea){
						for( std::vector<tp_pair_hist_element>::const_iterator cit_vPair = cit_vDistHist->first.begin();
							cit_vPair != cit_vDistHist->first.end(); cit_vPair++ ){
								pDistanceLabel[cit_vPair->second] = *pLabel_;
						}//for 
					}//if
			}
			if(NO_MERGE==*cit_vMergeFlags||MERGE_WITH_LEFT==*cit_vMergeFlags){
				(*pLabel_)++;
			}
	}//for
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
