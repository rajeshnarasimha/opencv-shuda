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