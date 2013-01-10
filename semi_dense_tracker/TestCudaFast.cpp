#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Helper.hpp"

__device__ short2 operator + (const short2 s2O1_, const short2 s2O2_);
__device__ short2 operator - (const short2 s2O1_, const short2 s2O2_);
__device__ float2 operator * (const float fO1_, const short2 s2O2_);
__device__ __host__ float2 operator + (const float2 f2O1_, const float2 f2O2_);
__device__ __host__ float2 operator - (const float2 f2O1_, const float2 f2O2_);
__device__  short2 convert2s2(const float2 f2O1_);

float dL1(const int4& n4Descriptor1_, const int4& n4Descriptor2_){
	float fDist = 0.f;
	uchar uD1,uD2;
	for (uchar u=0; u < 4; u++){
		uD1 = (n4Descriptor1_.x >> u*8) & 0xFF;
		uD2 = (n4Descriptor2_.x >> u*8) & 0xFF;
		fDist += abs(uD1 - uD2); 
		uD1 = (n4Descriptor1_.y >> u*8) & 0xFF;
		uD2 = (n4Descriptor2_.y >> u*8) & 0xFF;
		fDist += abs(uD1 - uD2); 
		uD1 = (n4Descriptor1_.z >> u*8) & 0xFF;
		uD2 = (n4Descriptor2_.z >> u*8) & 0xFF;
		fDist += abs(uD1 - uD2); 
		uD1 = (n4Descriptor1_.w >> u*8) & 0xFF;
		uD2 = (n4Descriptor2_.w >> u*8) & 0xFF;
		fDist += abs(uD1 - uD2); 
	}
	fDist /= 16;
	return fDist;
}
void devGetFastDescriptor(const cv::Mat& cvgmImage_, const int r, const int c, int4* pDescriptor_ ){
	pDescriptor_->x = pDescriptor_->y = pDescriptor_->z = pDescriptor_->w = 0;
	uchar Color;
	Color = cvgmImage_.ptr<uchar>(r-3)[c  ];//1
	pDescriptor_->x += Color; 
	pDescriptor_->x = pDescriptor_->x << 8;
	Color = cvgmImage_.ptr<uchar>(r-3)[c+1];//2
	pDescriptor_->x += Color; 
	pDescriptor_->x = pDescriptor_->x << 8;
	Color = cvgmImage_.ptr<uchar>(r-2)[c+2];//3
	pDescriptor_->x += Color; 
	pDescriptor_->x = pDescriptor_->x << 8;
	Color = cvgmImage_.ptr<uchar>(r-1)[c+3];//4
	pDescriptor_->x += Color; 

	Color = cvgmImage_.ptr<uchar>(r  )[c+3];//5
	pDescriptor_->y += Color; 
	pDescriptor_->y = pDescriptor_->y << 8;
	Color = cvgmImage_.ptr<uchar>(r+1)[c+3];//6
	pDescriptor_->y += Color; 
	pDescriptor_->y = pDescriptor_->y << 8;
	Color = cvgmImage_.ptr<uchar>(r+2)[c+2];//7
	pDescriptor_->y += Color; 
	pDescriptor_->y = pDescriptor_->y << 8;
	Color = cvgmImage_.ptr<uchar>(r+3)[c+1];//8
	pDescriptor_->y += Color; 

	Color = cvgmImage_.ptr<uchar>(r+3)[c  ];//9
	pDescriptor_->z += Color; 
	pDescriptor_->z = pDescriptor_->z << 8;
	Color= cvgmImage_.ptr<uchar>(r+3)[c-1];//10
	pDescriptor_->z += Color; 
	pDescriptor_->z = pDescriptor_->z << 8;
	Color= cvgmImage_.ptr<uchar>(r+2)[c-2];//11
	pDescriptor_->z += Color; 
	pDescriptor_->z = pDescriptor_->z << 8;
	Color= cvgmImage_.ptr<uchar>(r+1)[c-3];//12
	pDescriptor_->z += Color; 

	Color= cvgmImage_.ptr<uchar>(r  )[c-3];//13
	pDescriptor_->w += Color; 
	pDescriptor_->w = pDescriptor_->w << 8;
	Color= cvgmImage_.ptr<uchar>(r-1)[c-3];//14
	pDescriptor_->w += Color; 
	pDescriptor_->w = pDescriptor_->w << 8;
	Color= cvgmImage_.ptr<uchar>(r-2)[c-2];//15
	pDescriptor_->w += Color; 
	pDescriptor_->w = pDescriptor_->w << 8;
	Color= cvgmImage_.ptr<uchar>(r-3)[c-1];//16
	pDescriptor_->w += Color; 
	return;
}

class CTestPredictAndMatch{
public:
	cv::Mat_<int4>   _cvmParticleDescriptorsPrev;
	cv::Mat_<float>  _cvmParticleResponsesPrev;
	cv::Mat_<uchar>  _cvmParticlesAgePrev;
	cv::Mat_<short2> _cvmParticlesVelocityPrev;

	cv::Mat_<uchar>  _cvmImageCurr;
	cv::Mat_<int4>   _cvmParticleDescriptorsCurr;
	cv::Mat_<float>  _cvmParticleResponsesCurr;
	cv::Mat_<uchar>  _cvmParticlesAgeCurr;
	cv::Mat_<short2> _cvmParticlesVelocityCurr;

	float _fRho;

	float _fMatchThreshold;
	short _sSearchRange;
	unsigned short _usHalfSizeRound;
	unsigned int _devuNewlyAddedCounter;
	unsigned int _devuCounter;



__device__ __forceinline__ float devMatch(const float& fMatchThreshold_, const int4& n4DesPrev_, const short sSearchRange_,
	short2* ps2Loc_,int4 *pn4Descriptor_ ){
	float fResponse = 0.f;
	float fBestMatchedResponse;
	short2 s2Loc,s2BestLoc;
	int4 n4BestDescriptor;
	float fMinDist = 300.f;
	//search for the 7x7 neighbourhood for 
	for(short r = -sSearchRange_; r <= sSearchRange_; r++ ){
		for(short c = -sSearchRange_; c <= sSearchRange_; c++ ){
			s2Loc = *ps2Loc_ + make_short2( c, r ); 
			if(s2Loc.x < _usHalfSizeRound || s2Loc.x >= _cvmImageCurr.cols - _usHalfSizeRound || s2Loc.y < _usHalfSizeRound || s2Loc.y >= _cvmImageCurr.rows - _usHalfSizeRound ) continue;
			fResponse = _cvmParticleResponsesCurr.ptr<float>(s2Loc.y)[s2Loc.x];
			if( fResponse > 0.1f ){
				int4 n4Des; 
				devGetFastDescriptor(_cvmImageCurr,s2Loc.y,s2Loc.x,&n4Des);
				float fDist = dL1(n4Des,n4DesPrev_);
				if ( fDist < fMatchThreshold_ ){
					if (  fMinDist > fDist ){
						fMinDist = fDist;
						fBestMatchedResponse = fResponse;
						n4BestDescriptor = n4Des;
						s2BestLoc = s2Loc;		 
					}
				}
			}//if sailent corner exits
		}//for 
	}//for
	if(fMinDist < 300.f){
		*ps2Loc_ = s2BestLoc;
		*pn4Descriptor_ = n4BestDescriptor;
		return fBestMatchedResponse;
	}
	else{
		return -1.f;
	}
}//devMatch
__device__ __forceinline__ void operator () (){
	for (int c = 3; c<_cvmImageCurr.cols-4; c++ ){
		for (int r = 3; r<_cvmImageCurr.rows-4; r++ ){
			//if IsParticle( PixelLocation, cvgmParitclesResponse(i) )
			if ( _cvmParticleResponsesPrev.ptr<float>(r)[c] < 0.2f ) continue;
			//A) PredictLocation = PixelLocation + ParticleVelocity(i, PixelLocation);
			short2 s2PredictLoc = make_short2(c,r);// + cvmParticlesVelocityPrev.ptr<short2>(r)[c];
			//check whether the predicted point is inside the image
			//B) ActualLocation = Match(PredictLocation, cvgmBlurred(i),cvgmBlurred(i+1));
			int4 n4DesPrev = _cvmParticleDescriptorsPrev.ptr<int4>(s2PredictLoc.y)[s2PredictLoc.x];//;	devGetFastDescriptor(cvmBlurredPrev,r,c,&n4DesPrev);
			int4 n4DenCurr;
			float fResponse = devMatch( 70.f, n4DesPrev, _sSearchRange, &s2PredictLoc, &n4DenCurr );

			if(fResponse > 0.1f){
				_devuNewlyAddedCounter++;
				_cvmParticleDescriptorsCurr.ptr<int4>(s2PredictLoc.y)[s2PredictLoc.x] = n4DenCurr;
				_cvmParticlesVelocityCurr.ptr<short2>(s2PredictLoc.y)[s2PredictLoc.x] = convert2s2( _fRho * (s2PredictLoc - make_short2(c,r)) + (1.f - _fRho)* _cvmParticlesVelocityPrev.ptr<short2>(r)[c] + make_float2(.5f,.5f));//update velocity
				_cvmParticlesAgeCurr.ptr			 (s2PredictLoc.y)[s2PredictLoc.x] = _cvmParticlesAgePrev.ptr(r)[c] + 1; //update age
				_cvmParticleResponsesCurr.ptr<float> (s2PredictLoc.y)[s2PredictLoc.x] = -fResponse; //update response and location //marked as matched and it will be corrected in NoMaxAndCollection
			}
			else{//C) if no match found 
				_devuCounter ++;
			}//lost
		}//for
	}//for
	return;
}//operator()
};
unsigned int testCudaTrack(const float fMatchThreshold_, const short sSearchRange_, 
	const cv::gpu::GpuMat& cvgmParticleDescriptorsPrev_, const cv::gpu::GpuMat& cvgmParticleResponsesPrev_,
	const cv::gpu::GpuMat& cvgmParticlesAgePrev_,const cv::gpu::GpuMat& cvgmParticlesVelocityPrev_, 
	const cv::gpu::GpuMat& cvgmBlurredCurr_,
	cv::gpu::GpuMat* pcvgmSaliency_,
	cv::gpu::GpuMat* pcvgmParticlesAgeCurr_,cv::gpu::GpuMat* pcvgmParticlesVelocityCurr_,cv::gpu::GpuMat* pcvgmParticleDescriptorsCurr_){
	
	CTestPredictAndMatch cTPAM;
	
	cvgmBlurredCurr_.download(cTPAM._cvmImageCurr);
	cvgmParticleResponsesPrev_.download(cTPAM._cvmParticleResponsesPrev);
	cvgmParticlesVelocityPrev_.download(cTPAM._cvmParticlesVelocityPrev);
	cvgmParticlesAgePrev_.download(cTPAM._cvmParticlesAgePrev);
	cvgmParticleDescriptorsPrev_.download(cTPAM._cvmParticleDescriptorsPrev); 

	pcvgmSaliency_->download(cTPAM._cvmParticleResponsesCurr);
	pcvgmParticlesVelocityCurr_->download(cTPAM._cvmParticlesVelocityCurr);
	pcvgmParticlesAgeCurr_->download(cTPAM._cvmParticlesAgeCurr);
	pcvgmParticleDescriptorsCurr_->download(cTPAM._cvmParticleDescriptorsCurr);


	cTPAM._fRho = 0.75f;
	cTPAM._fMatchThreshold = fMatchThreshold_;
	cTPAM._sSearchRange = sSearchRange_;

	cTPAM._devuCounter = 0;
	cTPAM._devuNewlyAddedCounter = 0;
	cTPAM();
	unsigned int uDeleted = cTPAM._devuCounter;
	unsigned int uMatched = cTPAM._devuNewlyAddedCounter;
	return uDeleted;
}




