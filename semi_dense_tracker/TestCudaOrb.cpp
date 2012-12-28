#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <cuda.h>
#include <cuda_runtime.h>


__device__ short2 operator + (const short2 s2O1_, const short2 s2O2_);
__device__ short2 operator - (const short2 s2O1_, const short2 s2O2_);
__device__ short2 operator * (const float fO1_, const short2 s2O2_);

#define GET_VALUE(idx) \
	cvgmImg_.ptr<uchar>(s2Loc_.y + cvRound(pnPatternX_[idx] * sina + pnPatternY_[idx] * cosa))\
	[s2Loc_.x + cvRound(pnPatternX_[idx] * cosa - pnPatternY_[idx] * sina)]


struct OrbDescriptor
{
	__device__ static unsigned char calc(const cv::Mat& cvgmImg_, short2 s2Loc_, const short* pnPatternX_, const short* pnPatternY_, float sina, float cosa, int nDescIdx_)
	{
		pnPatternX_ += 16 * nDescIdx_; //compare 8 pairs of points, and that is 16 points in total
		pnPatternY_ += 16 * nDescIdx_;

		int t0, t1;
		unsigned char val;

		t0 = GET_VALUE(0); t1 = GET_VALUE(1);
		val = t0 < t1;

		t0 = GET_VALUE(2); t1 = GET_VALUE(3);
		val |= (t0 < t1) << 1;

		t0 = GET_VALUE(4); t1 = GET_VALUE(5);
		val |= (t0 < t1) << 2;

		t0 = GET_VALUE(6); t1 = GET_VALUE(7);
		val |= (t0 < t1) << 3;

		t0 = GET_VALUE(8); t1 = GET_VALUE(9);
		val |= (t0 < t1) << 4;

		t0 = GET_VALUE(10); t1 = GET_VALUE(11);
		val |= (t0 < t1) << 5;

		t0 = GET_VALUE(12); t1 = GET_VALUE(13);
		val |= (t0 < t1) << 6;

		t0 = GET_VALUE(14); t1 = GET_VALUE(15);
		val |= (t0 < t1) << 7;

		return val;
	}
};
namespace btl{ namespace device{ namespace semidense{
void cudaCalcAngles(const cv::gpu::GpuMat& cvgmImage_, const short2* pdevFinalKeyPointsLocations_, const unsigned int uPoints_,  const unsigned short usHalf_, 
	cv::gpu::GpuMat* pcvgmParticleAngle_);
}//semidense
}//device
}//btl


float testMatDiff(const cv::gpu::GpuMat& cvgm1_,const cv::gpu::GpuMat& cvgm2_ ){
	cv::Mat cvm1; cvgm1_.download(cvm1);
	cv::Mat cvm2; cvgm2_.download(cvm2);
	cvm1 = cvm1-cvm2;
	if (cvm1.type() == CV_32SC2){
		int nSum = 0;
		int* p = (int*)cvm1.data;
		for (unsigned int i=0; i<cvm1.total(); i++){
			nSum += abs(*p++); nSum += abs(*p++);
		}
		return float(nSum);
	}
	else if(cvm1.type() == CV_32FC1){
		float* p = (float*)cvm1.data;
		float fSum = 0.f;
		for (unsigned int i=0; i<cvm1.total(); i++){
			fSum += abs(*p++);
		}
		return fSum;
	}
	return -1.f;
}


void testCudaCollectParticlesAndOrbDescriptors(const cv::gpu::GpuMat& cvgmFinalKeyPointsLocationsAfterNonMax_, const cv::gpu::GpuMat& cvmFinalKeyPointsResponseAfterNonMax_, const cv::gpu::GpuMat& cvgmImage_,
	const unsigned int uTotalParticles_, const unsigned short usHalfPatchSize_,
	const cv::gpu::GpuMat& cvgmPattern_,
	cv::gpu::GpuMat* pcvgmParticleResponses_, cv::gpu::GpuMat* pcvgmParticleAngle_, cv::gpu::GpuMat* pcvgmParticleDescriptor_)
{

	btl::device::semidense::cudaCalcAngles(cvgmImage_, cvgmFinalKeyPointsLocationsAfterNonMax_.ptr<const short2>(0), uTotalParticles_,  usHalfPatchSize_, pcvgmParticleAngle_);


	cv::Mat cvmFinalKeyPointsLocationsAfterNonMax; cvgmFinalKeyPointsLocationsAfterNonMax_.download(cvmFinalKeyPointsLocationsAfterNonMax);
	cv::Mat cvmFinalKeyPointsResponseAfterNonMax;  cvmFinalKeyPointsResponseAfterNonMax_.download(cvmFinalKeyPointsResponseAfterNonMax);
	cv::Mat cvmBlurredPrev;                        cvgmImage_.download(cvmBlurredPrev);
	cv::Mat cvmPattern;							   cvgmPattern_.download(cvmPattern);
	cv::Mat cvmParticleAnglePrev;				   pcvgmParticleAngle_->download(cvmParticleAnglePrev);
	cv::Mat cvmParticleOrbDescriptorsPrev;         pcvgmParticleDescriptor_->download(cvmParticleOrbDescriptorsPrev);
	cv::Mat cvmParticleResponsesPrev;			   pcvgmParticleResponses_->download(cvmParticleResponsesPrev);
	for (unsigned int nKeyPointIdx=0; nKeyPointIdx <uTotalParticles_; nKeyPointIdx++ ){
		for (int nDescIdx = 0; nDescIdx <8; nDescIdx ++){
			const short2& s2Loc = cvmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(0)[nKeyPointIdx];
			if(s2Loc.x < usHalfPatchSize_*2 || s2Loc.x >= cvmBlurredPrev.cols- usHalfPatchSize_*2 || s2Loc.y < usHalfPatchSize_*2 || s2Loc.y >= cvmBlurredPrev.rows - usHalfPatchSize_*2) continue;
			cvmParticleResponsesPrev.ptr<float>(s2Loc.y)[s2Loc.x] = cvmFinalKeyPointsResponseAfterNonMax.ptr<float>(0)[nKeyPointIdx];
			float fAngle = cvmParticleAnglePrev.ptr<float>(s2Loc.y)[s2Loc.x];
			float fSina = sin(fAngle), fCosa = cos(fAngle);
			uchar ucDesc = OrbDescriptor::calc(cvmBlurredPrev, s2Loc, cvmPattern.ptr<short>(0), cvmPattern.ptr<short>(1), fSina, fCosa, nDescIdx);
			uchar* pD = (uchar*)(cvmParticleOrbDescriptorsPrev.ptr<int2>(s2Loc.y)+ s2Loc.x);
			pD[nDescIdx]= ucDesc;
		}
	}
	pcvgmParticleResponses_->upload(cvmParticleResponsesPrev);
	pcvgmParticleDescriptor_->upload(cvmParticleOrbDescriptorsPrev);
}

__constant__ uchar _popCountTable[] =
{
	0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};

class CPredictAndMatchOrb{
public:
	cv::Mat_<int2>   _cvmParticleOrbDescriptorsPrev;
	cv::Mat_<float>  _cvmParticleResponsesPrev;
	cv::Mat_<uchar>  _cvmParticlesAgePrev;
	cv::Mat_<short2> _cvmParticlesVelocityPrev;

	cv::Mat_<uchar>  _cvmImage;
	cv::Mat_<float>  _cvmParticleAnglesCurr;
	cv::Mat_<int2>   _cvmParticleOrbDescriptorsCurr;
	cv::Mat_<float>  _cvmParticleResponsesCurr;
	cv::Mat_<uchar>  _cvmParticlesAgeCurr;
	cv::Mat_<short2> _cvmParticlesVelocityCurr;

	float _fRho;

	unsigned short _usMatchThreshold;
	unsigned short _usHalfSize;
	short _sSearchRange;
	const short* _psPatternX;
	const short* _psPatternY;

	unsigned int _devuMathchedCounter;
	unsigned int _devuDeletedCounter;

	unsigned int _devuOther;

	__device__ __forceinline__ uchar dL(const uchar* pDesPrev_, const uchar* pDesCurr_) const{
		uchar ucRes = 0;
		for(short s = 0; s<8; s++)
			ucRes += _popCountTable[ pDesPrev_[s] ^ pDesCurr_[s] ];
		return ucRes;
	}

__device__ __forceinline__ float devMatchOrb( const unsigned short usMatchThreshold_, 
	const uchar* pDesPrev_, short2* ps2Loc_, uchar* pDesCur_ ){
		float fResponse = 0.f;
		float fAngle;
		float fBestMatchedResponse;
		short2 s2Loc,s2BestLoc;
		uchar ucMinDist = 255;
		uchar aDesCur[8] = {0,0,0,0, 0,0,0,0};
		//search for the 7x7 neighbourhood for 
		for(short r = -_sSearchRange; r < _sSearchRange+1; r++ ){
			for(short c = -_sSearchRange; c < _sSearchRange+1; c++ ){
				_devuOther++;
				s2Loc = *ps2Loc_ + make_short2( c, r ); 
				fResponse = _cvmParticleResponsesCurr.ptr<float>(s2Loc.y)[s2Loc.x];
				if( fResponse > 0 ){
					
					fAngle = _cvmParticleAnglesCurr.ptr<float>(s2Loc.y)[s2Loc.x];
					float fSina = sin(fAngle), fCosa = cos(fAngle); 
					for(int s = 0; s<8; s++)
						aDesCur[s] = OrbDescriptor::calc(_cvmImage, s2Loc, _psPatternX, _psPatternY, fSina, fCosa, s);
					uchar ucDist = dL(pDesPrev_,aDesCur);
					if ( ucDist < usMatchThreshold_ ){
						if (  ucMinDist > ucDist ){
							ucMinDist = ucDist;
							fBestMatchedResponse = fResponse;
							s2BestLoc = s2Loc;
							pDesCur_[0] = aDesCur[0];pDesCur_[1] = aDesCur[1];pDesCur_[2] = aDesCur[2];pDesCur_[3] = aDesCur[3];pDesCur_[4] = aDesCur[4];pDesCur_[5] = aDesCur[5];pDesCur_[6] = aDesCur[6];pDesCur_[7] = aDesCur[7];
						}
					}
				}//if sailent corner exits
			}//for 
		}//for
		if(ucMinDist < 255){
			*ps2Loc_ = s2BestLoc;
			return fBestMatchedResponse;
		}
		else
			return -1.f;
}
__device__ __forceinline__ void operator () (){
	_devuDeletedCounter =0;
	_devuMathchedCounter=0;
	_devuOther=0;
	for (int r=0; r<_cvmImage.rows; r++ ){
		for (int c=0; c<_cvmImage.cols; c++ ){
			if( c < 3 || c >= _cvmImage.cols - 4 || r < 3 || r >= _cvmImage.rows - 4 ) continue;

			//if IsParticle( PixelLocation, cvgmParitclesResponse(i) )
			if(_cvmParticleResponsesPrev.ptr<float>(r)[c] < 0.2f) continue;
			//A) PredictLocation = PixelLocation + ParticleVelocity(i, PixelLocation);
			short2 s2PredictLoc = make_short2(c,r);// + _cvgmParticlesVelocityPrev.ptr(r)[c];
			//B) ActualLocation = Match(PredictLocation, cvgmBlurred(i),cvgmBlurred(i+1));
			if (s2PredictLoc.x >=12 && s2PredictLoc.x < _cvmImage.cols-13 && s2PredictLoc.y >=12 && s2PredictLoc.y < _cvmImage.rows-13){
				//;	devGetFastDescriptor(_cvgmBlurredPrev,r,c,&n4DesPrev);
				const uchar* pDesPrev = (uchar*) ( _cvmParticleOrbDescriptorsPrev.ptr<int2>(r)+c);
				uchar* pDesCur = (uchar*)(_cvmParticleOrbDescriptorsCurr.ptr<int2>(s2PredictLoc.y)+ s2PredictLoc.x);
				float fResponse = devMatchOrb( _usMatchThreshold, pDesPrev, &s2PredictLoc, &*pDesCur );

				if( fResponse > 0 ){
					_devuMathchedCounter++;
					_cvmParticleOrbDescriptorsCurr.ptr<int2>(s2PredictLoc.y)[s2PredictLoc.x]=*((int2*)pDesCur);
					_cvmParticlesVelocityCurr.ptr<short2>(s2PredictLoc.y)[s2PredictLoc.x] = _fRho * (s2PredictLoc - make_short2(c,r)) + (1.f - _fRho)* _cvmParticlesVelocityPrev.ptr<short2>(r)[c];//update velocity
					_cvmParticlesAgeCurr.ptr             (s2PredictLoc.y)[s2PredictLoc.x] = _cvmParticlesAgePrev.ptr(s2PredictLoc.y)[s2PredictLoc.x] + 1; //update age
					_cvmParticleResponsesCurr.ptr<float> (s2PredictLoc.y)[s2PredictLoc.x] = -fResponse; //update response and location //marked as matched and it will be corrected in NoMaxAndCollection
				}
				else{//C) if no match found 
					((int2*)pDesCur)->x = ((int2*)pDesCur)->y = 0;  
					_devuDeletedCounter++;
				}//lost
			}
			else{
				_devuDeletedCounter++;
			}
		}
	}
	return;
}
};//class

unsigned int testCudaTrackOrb(const unsigned short usMatchThreshold_, const unsigned short usHalfSize_,const unsigned short sSearchRange_,
const short* psPatternX_, const short* psPatternY_,
const cv::gpu::GpuMat& cvgmParticleOrbDescriptorsPrev_, const cv::gpu::GpuMat& cvgmParticleResponsesPrev_, 
const cv::gpu::GpuMat& cvgmParticlesAgePrev_,const cv::gpu::GpuMat& cvgmParticlesVelocityPrev_, 
const cv::gpu::GpuMat& cvgmImage_, const cv::gpu::GpuMat& cvgmParticleAngleCurr_,
cv::gpu::GpuMat* pcvgmParticleResponsesCurr_,
cv::gpu::GpuMat* pcvgmParticlesAgeCurr_,cv::gpu::GpuMat* pcvgmParticlesVelocityCurr_,cv::gpu::GpuMat* pcvgmParticleOrbDescriptorsCurr_){

	CPredictAndMatchOrb cPAMO;
	cvgmImage_.download(cPAMO._cvmImage);
	cvgmParticleResponsesPrev_.download(cPAMO._cvmParticleResponsesPrev);
	cvgmParticleOrbDescriptorsPrev_.download(cPAMO._cvmParticleOrbDescriptorsPrev);
	cvgmParticlesVelocityPrev_.download(cPAMO._cvmParticlesVelocityPrev); 
	cvgmParticlesAgePrev_.download(cPAMO._cvmParticlesAgePrev); 

	cvgmParticleAngleCurr_.download(cPAMO._cvmParticleAnglesCurr);
	pcvgmParticleOrbDescriptorsCurr_->download(cPAMO._cvmParticleOrbDescriptorsCurr);
	pcvgmParticleResponsesCurr_->download(cPAMO._cvmParticleResponsesCurr);
	pcvgmParticlesVelocityCurr_->download(cPAMO._cvmParticlesVelocityCurr);
	pcvgmParticlesAgeCurr_->download(cPAMO._cvmParticlesAgeCurr);

	cPAMO._fRho = .75f;
	cPAMO._usMatchThreshold = usMatchThreshold_;
	cPAMO._usHalfSize = usHalfSize_;
	cPAMO._sSearchRange = sSearchRange_;
	cPAMO._psPatternX = psPatternX_;
	cPAMO._psPatternY = psPatternY_;
	cPAMO();

	return cPAMO._devuDeletedCounter;
}