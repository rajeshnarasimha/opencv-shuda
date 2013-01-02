#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

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
	else if(cvm1.type() == CV_8UC1){
		unsigned char* p = cvm1.data;
		float fSum = 0.f;
		for (unsigned int i=0; i<cvm1.total(); i++){
			fSum += abs(*p++);
		}
		return fSum;
	}
	else if (cvm1.type() == CV_16SC2){
		short nSum = 0;
		short* p = (short*)cvm1.data;
		for (unsigned int i=0; i<cvm1.total(); i++){
			nSum += abs(*p++); nSum += abs(*p++);
		}
		return float(nSum);
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

bool testCountParticlesAndOrbDescriptors(const cv::gpu::GpuMat cvgmParticleResponses_, const cv::gpu::GpuMat& cvgmParticleAngle_, const cv::gpu::GpuMat& cvgmParticleDescriptor_, int* pnCounter_){
	cv::Mat cvmParticleAngle;	cvgmParticleAngle_.download(cvmParticleAngle);
	cv::Mat cvmParticleResponses;	cvgmParticleResponses_.download(cvmParticleResponses);
	cv::Mat cvmParticleDescriptor;	cvgmParticleDescriptor_.download(cvmParticleDescriptor);
	bool bLegal = true;
	int nCount = 0; int nIllegal = 0;
	for (int r=0;r<cvmParticleResponses.rows; r++)	{
		for (int c=0; c<cvmParticleResponses.cols; c++) {
			if( cvmParticleResponses.ptr<float>(r)[c] > 0.1f ){
				nCount ++;
				bLegal = bLegal && (	cvmParticleDescriptor.ptr<int2>(r)[c].x != 0 && 
										cvmParticleDescriptor.ptr<int2>(r)[c].y != 0 &&
										cvmParticleResponses.ptr<float>(r)[c] > 0.f /*&&
										cvmParticleAngle.ptr<float>(r)[c] != 0.f */);

				if (!bLegal){
					nIllegal ++;
				}
			}
		}//for
	}//for
	
	*pnCounter_ = nCount;
	return bLegal;
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

	cv::Mat_<uchar>  _cvmImageCurr;
	cv::Mat_<float>  _cvgmSaliencyCurr;
	cv::Mat_<int2>   _cvmParticleDescriptorCurr;
	cv::Mat_<int2>   _cvmParticleDescriptorCurrTmp;
	cv::Mat_<float>  _cvmParticleResponseCurr;
	cv::Mat_<uchar>  _cvmParticlesAgeCurr;
	cv::Mat_<short2> _cvmParticlesVelocityCurr;

	cv::Mat_<uchar>  _cvmMinMatchDistance;

	float _fRho;

	unsigned short _usMatchThreshold;
	unsigned short _usHalfSize;
	unsigned short _usHalfSizeRound;
	short _sSearchRange;
	const short* _psPatternX;
	const short* _psPatternY;

	unsigned int _devuMathchedCounter;
	unsigned int _devuDeletedCounter;
	unsigned int _devuOther;
	unsigned int _devuTotal;
	unsigned int _devuTest1;

	unsigned int _uMaxMatchedKeyPoint;
	short2* _pcvgmMatchedKeyPointLocation;

	__device__ __forceinline__ uchar dL(const uchar* pDesPrev_, const uchar* pDesCurr_) const{
		uchar ucRes = 0;
		for(short s = 0; s<8; s++)
			ucRes += _popCountTable[ pDesPrev_[s] ^ pDesCurr_[s] ];
		return ucRes;
	}

__device__ __forceinline__ uchar devMatchOrb( const unsigned short usMatchThreshold_, 
	const uchar* pDesPrev_, const short2 s2PredicLoc_, short2* ps2BestLoc_, uchar* pBestDesCur_ ){
		float fResponse = 0.f;
		short2 s2Loc;
		uchar ucMinDist = 255;
		//search for the 7x7 neighbourhood for 
		for(short r = -_sSearchRange; r <= _sSearchRange; r++ ){
			for(short c = -_sSearchRange; c <= _sSearchRange; c++ ){

				s2Loc = s2PredicLoc_ + make_short2( c, r ); 
				if(s2Loc.x < _usHalfSizeRound || s2Loc.x >= _cvmImageCurr.cols - _usHalfSizeRound || s2Loc.y < _usHalfSizeRound || s2Loc.y >= _cvmImageCurr.rows - _usHalfSizeRound ) continue;
				fResponse = _cvgmSaliencyCurr.ptr<float>(s2Loc.y)[s2Loc.x];
				if( fResponse > 0.1f ){
					//assert(_cvmParticleDescriptorCurrTmp.ptr<int2>(s2Loc.y)[s2Loc.x].x!=0 &&_cvmParticleDescriptorCurrTmp.ptr<int2>(s2Loc.y)[s2Loc.x].y!=0 );
					const uchar* pDesCur = (uchar*)(_cvmParticleDescriptorCurrTmp.ptr<int2>(s2Loc.y)+ s2Loc.x);
					uchar ucDist = dL(pDesPrev_,pDesCur);
					if ( ucDist < usMatchThreshold_ ){
						if (  ucMinDist > ucDist ){
							ucMinDist = ucDist;
							*ps2BestLoc_ = s2Loc;
							pBestDesCur_[0] = pDesCur[0];pBestDesCur_[1] = pDesCur[1];pBestDesCur_[2] = pDesCur[2];pBestDesCur_[3] = pDesCur[3];pBestDesCur_[4] = pDesCur[4];pBestDesCur_[5] = pDesCur[5];pBestDesCur_[6] = pDesCur[6];pBestDesCur_[7] = pDesCur[7];
						}
					}
				}//if sailent corner exits
			}//for 
		}//for
		return ucMinDist;
}//devMatchOrb()

__device__ __forceinline__ void operator () (){
	short2 s2BestLoc;
	for (int r=0; r<_cvmImageCurr.rows; r++ ){
		for (int c=0; c<_cvmImageCurr.cols; c++ ){
			if( c < 3 || c >= _cvmImageCurr.cols - 4 || r < 3 || r >= _cvmImageCurr.rows - 4 ) continue;

			//if IsParticle( PixelLocation, cvgmParitclesResponse(i) )
			if(_cvmParticleResponsesPrev.ptr<float>(r)[c] < 0.2f) continue;
			const uchar* pDesPrev = (uchar*) ( _cvmParticleOrbDescriptorsPrev.ptr<int2>(r)+c);
			uchar aDesBest[8];
			
			const uchar ucDist = devMatchOrb( _usMatchThreshold, pDesPrev, make_short2(c,r), &s2BestLoc, &*aDesBest );
			if( ucDist < 64 ){
				_devuOther++;
				const uchar& ucMin = _cvmMinMatchDistance.ptr(s2BestLoc.y)[s2BestLoc.x];
				if(ucMin == 0xff){//if no matches before
					_devuTest1++;
					_devuMathchedCounter++;
					_cvmParticleDescriptorCurr.ptr<int2>(s2BestLoc.y)[s2BestLoc.x] = *((int2*)aDesBest);
					_cvmParticlesVelocityCurr.ptr<short2>(s2BestLoc.y)[s2BestLoc.x] = _fRho * (s2BestLoc - make_short2(c,r)) + (1.f - _fRho)* _cvmParticlesVelocityPrev.ptr<short2>(r)[c];//update velocity
					_cvmParticlesAgeCurr.ptr             (s2BestLoc.y)[s2BestLoc.x] = _cvmParticlesAgePrev.ptr(r)[c] + 1; //update age
					_cvmParticleResponseCurr.ptr<float> (s2BestLoc.y)[s2BestLoc.x] =  _cvgmSaliencyCurr.ptr<float>(s2BestLoc.y)[s2BestLoc.x]; //update response and location //marked as matched and it will be corrected in NoMaxAndCollection
					_cvmMinMatchDistance     .ptr(s2BestLoc.y)[s2BestLoc.x] = ucDist;
				}
				else{
					_devuDeletedCounter++;
					if (ucMin > ucDist){
						_cvmParticleDescriptorCurr.ptr<int2>(s2BestLoc.y)[s2BestLoc.x] = *((int2*)aDesBest);
						_cvmParticlesVelocityCurr.ptr<short2>(s2BestLoc.y)[s2BestLoc.x] = _fRho * (s2BestLoc - make_short2(c,r)) + (1.f - _fRho)* _cvmParticlesVelocityPrev.ptr<short2>(r)[c];//update velocity
						_cvmParticlesAgeCurr.ptr             (s2BestLoc.y)[s2BestLoc.x] = _cvmParticlesAgePrev.ptr(r)[c] + 1; //update age
						_cvmParticleResponseCurr.ptr<float> (s2BestLoc.y)[s2BestLoc.x] = _cvgmSaliencyCurr.ptr<float>(s2BestLoc.y)[s2BestLoc.x]; //update response and location //marked as matched and it will be corrected in NoMaxAndCollection
						_cvmMinMatchDistance     .ptr(s2BestLoc.y)[s2BestLoc.x] = ucDist;
					}//if
				}//else
			}
			else{//C) if no match found 
				_devuDeletedCounter++;
			}//lost
		}//for
	}//for
	return;
}
};//class

unsigned int testCudaTrackOrb(const unsigned short usMatchThreshold_, const unsigned short usHalfSize_,const unsigned short sSearchRange_,
							  const short* psPatternX_, const short* psPatternY_, const unsigned int uMaxMatchedKeyPoints_,
							  const cv::gpu::GpuMat& cvgmParticleOrbDescriptorsPrev_, const cv::gpu::GpuMat& cvgmParticleResponsesPrev_, 
							  const cv::gpu::GpuMat& cvgmParticlesAgePrev_,const cv::gpu::GpuMat& cvgmParticlesVelocityPrev_, 
							  const cv::gpu::GpuMat& cvgmImage_, const cv::gpu::GpuMat& cvgmParticleOrbDescriptorsCurrTmp_,
							  const cv::gpu::GpuMat& cvgmSaliencyCurr_,
							  cv::gpu::GpuMat* pcvgmMinMatchDistance_,
							  cv::gpu::GpuMat* pcvgmParticleResponsesCurr_,
							  cv::gpu::GpuMat* pcvgmParticlesAgeCurr_,cv::gpu::GpuMat* pcvgmParticleVelocityCurr_,cv::gpu::GpuMat* pcvgmParticleOrbDescriptorsCurr_){

	CPredictAndMatchOrb cPAMO;
	cvgmImage_.download(cPAMO._cvmImageCurr);
	cvgmParticleResponsesPrev_.download(cPAMO._cvmParticleResponsesPrev);
	cvgmParticleOrbDescriptorsPrev_.download(cPAMO._cvmParticleOrbDescriptorsPrev);
	cvgmParticlesVelocityPrev_.download(cPAMO._cvmParticlesVelocityPrev); 
	cvgmParticlesAgePrev_.download(cPAMO._cvmParticlesAgePrev); 

	cvgmSaliencyCurr_.download(cPAMO._cvgmSaliencyCurr);
	cvgmParticleOrbDescriptorsCurrTmp_.download(cPAMO._cvmParticleDescriptorCurrTmp);
	pcvgmParticleOrbDescriptorsCurr_->download(cPAMO._cvmParticleDescriptorCurr);
	pcvgmParticleResponsesCurr_->download(cPAMO._cvmParticleResponseCurr);
	pcvgmParticleVelocityCurr_->download(cPAMO._cvmParticlesVelocityCurr);
	pcvgmParticlesAgeCurr_->download(cPAMO._cvmParticlesAgeCurr);
	pcvgmMinMatchDistance_->setTo(255);
	pcvgmMinMatchDistance_->download(cPAMO._cvmMinMatchDistance);

	cPAMO._fRho = .75f;
	cPAMO._usMatchThreshold = usMatchThreshold_;
	cPAMO._usHalfSize = usHalfSize_;
	cPAMO._usHalfSizeRound = (unsigned short)(usHalfSize_*1.5);
	cPAMO._sSearchRange = sSearchRange_;
	cPAMO._psPatternX = psPatternX_;
	cPAMO._psPatternY = psPatternY_;

	cPAMO._uMaxMatchedKeyPoint = uMaxMatchedKeyPoints_;
	cPAMO._devuDeletedCounter = 0;
	cPAMO._devuMathchedCounter = 0;
	cPAMO._devuOther = 0;
	cPAMO._devuTest1 = 0;
	cPAMO();

	pcvgmParticleOrbDescriptorsCurr_->upload(cPAMO._cvmParticleDescriptorCurr);
	pcvgmParticleResponsesCurr_->upload(cPAMO._cvmParticleResponseCurr);
	pcvgmParticleVelocityCurr_->upload(cPAMO._cvmParticlesVelocityCurr);
	pcvgmParticlesAgeCurr_->upload(cPAMO._cvmParticlesAgeCurr);

	cPAMO._devuOther;
	cPAMO._devuDeletedCounter;
	return cPAMO._devuMathchedCounter;
}

struct SCollectUnMatchedKeyPoints{
	cv::Mat_<float> _cvmSaliency;
	unsigned int _uNewlyAddedCount;

	unsigned int _uTotal;
	short2* _ps2NewlyAddedKeyPointLocation; 
	float* _pfNewlyAddedKeyPointResponse;

	int _devuCounter;

	cv::Mat_<float> _cvmParticleResponseCurr;
	cv::Mat_<int2>  _cvmParticleDescriptorCurr;

	__device__ __forceinline__ void operator () (){

		for (int c=0; c<_cvmParticleResponseCurr.cols; c++){
			for (int r=0; r<_cvmParticleResponseCurr.rows; r++){
				if( c < 3 || c >= _cvmParticleResponseCurr.cols - 4 || r < 3 || r >= _cvmParticleResponseCurr.rows - 4 ) continue;
				if(_cvmSaliency.ptr<float>(r)[c] == _cvmParticleResponseCurr.ptr<float>(r)[c]) continue; // it is a matched key points
				if(_cvmSaliency.ptr<float>(r)[c] > 0.f ){
					const unsigned int nIdx = _devuCounter++;
					if (nIdx >= _uTotal) continue;
					_ps2NewlyAddedKeyPointLocation[nIdx] = make_short2(c,r);
					_pfNewlyAddedKeyPointResponse[nIdx] = _cvmSaliency.ptr<float>(r)[c];
				}
			}
		}

		return;
	}//operator()
};

__global__ void kernerlAddNewParticles( const unsigned int uTotalParticles_,   
	const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, 
	const cv::Mat_<int2> cvmParticleDescriptorTmp_,
	cv::Mat_<float>& cvmParticleResponse_, cv::Mat_<int2>& cvmParticleDescriptor_){

		for (int nKeyPointIdx =0; nKeyPointIdx < uTotalParticles_; nKeyPointIdx++){
			const short2& s2Loc = ps2KeyPointsLocations_[nKeyPointIdx];
			cvmParticleResponse_.ptr<float>(s2Loc.y)[s2Loc.x] = pfKeyPointsResponse_[nKeyPointIdx];
			cvmParticleDescriptor_.ptr<int2>(s2Loc.y)[s2Loc.x] = cvmParticleDescriptorTmp_.ptr<int2>(s2Loc.y)[s2Loc.x]; 
		}
		return;
}

namespace btl{ namespace device{ namespace semidense{
void thrustSort(short2* pnLoc_, float* pfResponse_, const unsigned int nCorners_);
}}}
void testCudaCollectNewlyAddedKeyPoints(unsigned int uNewlyAdded_, unsigned int uMaxKeyPointsAfterNonMax_, 
	const cv::gpu::GpuMat& cvgmSaliency_,const cv::gpu::GpuMat& cvgmParticleResponseCurr_, const cv::gpu::GpuMat& cvgmParticleDescriptorCurrTmp_,  
	cv::gpu::GpuMat* pcvgmNewlyAddedKeyPointLocation_, cv::gpu::GpuMat* pcvgmNewlyAddedKeyPointResponse_,
	cv::gpu::GpuMat* pcvgmParticleResponseCurr_, cv::gpu::GpuMat* pcvgmParticleDescriptorCurr_){
	
		if(!uNewlyAdded_) return;
		SCollectUnMatchedKeyPoints sCUMKP;
		cvgmSaliency_.download(sCUMKP._cvmSaliency);
		sCUMKP._uNewlyAddedCount = uNewlyAdded_;

		sCUMKP._uTotal = uMaxKeyPointsAfterNonMax_;
		cv::Mat cvmNewlyAddedKeyPointLocation;
		pcvgmNewlyAddedKeyPointLocation_->download(cvmNewlyAddedKeyPointLocation);
		sCUMKP._ps2NewlyAddedKeyPointLocation = cvmNewlyAddedKeyPointLocation.ptr<short2>(); 
		cv::Mat cvmNewlyAddedKeyPointResponse;
		pcvgmNewlyAddedKeyPointResponse_->download(cvmNewlyAddedKeyPointResponse);
		sCUMKP._pfNewlyAddedKeyPointResponse = cvmNewlyAddedKeyPointResponse.ptr<float>();

		pcvgmParticleResponseCurr_->download(sCUMKP._cvmParticleResponseCurr);
		pcvgmParticleDescriptorCurr_->download(sCUMKP._cvmParticleDescriptorCurr);

		sCUMKP._devuCounter = 0;
		sCUMKP();

		unsigned int uUnMatched = sCUMKP._devuCounter; 
		//sort 
		pcvgmNewlyAddedKeyPointLocation_->upload(cvmNewlyAddedKeyPointLocation);
		pcvgmNewlyAddedKeyPointResponse_->upload(cvmNewlyAddedKeyPointResponse);
		btl::device::semidense::thrustSort(pcvgmNewlyAddedKeyPointLocation_->ptr<short2>(),pcvgmNewlyAddedKeyPointResponse_->ptr<float>(),uUnMatched);

		pcvgmNewlyAddedKeyPointLocation_->download(cvmNewlyAddedKeyPointLocation);
		pcvgmNewlyAddedKeyPointResponse_->download(cvmNewlyAddedKeyPointResponse);

		cv::Mat_<int2> cvmParticleDescriptorCurrTmp;
		cv::Mat_<float> cvmParticleResponseCurr;
		cv::Mat_<int2> cvmParticleDescriptorCurr;
		cvgmParticleDescriptorCurrTmp_.download(cvmParticleDescriptorCurrTmp);
		pcvgmParticleResponseCurr_->download(cvmParticleResponseCurr);
		pcvgmParticleDescriptorCurr_->download(cvmParticleDescriptorCurr);
		kernerlAddNewParticles(uNewlyAdded_<uUnMatched?uNewlyAdded_:uUnMatched,cvmNewlyAddedKeyPointLocation.ptr<short2>(),cvmNewlyAddedKeyPointResponse.ptr<float>(),
			cvmParticleDescriptorCurrTmp,cvmParticleResponseCurr,cvmParticleDescriptorCurr);
		pcvgmParticleResponseCurr_->upload(cvmParticleResponseCurr);
		pcvgmParticleDescriptorCurr_->upload(cvmParticleDescriptorCurr);

		return;
}