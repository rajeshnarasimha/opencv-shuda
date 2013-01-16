#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

#include "TestCudaOrb.h"

__device__ short2 operator + (const short2 s2O1_, const short2 s2O2_);
__device__ short2 operator - (const short2 s2O1_, const short2 s2O2_);
__device__ float2 operator * (const float fO1_, const short2 s2O2_);
__device__ __host__ float2 operator + (const float2 f2O1_, const float2 f2O2_);
__device__ __host__ float2 operator - (const float2 f2O1_, const float2 f2O2_);
__device__  short2 convert2s2(const float2 f2O1_);

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

bool testCountMinDistAndMatchedLocation(const cv::gpu::GpuMat cvgmMinMatchDistance_, const cv::gpu::GpuMat& cvgmMatchedLocationPrev_, int* pnCounter_){
	cv::Mat cvmMatchedLocationPrev;	cvgmMatchedLocationPrev_.download(cvmMatchedLocationPrev);
	cv::Mat cvmMinMatchDistance;	cvgmMinMatchDistance_.download(cvmMinMatchDistance);
	bool bLegal = true;
	int nCount = 0; int nIllegal = 0;
	for (int r=0;r<cvmMinMatchDistance.rows; r++)	{
		for (int c=0; c<cvmMinMatchDistance.cols; c++) {
			if( cvmMinMatchDistance.ptr(r)[c] < 255 ){
				nCount ++;
				bLegal = bLegal && ( cvmMatchedLocationPrev.ptr<short2>(r)[c].x != 0 && 
									 cvmMatchedLocationPrev.ptr<short2>(r)[c].y != 0 );

				if (!bLegal){
					nIllegal ++;
				}
			}
		}//for
	}//for
	
	*pnCounter_ = nCount;
	return bLegal;
}

bool testCountResponseAndDescriptor(const cv::gpu::GpuMat cvgmParticleResponse_, const cv::gpu::GpuMat& cvgmParticleDescriptor_, int* pnCounter_){
	cv::Mat cvmParticleDescriptor;	cvgmParticleDescriptor_.download(cvmParticleDescriptor);
	cv::Mat cvmParticleResponse;	cvgmParticleResponse_.download(cvmParticleResponse);
	bool bLegal = true;
	int nCount = 0; int nIllegal = 0;
	for (int r=0;r<cvmParticleResponse.rows; r++)	{
		for (int c=0; c<cvmParticleResponse.cols; c++) {
			if( cvmParticleResponse.ptr<float>(r)[c] > 0.1f ){
				nCount ++;
				bLegal = bLegal && ( cvmParticleDescriptor.ptr<int2>(r)[c].x != 0 && 
					cvmParticleDescriptor.ptr<int2>(r)[c].y != 0 );

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
	//cv::Mat_<uchar>  _cvmParticlesAgePrev;
	//cv::Mat_<short2> _cvmParticlesVelocityPrev;

	//cv::Mat_<uchar>  _cvmImageCurr;
	cv::Mat_<float>  _cvmSaliencyCurr;
	//cv::Mat_<int2>   _cvmParticleDescriptorCurr;
	cv::Mat_<int2>   _cvmParticleDescriptorCurrTmp;
	//cv::Mat_<float>  _cvmParticleResponseCurr;
	//cv::Mat_<uchar>  _cvmParticlesAgeCurr;
	//cv::Mat_<short2> _cvmParticlesVelocityCurr;

	cv::Mat_<uchar>  _cvmMinMatchDistance;
	cv::Mat_<short2> _cvmMatchedLocationPrev;

	//float _fRho;

	short _sSearchRange;
	unsigned short _usMatchThreshold;
	unsigned short _usHalfSize;
	unsigned short _usHalfSizeRound;

	const short* _psPatternX;
	const short* _psPatternY;

	unsigned int _devuMathchedCounter;
	unsigned int _devuDeletedCounter;
	unsigned int _devuOther;
	unsigned int _devuTotal;
	unsigned int _devuTest1;

	//unsigned int _uMaxMatchedKeyPoint;
	//short2* _pcvgmMatchedKeyPointLocation;

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
				if(s2Loc.x < _usHalfSizeRound || s2Loc.x >= _cvmParticleResponsesPrev.cols - _usHalfSizeRound || s2Loc.y < _usHalfSizeRound || s2Loc.y >= _cvmParticleResponsesPrev.rows - _usHalfSizeRound ) continue;
				fResponse = _cvmSaliencyCurr.ptr<float>(s2Loc.y)[s2Loc.x];
				if( fResponse > 0.1f ){
					//assert(_cvmParticleDescriptorCurrTmp.ptr<int2>(s2Loc.y)[s2Loc.x].x!=0 &&_cvmParticleDescriptorCurrTmp.ptr<int2>(s2Loc.y)[s2Loc.x].y!=0 );
					const uchar* pDesCur = (uchar*)(_cvmParticleDescriptorCurrTmp.ptr<int2>(s2Loc.y)+ s2Loc.x);
					uchar ucDist = dL(pDesPrev_,pDesCur);
					if ( ucDist < usMatchThreshold_ ){
						if (  ucMinDist > ucDist ){
							ucMinDist = ucDist;
							*ps2BestLoc_ = s2Loc;
							//pBestDesCur_[0] = pDesCur[0];pBestDesCur_[1] = pDesCur[1];pBestDesCur_[2] = pDesCur[2];pBestDesCur_[3] = pDesCur[3];pBestDesCur_[4] = pDesCur[4];pBestDesCur_[5] = pDesCur[5];pBestDesCur_[6] = pDesCur[6];pBestDesCur_[7] = pDesCur[7];
						}
					}
				}//if sailent corner exits
			}//for 
		}//for
		return ucMinDist;
}//devMatchOrb()

__device__ __forceinline__ void operator () (){
	short2 s2BestLoc;
	for (int r=0; r<_cvmParticleResponsesPrev.rows; r++ ){
		for (int c=0; c<_cvmParticleResponsesPrev.cols; c++ ){
			if( c < 3 || c >= _cvmParticleResponsesPrev.cols - 4 || r < 3 || r >= _cvmParticleResponsesPrev.rows - 4 ) continue;

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
					//_cvmParticleDescriptorCurr.ptr<int2>(s2BestLoc.y)[s2BestLoc.x] = *((int2*)aDesBest);
					//_cvmParticlesVelocityCurr.ptr<short2>(s2BestLoc.y)[s2BestLoc.x] = _fRho * (s2BestLoc - make_short2(c,r)) + (1.f - _fRho)* _cvmParticlesVelocityPrev.ptr<short2>(r)[c];//update velocity
					//_cvmParticlesAgeCurr.ptr             (s2BestLoc.y)[s2BestLoc.x] = _cvmParticlesAgePrev.ptr(r)[c] + 1; //update age
					//_cvmParticleResponseCurr.ptr<float> (s2BestLoc.y)[s2BestLoc.x] =  _cvmSaliencyCurr.ptr<float>(s2BestLoc.y)[s2BestLoc.x]; //update response and location //marked as matched and it will be corrected in NoMaxAndCollection
					_cvmMinMatchDistance     .ptr(s2BestLoc.y)[s2BestLoc.x] = ucDist;
					_cvmMatchedLocationPrev  .ptr<short2>(s2BestLoc.y)[s2BestLoc.x] = make_short2(c,r);
				}
				else{
					_devuDeletedCounter++;
					if (ucMin > ucDist){
						//_cvmParticleDescriptorCurr.ptr<int2>(s2BestLoc.y)[s2BestLoc.x] = *((int2*)aDesBest);
						//_cvmParticlesVelocityCurr.ptr<short2>(s2BestLoc.y)[s2BestLoc.x] = _fRho * (s2BestLoc - make_short2(c,r)) + (1.f - _fRho)* _cvmParticlesVelocityPrev.ptr<short2>(r)[c];//update velocity
						//_cvmParticlesAgeCurr.ptr             (s2BestLoc.y)[s2BestLoc.x] = _cvmParticlesAgePrev.ptr(r)[c] + 1; //update age
						//_cvmParticleResponseCurr.ptr<float> (s2BestLoc.y)[s2BestLoc.x] = _cvmSaliencyCurr.ptr<float>(s2BestLoc.y)[s2BestLoc.x]; //update response and location //marked as matched and it will be corrected in NoMaxAndCollection
						_cvmMinMatchDistance     .ptr(s2BestLoc.y)[s2BestLoc.x] = ucDist;
						_cvmMatchedLocationPrev  .ptr<short2>(s2BestLoc.y)[s2BestLoc.x] = make_short2(c,r);
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

unsigned int testCudaTrackOrb(const unsigned short usMatchThreshold_, const unsigned short usHalfSize_, const short sSearchRange_,
							const short* psPatternX_, const short* psPatternY_, /*const unsigned int uMaxMatchedKeyPoints_,*/
							const cv::gpu::GpuMat& cvgmParticleOrbDescriptorsPrev_, const cv::gpu::GpuMat& cvgmParticleResponsesPrev_, 
							/*const cv::gpu::GpuMat& cvgmParticlesAgePrev_,const cv::gpu::GpuMat& cvgmParticlesVelocityPrev_, 
							const cv::gpu::GpuMat& cvgmImage_,*/ const cv::gpu::GpuMat& cvgmParticleDescriptorCurrTmp_,
							const cv::gpu::GpuMat& cvgmSaliencyCurr_,
							/*cv::gpu::GpuMat* pcvgmMutex_,*/
							cv::gpu::GpuMat* pcvgmMinMatchDistance_,
							cv::gpu::GpuMat* pcvgmMatchedLocationPrev_
							/*cv::gpu::GpuMat* pcvgmParticleResponsesCurr_,
							cv::gpu::GpuMat* pcvgmParticlesAgeCurr_,cv::gpu::GpuMat* pcvgmParticlesVelocityCurr_,cv::gpu::GpuMat* pcvgmParticleOrbDescriptorsCurr_*/){

	CPredictAndMatchOrb cPAMO;
	//cvgmImage_.download(cPAMO._cvmImageCurr);
	cvgmParticleResponsesPrev_.download(cPAMO._cvmParticleResponsesPrev);
	cvgmParticleOrbDescriptorsPrev_.download(cPAMO._cvmParticleOrbDescriptorsPrev);
	//cvgmParticlesVelocityPrev_.download(cPAMO._cvmParticlesVelocityPrev); 
	//cvgmParticlesAgePrev_.download(cPAMO._cvmParticlesAgePrev); 

	cvgmSaliencyCurr_.download(cPAMO._cvmSaliencyCurr);
	cvgmParticleDescriptorCurrTmp_.download(cPAMO._cvmParticleDescriptorCurrTmp);
	//pcvgmParticleOrbDescriptorsCurr_->download(cPAMO._cvmParticleDescriptorCurr);
	//pcvgmParticleResponsesCurr_->download(cPAMO._cvmParticleResponseCurr);
	//pcvgmParticleVelocityCurr_->download(cPAMO._cvmParticlesVelocityCurr);
	//pcvgmParticlesAgeCurr_->download(cPAMO._cvmParticlesAgeCurr);
	pcvgmMinMatchDistance_->setTo(255);
	pcvgmMinMatchDistance_->download(cPAMO._cvmMinMatchDistance);
	pcvgmMatchedLocationPrev_->setTo(cv::Scalar::all(0));
	pcvgmMatchedLocationPrev_->download(cPAMO._cvmMatchedLocationPrev);

	//cPAMO._fRho = .75f;
	cPAMO._usMatchThreshold = usMatchThreshold_;
	cPAMO._usHalfSize = usHalfSize_;
	cPAMO._usHalfSizeRound = (unsigned short)(usHalfSize_*1.5);
	cPAMO._sSearchRange = sSearchRange_;
	cPAMO._psPatternX = psPatternX_;
	cPAMO._psPatternY = psPatternY_;

	//cPAMO._uMaxMatchedKeyPoint = uMaxMatchedKeyPoints_;
	cPAMO._devuDeletedCounter = 0;
	cPAMO._devuMathchedCounter = 0;
	cPAMO._devuOther = 0;
	cPAMO._devuTest1 = 0;
	cPAMO();

	pcvgmMinMatchDistance_->upload(cPAMO._cvmMinMatchDistance);
	pcvgmMatchedLocationPrev_->upload(cPAMO._cvmMatchedLocationPrev);
	//pcvgmParticleOrbDescriptorsCurr_->upload(cPAMO._cvmParticleDescriptorCurr);
	//pcvgmParticleResponsesCurr_->upload(cPAMO._cvmParticleResponseCurr);
	//pcvgmParticleVelocityCurr_->upload(cPAMO._cvmParticlesVelocityCurr);
	//pcvgmParticlesAgeCurr_->upload(cPAMO._cvmParticlesAgeCurr);

	cPAMO._devuOther;
	cPAMO._devuDeletedCounter;
	return cPAMO._devuMathchedCounter;
}

struct SCollectUnMatchedKeyPoints{

	int _devuCounter;
	int _devuOther;

	cv::Mat_<float> _cvmSaliency;
	cv::Mat_<int2>  _cvmParticleDescriptorCurrTmp;

	cv::Mat_<short2> _cvmParticleVelocityPrev;
	cv::Mat_<uchar>  _cvmParticleAgePrev;
	cv::Mat_<short2> _cvmParticleVelocityCurr;
	cv::Mat_<uchar>  _cvmParticleAgeCurr;
	cv::Mat_<float>  _cvmParticleResponseCurr;
	cv::Mat_<int2>   _cvmParticleDescriptorCurr;

	cv::Mat_<short2> _cvmMatchedLocationPrev;
	cv::Mat_<uchar>  _cvmMinMatchDistance;

	unsigned int _uMaxMatchedKeyPoint;
	unsigned int _uMaxNewKeyPoint;
	float _fRho;

	short2* _ps2NewlyAddedKeyPointLocation; 
	float*  _pfNewlyAddedKeyPointResponse;

	short2* _ps2MatchedKeyPointLocation;
	float*  _pfMatchedKeyPointResponse;


	__device__ __forceinline__ void operator () (){

		for (int c=0; c<_cvmParticleResponseCurr.cols; c++){
			for (int r=0; r<_cvmParticleResponseCurr.rows; r++){
				if( c < 0 || c >= _cvmParticleResponseCurr.cols || r < 0 || r >= _cvmParticleResponseCurr.rows ) continue;
				
				_cvmParticleVelocityCurr  .ptr<short2>(r)[c] = make_short2(0,0);
				_cvmParticleAgeCurr		  .ptr<uchar>(r)[c] = 0;
				_cvmParticleResponseCurr  .ptr<float>(r)[c] = 0.f;
				_cvmParticleDescriptorCurr.ptr<int2>(r)[c] = make_int2(0,0);

				const float& fResponse = _cvmSaliency.ptr<float>(r)[c];
				if( fResponse < 0.1f ) continue;

				if(_cvmMinMatchDistance.ptr<uchar>(r)[c] == 255 ){
					const unsigned int nIdx = ++_devuCounter;
					if (nIdx >= _uMaxNewKeyPoint) continue;
					_ps2NewlyAddedKeyPointLocation[nIdx] = make_short2(c,r);
					_pfNewlyAddedKeyPointResponse[nIdx]  = fResponse;
				}
				else{
					const short2& s2PrevLoc = _cvmMatchedLocationPrev.ptr<short2>(r)[c];
					
					const unsigned int nIdx = ++_devuOther;//count Matched
					if( nIdx >= _uMaxMatchedKeyPoint) continue;
					_ps2MatchedKeyPointLocation[nIdx] = make_short2(c,r);
					_pfMatchedKeyPointResponse[nIdx]  = fResponse;

					_cvmParticleResponseCurr  .ptr<float>(r)[c] = fResponse; 
					_cvmParticleDescriptorCurr.ptr<int2>(r)[c] = _cvmParticleDescriptorCurrTmp.ptr<int2>(r)[c];
					_cvmParticleVelocityCurr  .ptr<short2>(r)[c] = make_short2(c,r) - s2PrevLoc;
						//convert2s2( _fRho * (make_short2(c,r) - s2PrevLoc) + (1.f - _fRho)* _cvmParticleVelocityPrev.ptr<short2>(s2PrevLoc.y)[s2PrevLoc.x] + make_float2(0.5,0.5));//update velocity
					_cvmParticleAgeCurr	      .ptr<uchar>(r)[c] = _cvmParticleAgePrev.ptr<uchar>(s2PrevLoc.y)[s2PrevLoc.x] + 1; //update age
				}
			}
		}

		return;
	}//operator()
};

__global__ void kernelAddNewParticles( const unsigned int uTotalParticles_,   
	const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, 
	const cv::Mat_<int2> cvmParticleDescriptorTmp_,
	cv::Mat_<float>& cvmParticleResponse_, cv::Mat_<int2>& cvmParticleDescriptor_){

		for (unsigned int nKeyPointIdx =0; nKeyPointIdx < uTotalParticles_; nKeyPointIdx++){
			const short2& s2Loc = ps2KeyPointsLocations_[nKeyPointIdx];
			cvmParticleResponse_.ptr<float>(s2Loc.y)[s2Loc.x] = pfKeyPointsResponse_[nKeyPointIdx];
			cvmParticleDescriptor_.ptr<int2>(s2Loc.y)[s2Loc.x] = cvmParticleDescriptorTmp_.ptr<int2>(s2Loc.y)[s2Loc.x]; 
		}
		return;
}

namespace btl{ namespace device{ namespace semidense{
void thrustSort(short2* pnLoc_, float* pfResponse_, const unsigned int nCorners_);
}}}
void testCudaCollectNewlyAddedKeyPoints(unsigned int uTotalParticles_, unsigned int uMaxNewKeyPoints_, const float fRho_,
										const cv::gpu::GpuMat& cvgmSaliency_,/*const cv::gpu::GpuMat& cvgmParticleResponseCurrTmp_,*/
										const cv::gpu::GpuMat& cvgmParticleDescriptorCurrTmp_,
										const cv::gpu::GpuMat& cvgmParticleVelocityPrev_,
										const cv::gpu::GpuMat& cvgmParticleAgePrev_,
										const cv::gpu::GpuMat& cvgmMinMatchDistance_,
										const cv::gpu::GpuMat& cvgmMatchedLocationPrev_,
										cv::gpu::GpuMat* pcvgmNewlyAddedKeyPointLocation_, cv::gpu::GpuMat* pcvgmNewlyAddedKeyPointResponse_,
										cv::gpu::GpuMat* pcvgmMatchedKeyPointLocation_, cv::gpu::GpuMat* pcvgmMatchedKeyPointResponse_,
										cv::gpu::GpuMat* pcvgmParticleResponseCurr_, cv::gpu::GpuMat* pcvgmParticleDescriptorCurr_,
										cv::gpu::GpuMat* pcvgmParticleVelocityCurr_, cv::gpu::GpuMat* pcvgmParticleAgeCurr_){
	
		if(!uTotalParticles_) return;

		SCollectUnMatchedKeyPoints sCUMKP;
		int nCounter = 0;
		bool bIsLegal = testCountResponseAndDescriptor(cvgmSaliency_,cvgmParticleDescriptorCurrTmp_,&nCounter);

		cvgmSaliency_.download(sCUMKP._cvmSaliency);
		cvgmParticleDescriptorCurrTmp_.download(sCUMKP._cvmParticleDescriptorCurrTmp);

		cvgmParticleVelocityPrev_.download(sCUMKP._cvmParticleVelocityPrev);
		cvgmParticleAgePrev_.download(sCUMKP._cvmParticleAgePrev);

		cvgmMinMatchDistance_.download(sCUMKP._cvmMinMatchDistance);
		cvgmMatchedLocationPrev_.download(sCUMKP._cvmMatchedLocationPrev);

		pcvgmParticleResponseCurr_->download(sCUMKP._cvmParticleResponseCurr);
		pcvgmParticleDescriptorCurr_->download(sCUMKP._cvmParticleDescriptorCurr);
		pcvgmParticleVelocityCurr_->download(sCUMKP._cvmParticleVelocityCurr);
		pcvgmParticleAgeCurr_->download(sCUMKP._cvmParticleAgeCurr);

		sCUMKP._uMaxMatchedKeyPoint = uTotalParticles_;
		sCUMKP._uMaxNewKeyPoint     = uMaxNewKeyPoints_; //the size of the newly added keypoint
		sCUMKP._fRho                = fRho_;
		cv::Mat cvmTmpNewlyAddedKeyPointLocation;	pcvgmNewlyAddedKeyPointLocation_->download(cvmTmpNewlyAddedKeyPointLocation);
		cv::Mat cvmTmpNewlyAddedKeyPointResponse;   pcvgmNewlyAddedKeyPointResponse_->download(cvmTmpNewlyAddedKeyPointResponse);
		cv::Mat cvmTmpMatchedKeyPointLocation;      pcvgmMatchedKeyPointLocation_->download(cvmTmpMatchedKeyPointLocation);
		cv::Mat cvmTmpMatchedKeyPointResponse;      pcvgmMatchedKeyPointResponse_->download(cvmTmpMatchedKeyPointResponse);
		sCUMKP._ps2NewlyAddedKeyPointLocation = cvmTmpNewlyAddedKeyPointLocation.ptr<short2>(); 
		sCUMKP._pfNewlyAddedKeyPointResponse  = cvmTmpNewlyAddedKeyPointResponse.ptr<float>();
		sCUMKP._ps2MatchedKeyPointLocation    = cvmTmpMatchedKeyPointLocation.ptr<short2>(); 
		sCUMKP._pfMatchedKeyPointResponse     = cvmTmpMatchedKeyPointResponse.ptr<float>();
		
		sCUMKP._devuCounter = 0;
		sCUMKP._devuOther = 0; 

		

		sCUMKP();
		nCounter = 0;
		bIsLegal = testCountResponseAndDescriptor(*pcvgmParticleResponseCurr_, *pcvgmParticleDescriptorCurr_, &nCounter);

		unsigned int uNew     = sCUMKP._devuCounter;
		unsigned int uMatched = sCUMKP._devuOther; 
		//sort 
		pcvgmNewlyAddedKeyPointLocation_->upload(cvmTmpNewlyAddedKeyPointLocation);
		pcvgmNewlyAddedKeyPointResponse_->upload(cvmTmpNewlyAddedKeyPointResponse);
		btl::device::semidense::thrustSort(pcvgmNewlyAddedKeyPointLocation_->ptr<short2>(),pcvgmNewlyAddedKeyPointResponse_->ptr<float>(),uNew);
		pcvgmNewlyAddedKeyPointLocation_->download(cvmTmpNewlyAddedKeyPointLocation);
		pcvgmNewlyAddedKeyPointResponse_->download(cvmTmpNewlyAddedKeyPointResponse);

		unsigned int uNewlyAdded = uTotalParticles_>uMatched?(uTotalParticles_-uMatched):0;	if(!uNewlyAdded) return;
		uNewlyAdded = uNewlyAdded<uNew?uNewlyAdded:uNew;//get min( uNewlyAdded, uNew );
		kernelAddNewParticles(uNewlyAdded,cvmTmpNewlyAddedKeyPointLocation.ptr<short2>(),cvmTmpNewlyAddedKeyPointResponse.ptr<float>(),
			sCUMKP._cvmParticleDescriptorCurrTmp,sCUMKP._cvmParticleResponseCurr,sCUMKP._cvmParticleDescriptorCurr);

		pcvgmParticleResponseCurr_->upload(sCUMKP._cvmParticleResponseCurr);
		pcvgmParticleDescriptorCurr_->upload(sCUMKP._cvmParticleDescriptorCurr);
		pcvgmParticleVelocityCurr_->upload(sCUMKP._cvmParticleVelocityCurr);
		pcvgmParticleAgeCurr_->upload(sCUMKP._cvmParticleAgeCurr);

		return;
}