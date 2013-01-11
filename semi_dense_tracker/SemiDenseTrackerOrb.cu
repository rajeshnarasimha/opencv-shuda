#include <thrust/sort.h>

#include <opencv2/gpu/gpumat.hpp>
#include <opencv2/gpu/device/common.hpp>
#include <opencv2/gpu/device/utility.hpp>
#include <opencv2/gpu/device/functional.hpp>

#include "CudaHelper.hpp"

#define GRAY
bool testCountResponseAndDescriptor(const cv::gpu::GpuMat cvgmParticleResponse_, const cv::gpu::GpuMat& cvgmParticleDescriptor_, int* pnCounter_);

namespace btl { namespace device {  namespace semidense  {
	

__constant__ int c_u_max[32];

void loadUMax(const int* pUMax_, int nCount_)
{
    cudaSafeCall( cudaMemcpyToSymbol(c_u_max, pUMax_, nCount_ * sizeof(int)) );
}

__global__ void kernelICAngle(const cv::gpu::PtrStepSz<uchar> cvgmImage_, const short2* loc_, const unsigned int nPoints_, const unsigned short usHalfPatch_, cv::gpu::DevMem2D_<float> cvgmAngles_)
{
    __shared__ int smem[8 * 32];//Every thread in the block shares the shared memory

    volatile int* srow = smem + threadIdx.y * blockDim.x; //The volatile keyword specifies that the value associated with 
														  //the name that follows can be modified by actions other than those in the user application. 

    const int nPtIdx = blockIdx.x * blockDim.y + threadIdx.y;

    if (nPtIdx >= nPoints_) return;
    
    int m_01 = 0, m_10 = 0;

    const short2 loc = loc_[nPtIdx];

	if (loc.x < usHalfPatch_ || loc.x >= cvgmImage_.cols - usHalfPatch_ || loc.y < usHalfPatch_ || loc.y >= cvgmImage_.rows - usHalfPatch_ ) return;

    // Treat the center line differently, v=0
    for (int u = threadIdx.x - usHalfPatch_; u <= usHalfPatch_; u += blockDim.x)
        m_10 += u * cvgmImage_(loc.y, loc.x + u);

    cv::gpu::device::reduce<32>(srow, m_10, threadIdx.x, cv::gpu::device::plus<volatile int>());

    for (int v = 1; v <= usHalfPatch_; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int m_sum = 0;
        const int d = c_u_max[v];//1/4 circular patch

        for (int u = threadIdx.x - d; u <= d; u += blockDim.x)
        {
            int val_plus = cvgmImage_(loc.y + v, loc.x + u);
            int val_minus = cvgmImage_(loc.y - v, loc.x + u);

            v_sum += (val_plus - val_minus);
            m_sum += u * (val_plus + val_minus);
        }

        cv::gpu::device::reduce<32>(srow, v_sum, threadIdx.x, cv::gpu::device::plus<volatile int>());
        cv::gpu::device::reduce<32>(srow, m_sum, threadIdx.x, cv::gpu::device::plus<volatile int>());

        m_10 += m_sum;
        m_01 += v * v_sum;
    }

    if (threadIdx.x == 0){
        float kp_dir = ::atan2f((float)m_01, (float)m_10);
        kp_dir += (kp_dir < 0) * (2.0f * CV_PI);
        //kp_dir *= 180.0f / CV_PI;
        cvgmAngles_.ptr(loc.y)[loc.x] = kp_dir;
    }
	return;
}

void cudaCalcAngles(const cv::gpu::GpuMat& cvgmImage_, const short2* pdevFinalKeyPointsLocations_, const unsigned int uPoints_,  const unsigned short usHalf_, cv::gpu::GpuMat* pcvgmParticleAngle_){
	dim3 block(32, 8);
    dim3 grid;
    grid.x = cv::gpu::divUp(uPoints_, block.y);

    kernelICAngle<<<grid, block, 0, 0>>>(cvgmImage_, pdevFinalKeyPointsLocations_, uPoints_, usHalf_, *pcvgmParticleAngle_);

    cudaSafeCall( cudaGetLastError() );
	cudaSafeCall( cudaDeviceSynchronize() );
	return;
}

#define GET_VALUE(idx) \
    cvgmImg_(s2Loc_.y + __float2int_rn(pnPatternX_[idx] * sina + pnPatternY_[idx] * cosa), \
             s2Loc_.x + __float2int_rn(pnPatternX_[idx] * cosa - pnPatternY_[idx] * sina))


struct OrbDescriptor
{
    __device__ static unsigned char calc(const cv::gpu::PtrStepb& cvgmImg_, short2 s2Loc_, const short* pnPatternX_, const short* pnPatternY_, float sina, float cosa, int nDescIdx_)
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


__global__ void kernerlCollectParticlesAndOrbDescriptors( 
	const cv::gpu::DevMem2D_<uchar> cvgmImage_,const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_,
	const unsigned int uTotalParticles_, 
	const short* psPatternX_, const short* psPatternY_, const unsigned short usHalfPatchSizeRound_,
	cv::gpu::DevMem2D_<float> cvgmParticleResponses_, cv::gpu::DevMem2D_<float> cvgmParticleAngle_, cv::gpu::DevMem2D_<int2> cvgmParticleOrbDescriptors_){

	const int nKeyPointIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (nKeyPointIdx >= uTotalParticles_) return;

	const short2& s2Loc = ps2KeyPointsLocations_[nKeyPointIdx];
	if(s2Loc.x < usHalfPatchSizeRound_ || s2Loc.x >= cvgmImage_.cols - usHalfPatchSizeRound_ || s2Loc.y < usHalfPatchSizeRound_ || s2Loc.y >= cvgmImage_.rows - usHalfPatchSizeRound_ ) return;

	const int nDescIdx = threadIdx.y + blockIdx.y * blockDim.y;

	cvgmParticleResponses_.ptr(s2Loc.y)[s2Loc.x] = pfKeyPointsResponse_[nKeyPointIdx];
	float fAngle = cvgmParticleAngle_.ptr(s2Loc.y)[s2Loc.x];
	float fSina, fCosa;  ::sincosf(fAngle, &fSina, &fCosa);
	uchar ucDesc = OrbDescriptor::calc(cvgmImage_, s2Loc, psPatternX_, psPatternY_, fSina, fCosa, nDescIdx);
	uchar* pD = (uchar*)(cvgmParticleOrbDescriptors_.ptr(s2Loc.y)+ s2Loc.x);
	pD[nDescIdx]= ucDesc;
}
// it fills the 1.pcvgmParticleResponses_, 2.pcvgmParticleAngle_, 3.pcvgmParticleDescriptor_
void cudaCollectParticlesAndOrbDescriptors(
		const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, const cv::gpu::GpuMat& cvgmImage_,
		const unsigned int uTotalParticles_, const unsigned short usHalfPatchSize_,
		const short* psPatternX_, const short* psPatternY_,
		cv::gpu::GpuMat* pcvgmParticleResponses_, cv::gpu::GpuMat* pcvgmParticleAngle_, cv::gpu::GpuMat* pcvgmParticleDescriptor_){

	if(uTotalParticles_ == 0) return;
	//calc corner angle
	cudaCalcAngles(cvgmImage_, ps2KeyPointsLocations_, uTotalParticles_,  usHalfPatchSize_, pcvgmParticleAngle_);

	dim3 block(32,8);
    dim3 grid;
    grid.x = cv::gpu::divUp(uTotalParticles_, block.x);
	grid.y = cv::gpu::divUp(8, 8);
	kernerlCollectParticlesAndOrbDescriptors<<<grid, block>>>( 
		cvgmImage_, ps2KeyPointsLocations_, pfKeyPointsResponse_, 
		uTotalParticles_, 
		psPatternX_, psPatternY_,(unsigned short)(usHalfPatchSize_*1.5), //it is the roughly sqrt(2)* usHalfPatchSize_
		*pcvgmParticleResponses_, *pcvgmParticleAngle_, *pcvgmParticleDescriptor_);
	return;
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
	cv::gpu::DevMem2D_<int2>   _cvgmParticleOrbDescriptorsPrev;
	cv::gpu::DevMem2D_<float>  _cvgmParticleResponsesPrev;
	
	cv::gpu::DevMem2D_<int2>   _cvgmParticleDescriptorCurrTmp; //store the orb descriptor for each salient point
	cv::gpu::DevMem2D_<float>  _cvgmSaliencyCurr;

	cv::gpu::DevMem2D_<uchar>  _cvgmMinMatchDistance;
	cv::gpu::DevMem2D_<short2> _cvgmMatchedLocationPrev;

	short _sSearchRange;
	unsigned short _usMatchThreshold;
	unsigned short _usHalfSize;
	unsigned short _usHalfSizeRound;//the patch will be rotated according to it main angle
									//therefore the patch half size have to be sqrt(2)*HalfSize 
	                                //it's roughly 1.5 * HalfSize

	__device__ __forceinline__ uchar dL(const uchar* pDesPrev_, const uchar* pDesCurr_) const{
		uchar ucRes = 0;
		for(short s = 0; s<8; s++)
			ucRes += _popCountTable[ pDesPrev_[s] ^ pDesCurr_[s] ];
		return ucRes;
	}
	__device__ __forceinline__ uchar devMatchOrb( const unsigned short usMatchThreshold_, 
												  const uchar* pDesPrev_, const short2 s2PredicLoc_, short2* ps2BestLoc_){
		float fResponse = 0.f;
		short2 s2Loc;
		uchar ucMinDist = 255;
		//search for the 7x7 neighbourhood for 
		for(short r = -_sSearchRange; r <= _sSearchRange; r++ ){
			for(short c = -_sSearchRange; c <= _sSearchRange; c++ ){
				s2Loc = s2PredicLoc_ + make_short2( c, r ); 
				if(s2Loc.x < _usHalfSizeRound || s2Loc.x >= _cvgmParticleResponsesPrev.cols - _usHalfSizeRound || s2Loc.y < _usHalfSizeRound || s2Loc.y >= _cvgmParticleResponsesPrev.rows - _usHalfSizeRound ) continue;
				fResponse = _cvgmSaliencyCurr.ptr(s2Loc.y)[s2Loc.x];
				if( fResponse > 0.1f ){
					const uchar* pDesCur = (uchar*)(_cvgmParticleDescriptorCurrTmp.ptr(s2Loc.y)+ s2Loc.x);
					uchar ucDist = dL(pDesPrev_,pDesCur);
					if ( ucDist < usMatchThreshold_ ){
						if (  ucMinDist > ucDist ){
							ucMinDist = ucDist;
							*ps2BestLoc_ = s2Loc;
						}
					}
				}//if sailent corner exits
			}//for 
		}//for
		return ucMinDist;
	}//devMatchOrb()
		

	__device__ __forceinline__ void operator () (){
		const int c = threadIdx.x + blockIdx.x * blockDim.x;
		const int r = threadIdx.y + blockIdx.y * blockDim.y;

		if( c < _usHalfSizeRound || c >= _cvgmParticleResponsesPrev.cols - _usHalfSizeRound || r < _usHalfSizeRound || r >= _cvgmParticleResponsesPrev.rows - _usHalfSizeRound ) return;
		if(_cvgmParticleResponsesPrev.ptr(r)[c] < 0.1f) return;
		const uchar* pDesPrev = (uchar*) ( _cvgmParticleOrbDescriptorsPrev.ptr(r)+c);

		short2 s2BestLoc; 
		const uchar ucDist = devMatchOrb( _usMatchThreshold, pDesPrev,make_short2(c,r), &s2BestLoc );
		
		if( ucDist < 64 ){ //64 is the max distance
			const uchar ucMin = _cvgmMinMatchDistance.ptr(s2BestLoc.y)[s2BestLoc.x];//competing for the same memory
			if( ucMin == uchar(0xff) ) {//it has NEVER been matched before.
				atomicInc(&_devuNewlyAddedCounter, (unsigned int)(-1));//deleted particle counter increase by 1
				_cvgmMinMatchDistance     .ptr(s2BestLoc.y)[s2BestLoc.x] = ucDist;
				_cvgmMatchedLocationPrev  .ptr(s2BestLoc.y)[s2BestLoc.x] = make_short2(c,r);
			}
			else{//it has been matched 
				//double match means one of them will be removed
				atomicInc(&_devuCounter, (unsigned int)(-1));//deleted particle counter increase by 1
				if ( ucMin > ucDist ){//record it if it is a better match than previous match
					_cvgmMinMatchDistance     .ptr(s2BestLoc.y)[s2BestLoc.x] = ucDist;
					_cvgmMatchedLocationPrev  .ptr(s2BestLoc.y)[s2BestLoc.x] = make_short2(c,r);
				}//if
			}//else
			//unlock(s2BestLoc.y,s2BestLoc.x);
		}//if
		else{//C) if no match found 
			atomicInc(&_devuCounter, (unsigned int)(-1));//deleted particle counter increase by 1
		}//lost
		return;
	}
};//class CPredictAndMatchOrb

__global__ void kernelPredictAndMatchOrb(CPredictAndMatchOrb cPAMO_ ){
	cPAMO_ ();
}
//after tracking, the matched particles are filled into the pcvgmParticleResponsesCurr_, pcvgmParticlesAgeCurr_, pcvgmParticlesVelocityCurr_, 
//and pcvgmParticleOrbDescriptorsCurr_, moreover, the cvgmSaliencyCurr_
unsigned int cudaTrackOrb(const unsigned short usMatchThreshold_, const unsigned short usHalfSize_, const short sSearchRange_,
							const cv::gpu::GpuMat& cvgmParticleOrbDescriptorPrev_, const cv::gpu::GpuMat& cvgmParticleResponsePrev_, 
							const cv::gpu::GpuMat& cvgmParticleDescriptorCurrTmp_,  const cv::gpu::GpuMat& cvgmSaliencyCurr_,
							cv::gpu::GpuMat* pcvgmMinMatchDistance_,
							cv::gpu::GpuMat* pcvgmMatchedLocationPrev_){
	
	dim3 block(32,8);
	dim3 grid;
	grid.x = cv::gpu::divUp(cvgmParticleResponsePrev_.cols, block.x);
    grid.y = cv::gpu::divUp(cvgmParticleResponsePrev_.rows, block.y);

	CPredictAndMatchOrb cPAMO;
	cPAMO._cvgmParticleOrbDescriptorsPrev = cvgmParticleOrbDescriptorPrev_;
	cPAMO._cvgmParticleResponsesPrev = cvgmParticleResponsePrev_;

	cPAMO._cvgmParticleDescriptorCurrTmp = cvgmParticleDescriptorCurrTmp_;
	cPAMO._cvgmSaliencyCurr = cvgmSaliencyCurr_;

	pcvgmMinMatchDistance_->setTo(255);
	cPAMO._cvgmMinMatchDistance = *pcvgmMinMatchDistance_;
	pcvgmMatchedLocationPrev_->setTo(cv::Scalar::all(0));
	cPAMO._cvgmMatchedLocationPrev = *pcvgmMatchedLocationPrev_; 

	cPAMO._usMatchThreshold = usMatchThreshold_;
	cPAMO._usHalfSize = usHalfSize_;
	cPAMO._usHalfSizeRound = (unsigned short)(usHalfSize_*1.5);
	cPAMO._sSearchRange = sSearchRange_;

	void* pCounter;
    cudaSafeCall( cudaGetSymbolAddress(&pCounter, _devuCounter) );
	cudaSafeCall( cudaMemset(pCounter, 0, sizeof(unsigned int)) );

	void* pCounterMatch;
    cudaSafeCall( cudaGetSymbolAddress(&pCounterMatch, _devuNewlyAddedCounter) );
	cudaSafeCall( cudaMemset(pCounterMatch, 0, sizeof(unsigned int)) );

	void* pCounterOther;
    cudaSafeCall( cudaGetSymbolAddress(&pCounterOther, _devuOther) );
	cudaSafeCall( cudaMemset(pCounterOther, 0, sizeof(unsigned int)) );

	kernelPredictAndMatchOrb<<<grid, block>>>(cPAMO);
	cudaSafeCall( cudaGetLastError() );
    cudaSafeCall( cudaDeviceSynchronize() );

    unsigned int uDeleted ;
    cudaSafeCall( cudaMemcpy(&uDeleted, pCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	unsigned int uMatched ;
    cudaSafeCall( cudaMemcpy(&uMatched, pCounterMatch, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	unsigned int uOther ;
    cudaSafeCall( cudaMemcpy(&uOther, pCounterOther, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

	return uMatched;
}//cudaTrackOrb


void thrustSort(short2* pnLoc_, float* pfResponse_, const unsigned int nCorners_);
struct SCollectUnMatchedKeyPoints{
	
	cv::gpu::DevMem2D_<float> _cvgmSaliency;
	cv::gpu::DevMem2D_<int2>  _cvgmParticleDescriptorCurrTmp;

	cv::gpu::DevMem2D_<short2> _cvgmParticleVelocityPrev;
	cv::gpu::DevMem2D_<uchar>  _cvgmParticleAgePrev;
	cv::gpu::DevMem2D_<short2> _cvgmParticleVelocityCurr;
	cv::gpu::DevMem2D_<uchar>  _cvgmParticleAgeCurr;
	cv::gpu::DevMem2D_<float>  _cvgmParticleResponseCurr;
	cv::gpu::DevMem2D_<int2>   _cvgmParticleDescriptorCurr;

	cv::gpu::DevMem2D_<short2> _cvgmMatchedLocationPrev;
	cv::gpu::DevMem2D_<uchar>  _cvgmMinMatchDistance;

	unsigned int _uMaxMatchedKeyPoint;
	unsigned int _uMaxNewKeyPoint;
	float _fRho;
	short2* _ps2NewlyAddedKeyPointLocation; 
	float*  _pfNewlyAddedKeyPointResponse;

	short2* _ps2MatchedKeyPointLocation; 
	float*  _pfMatchedKeyPointResponse;

	
	__device__ __forceinline__ void operator () (){
		const int c = threadIdx.x + blockIdx.x * blockDim.x;
		const int r = threadIdx.y + blockIdx.y * blockDim.y;

		if( c < 0 || c >= _cvgmParticleResponseCurr.cols || r < 0 || r >= _cvgmParticleResponseCurr.rows ) return;
		_cvgmParticleVelocityCurr  .ptr(r)[c] = make_short2(0,0);
		_cvgmParticleAgeCurr	   .ptr(r)[c] = 0;
		_cvgmParticleResponseCurr  .ptr(r)[c] = 0.f;
		_cvgmParticleDescriptorCurr.ptr(r)[c] = make_int2(0,0);
		const float& fResponse = _cvgmSaliency.ptr(r)[c];

		if( fResponse < 0.1f ) return; 

		if(_cvgmMinMatchDistance.ptr(r)[c] == 255 ){
			const unsigned int nIdx = atomicInc(&_devuCounter, (unsigned int)(-1));//count Else
			if (nIdx >= _uMaxNewKeyPoint) return;
			_ps2NewlyAddedKeyPointLocation[nIdx] = make_short2(c,r);
			_pfNewlyAddedKeyPointResponse[nIdx]  = fResponse ;
		}
		else{
			const short2& s2PrevLoc = _cvgmMatchedLocationPrev.ptr(r)[c];
			
			const unsigned int nIdx = atomicInc(&_devuOther, (unsigned int)(-1));//count Matched
			if( nIdx >= _uMaxMatchedKeyPoint) return;
			_ps2MatchedKeyPointLocation[nIdx] = make_short2(c,r);
			_pfMatchedKeyPointResponse[nIdx]  = fResponse;
			
			_cvgmParticleResponseCurr  .ptr(r)[c] = fResponse; 
			_cvgmParticleDescriptorCurr.ptr(r)[c] = _cvgmParticleDescriptorCurrTmp.ptr(r)[c];
			_cvgmParticleVelocityCurr  .ptr(r)[c] = make_short2(c,r) - s2PrevLoc;
				//convert2s2( _fRho * (make_short2(c,r) - s2PrevLoc) + (1.f - _fRho)* _cvgmParticleVelocityPrev.ptr(s2PrevLoc.y)[s2PrevLoc.x] + make_float2(.5f,.5f));//update velocity
			_cvgmParticleAgeCurr	   .ptr(r)[c] = _cvgmParticleAgePrev.ptr(s2PrevLoc.y)[s2PrevLoc.x] + 1; //update age
		}
		return;
	}//operator()
};//SCollectUnMatchedKeyPoints
__global__ void kernelCollectUnMatched(SCollectUnMatchedKeyPoints sCUMKP_){
	sCUMKP_ ();
}

__global__ void kernerlAddNewParticles( const unsigned int uTotalParticles_,   
										const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, 
										const cv::gpu::DevMem2D_<int2> cvgmParticleDescriptorTmp_,
										cv::gpu::DevMem2D_<float> cvgmParticleResponse_, cv::gpu::DevMem2D_<int2> cvgmParticleDescriptor_){

	const int nKeyPointIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (nKeyPointIdx >= uTotalParticles_) return;

	const short2& s2Loc = ps2KeyPointsLocations_[nKeyPointIdx];
	cvgmParticleResponse_.ptr(s2Loc.y)[s2Loc.x] = pfKeyPointsResponse_[nKeyPointIdx];
	cvgmParticleDescriptor_.ptr(s2Loc.y)[s2Loc.x] = cvgmParticleDescriptorTmp_.ptr(s2Loc.y)[s2Loc.x]; 
	return; 
}

void cudaCollectKeyPointOrb(unsigned int uTotalParticles_, unsigned int uMaxNewKeyPoints_, const float fRho_,
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
	
	sCUMKP._cvgmSaliency				  = cvgmSaliency_;//store all non-max salient points
	sCUMKP._cvgmParticleDescriptorCurrTmp = cvgmParticleDescriptorCurrTmp_;//store all non-max salient descriptors

	sCUMKP._cvgmParticleVelocityPrev = cvgmParticleVelocityPrev_;
	sCUMKP._cvgmParticleAgePrev = cvgmParticleAgePrev_;

	sCUMKP._cvgmMinMatchDistance = cvgmMinMatchDistance_;
	sCUMKP._cvgmMatchedLocationPrev = cvgmMatchedLocationPrev_;

	sCUMKP._cvgmParticleResponseCurr = *pcvgmParticleResponseCurr_;
	sCUMKP._cvgmParticleDescriptorCurr = *pcvgmParticleDescriptorCurr_;
	sCUMKP._cvgmParticleVelocityCurr = *pcvgmParticleVelocityCurr_;
	sCUMKP._cvgmParticleAgeCurr = *pcvgmParticleAgeCurr_;

	sCUMKP._uMaxMatchedKeyPoint = uTotalParticles_;
	sCUMKP._uMaxNewKeyPoint     = uMaxNewKeyPoints_; //the size of the newly added keypoint
	sCUMKP._fRho                = fRho_;

	sCUMKP._ps2NewlyAddedKeyPointLocation = pcvgmNewlyAddedKeyPointLocation_->ptr<short2>(); 
	sCUMKP._pfNewlyAddedKeyPointResponse  = pcvgmNewlyAddedKeyPointResponse_->ptr<float>();
	sCUMKP._ps2MatchedKeyPointLocation    = pcvgmMatchedKeyPointLocation_->ptr<short2>(); 
	sCUMKP._pfMatchedKeyPointResponse     = pcvgmMatchedKeyPointResponse_->ptr<float>();

	void* pNewCounter;
    cudaSafeCall( cudaGetSymbolAddress(&pNewCounter, _devuCounter) );
	cudaSafeCall( cudaMemset(pNewCounter, 0, sizeof(unsigned int)) );
	
	void* pMatchedCounter;
    cudaSafeCall( cudaGetSymbolAddress(&pMatchedCounter, _devuOther) );
	cudaSafeCall( cudaMemset(pMatchedCounter, 0, sizeof(unsigned int)) );

	dim3 block(32,8);
	dim3 grid;
	grid.x = cv::gpu::divUp(pcvgmParticleResponseCurr_->cols, block.x);
    grid.y = cv::gpu::divUp(pcvgmParticleResponseCurr_->rows, block.y);
	//collect new(unmatched) and matched
	kernelCollectUnMatched<<<grid, block>>>(sCUMKP);
	cudaSafeCall( cudaGetLastError() );

	/*int nCount = 0;
	bool bIsLegal = testCountResponseAndDescriptor(*pcvgmParticleResponseCurr_, *pcvgmParticleDescriptorCurr_, &nCount);*/
	unsigned int uNew;
    cudaSafeCall( cudaMemcpy(&uNew, pNewCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	unsigned int uMatched;
    cudaSafeCall( cudaMemcpy(&uMatched, pMatchedCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

	//sort 
	thrustSort(pcvgmNewlyAddedKeyPointLocation_->ptr<short2>(), pcvgmNewlyAddedKeyPointResponse_->ptr<float>(), uNew);
	
	unsigned int uNewlyAdded = uTotalParticles_>uMatched?(uTotalParticles_-uMatched):0;	if(!uNewlyAdded) return;
	uNewlyAdded = uNewlyAdded<uNew?uNewlyAdded:uNew;//get min( uNewlyAdded, uNew );
	//add the first uTotalParticles_ 
	grid.x = cv::gpu::divUp(uTotalParticles_, block.x);
	grid.y = cv::gpu::divUp(8, 8);
	kernerlAddNewParticles<<<grid, block>>>(uNewlyAdded, pcvgmNewlyAddedKeyPointLocation_->ptr<short2>(), pcvgmNewlyAddedKeyPointResponse_->ptr<float>(),
											sCUMKP._cvgmParticleDescriptorCurrTmp ,
											sCUMKP._cvgmParticleResponseCurr, sCUMKP._cvgmParticleDescriptorCurr);
	cudaSafeCall( cudaGetLastError() );
}


}//semidense
}//device
}//btl