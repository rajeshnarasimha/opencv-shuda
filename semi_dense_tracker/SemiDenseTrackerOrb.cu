#include <thrust/sort.h>

#include <opencv2/gpu/gpumat.hpp>
#include <opencv2/gpu/device/common.hpp>
#include <opencv2/gpu/device/utility.hpp>
#include <opencv2/gpu/device/functional.hpp>

#include "CudaHelper.hpp"

#define GRAY

namespace btl { namespace device {  namespace semidense  {
	

__constant__ int c_u_max[32];

void loadUMax(const int* pUMax_, int nCount_)
{
    cudaSafeCall( cudaMemcpyToSymbol(c_u_max, pUMax_, nCount_ * sizeof(int)) );
}

__global__ void kernelICAngle(const cv::gpu::PtrStepb image, const short2* loc_, const unsigned int nPoints_, const unsigned short half_k, cv::gpu::DevMem2D_<float> cvgmAngles_)
{
    __shared__ int smem[8 * 32];//Every thread in the block shares the shared memory

    volatile int* srow = smem + threadIdx.y * blockDim.x; //The volatile keyword specifies that the value associated with the name that follows can be modified by actions other than those in the user application. 

    const int nPtIdx = blockIdx.x * blockDim.y + threadIdx.y;

    if (nPtIdx < nPoints_)
    {
        int m_01 = 0, m_10 = 0;

        const short2 loc = loc_[nPtIdx];

        // Treat the center line differently, v=0
        for (int u = threadIdx.x - half_k; u <= half_k; u += blockDim.x)
            m_10 += u * image(loc.y, loc.x + u);

        cv::gpu::device::reduce<32>(srow, m_10, threadIdx.x, cv::gpu::device::plus<volatile int>());

        for (int v = 1; v <= half_k; ++v)
        {
            // Proceed over the two lines
            int v_sum = 0;
            int m_sum = 0;
            const int d = c_u_max[v];//1/4 circular patch

            for (int u = threadIdx.x - d; u <= d; u += blockDim.x)
            {
                int val_plus = image(loc.y + v, loc.x + u);
                int val_minus = image(loc.y - v, loc.x + u);

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
	const short* psPatternX_, const short* psPatternY_, const unsigned short usHalfPatchSize_,
	cv::gpu::DevMem2D_<float> cvgmParticleResponses_, cv::gpu::DevMem2D_<float> cvgmParticleAngle_, cv::gpu::DevMem2D_<int2> cvgmParticleOrbDescriptors_){

	const int nKeyPointIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (nKeyPointIdx >= uTotalParticles_) return;

	const short2& s2Loc = ps2KeyPointsLocations_[nKeyPointIdx];
	if(s2Loc.x < usHalfPatchSize_*2 || s2Loc.x >= cvgmImage_.cols - usHalfPatchSize_*2 || s2Loc.y < usHalfPatchSize_*2 || s2Loc.y >= cvgmImage_.rows - usHalfPatchSize_*2 ) return;

	const int nDescIdx = threadIdx.y + blockIdx.y * blockDim.y;

	cvgmParticleResponses_.ptr(s2Loc.y)[s2Loc.x] = pfKeyPointsResponse_[nKeyPointIdx];
	float fAngle = cvgmParticleAngle_.ptr(s2Loc.y)[s2Loc.x];
	float fSina, fCosa;  ::sincosf(fAngle, &fSina, &fCosa);
	uchar ucDesc = OrbDescriptor::calc(cvgmImage_, s2Loc, psPatternX_, psPatternY_, fSina, fCosa, nDescIdx);
	uchar* pD = (uchar*)(cvgmParticleOrbDescriptors_.ptr(s2Loc.y)+ s2Loc.x);
	pD[nDescIdx]= ucDesc;
}
// it fill in the 1.pcvgmParticleResponses_, 2.pcvgmParticleAngle_, 3.pcvgmParticleDescriptor_
// 1. pcvgmParticleResponses_ is also the input which holds the saliency score after non-max supression.
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
		psPatternX_, psPatternY_,usHalfPatchSize_,
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
	cv::gpu::DevMem2D_<uchar>  _cvgmParticlesAgePrev;
	cv::gpu::DevMem2D_<short2> _cvgmParticlesVelocityPrev;
	
	cv::gpu::DevMem2D_<uchar>  _cvgmImageCurr;
	cv::gpu::DevMem2D_<float>  _cvgmParticleAnglesCurr;
	cv::gpu::DevMem2D_<int2>   _cvgmParticleOrbDescriptorsCurr;
	cv::gpu::DevMem2D_<float>  _cvgmParticleResponsesCurr;
	cv::gpu::DevMem2D_<uchar>  _cvgmParticlesAgeCurr;
	cv::gpu::DevMem2D_<short2> _cvgmParticlesVelocityCurr;

	float _fRho;

	unsigned short _usMatchThreshold;
	unsigned short _usHalfSize;
	short _sSearchRange;

	const short* _psPatternX;
	const short* _psPatternY;

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
				atomicInc(&_devuOther, (unsigned int)(-1));//deleted particle counter increase by 1
				s2Loc = *ps2Loc_ + make_short2( c, r ); 
				fResponse = _cvgmParticleResponsesCurr.ptr(s2Loc.y)[s2Loc.x];
				if( fResponse > 0 ){
					fAngle = _cvgmParticleAnglesCurr.ptr(s2Loc.y)[s2Loc.x];
					float fSina, fCosa;  ::sincosf(fAngle, &fSina, &fCosa);
					for(short s = 0; s<8; s++)
						aDesCur[s] = OrbDescriptor::calc(_cvgmImageCurr, s2Loc, _psPatternX, _psPatternY, fSina, fCosa, s);
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
		const int c = threadIdx.x + blockIdx.x * blockDim.x;
		const int r = threadIdx.y + blockIdx.y * blockDim.y;

		if( c < 3 || c >= _cvgmImageCurr.cols - 4 || r < 3 || r >= _cvgmImageCurr.rows - 4 ) return;
		//if IsParticle( PixelLocation, cvgmParitclesResponse(i) )
		if(_cvgmParticleResponsesPrev.ptr(r)[c] < 0.2f) return;
		//A) PredictLocation = PixelLocation + ParticleVelocity(i, PixelLocation);
		short2 s2PredictLoc = make_short2(c,r);// + _cvgmParticlesVelocityPrev.ptr(r)[c];
		//B) ActualLocation = Match(PredictLocation, cvgmBlurred(i),cvgmBlurred(i+1));
		if (s2PredictLoc.x >=12 && s2PredictLoc.x < _cvgmImageCurr.cols-13 && s2PredictLoc.y >=12 && s2PredictLoc.y < _cvgmImageCurr.rows-13)
		{
			//;	devGetFastDescriptor(_cvgmBlurredPrev,r,c,&n4DesPrev);
			const uchar* pDesPrev = (uchar*) ( _cvgmParticleOrbDescriptorsPrev.ptr(r)+c);
			uchar* pDesCur = (uchar*)(_cvgmParticleOrbDescriptorsCurr.ptr(s2PredictLoc.y)+ s2PredictLoc.x);
			float fResponse = devMatchOrb( _usMatchThreshold, pDesPrev, &s2PredictLoc, &*pDesCur );
		
			if( fResponse > 0 ){
				atomicInc(&_devuNewlyAddedCounter, (unsigned int)(-1));//deleted particle counter increase by 1
				_cvgmParticleOrbDescriptorsCurr.ptr(s2PredictLoc.y)[s2PredictLoc.x]=*((int2*)pDesCur);
				_cvgmParticlesVelocityCurr.ptr(s2PredictLoc.y)[s2PredictLoc.x] = _fRho * (s2PredictLoc - make_short2(c,r)) + (1.f - _fRho)* _cvgmParticlesVelocityPrev.ptr(r)[c];//update velocity
				_cvgmParticlesAgeCurr.ptr     (s2PredictLoc.y)[s2PredictLoc.x] = _cvgmParticlesAgePrev.ptr(r)[c] + 1; //update age
				_cvgmParticleResponsesCurr.ptr(s2PredictLoc.y)[s2PredictLoc.x] = -fResponse; //update response and location //marked as matched and it will be corrected in NoMaxAndCollection
			}
			else{//C) if no match found 
				((int2*)pDesCur)->x = ((int2*)pDesCur)->y = 0;  
				atomicInc(&_devuCounter, (unsigned int)(-1));//deleted particle counter increase by 1
			}//lost
		}
		else{
			atomicInc(&_devuCounter, (unsigned int)(-1));//deleted particle counter increase by 1
		}
		return;
	}
};//class CPredictAndMatchOrb

__global__ void kernelPredictAndMatchOrb(CPredictAndMatchOrb cPAMO_){
	cPAMO_ ();
}

unsigned int cudaTrackOrb(const unsigned short usMatchThreshold_, const unsigned short usHalfSize_, const short sSearchRange_,
	const short* psPatternX_, const short* psPatternY_,
	const cv::gpu::GpuMat& cvgmParticleOrbDescriptorsPrev_, const cv::gpu::GpuMat& cvgmParticleResponsesPrev_, 
	const cv::gpu::GpuMat& cvgmParticlesAgePrev_,const cv::gpu::GpuMat& cvgmParticlesVelocityPrev_, 
	const cv::gpu::GpuMat& cvgmImage_, const cv::gpu::GpuMat& cvgmParticleAngleCurr_,
	cv::gpu::GpuMat* pcvgmParticleResponsesCurr_,
	cv::gpu::GpuMat* pcvgmParticlesAgeCurr_,cv::gpu::GpuMat* pcvgmParticlesVelocityCurr_,cv::gpu::GpuMat* pcvgmParticleOrbDescriptorsCurr_){
	
	dim3 block(32,8);
	dim3 grid;
	grid.x = cv::gpu::divUp(cvgmImage_.cols - 6, block.x); //6 is the size-1 of the Bresenham circle
    grid.y = cv::gpu::divUp(cvgmImage_.rows - 6, block.y);

	CPredictAndMatchOrb cPAMO;
	cPAMO._cvgmImageCurr = cvgmImage_;
	cPAMO._cvgmParticleOrbDescriptorsPrev = cvgmParticleOrbDescriptorsPrev_;
	cPAMO._cvgmParticleResponsesPrev = cvgmParticleResponsesPrev_;
	cPAMO._cvgmParticlesVelocityPrev = cvgmParticlesVelocityPrev_;
	cPAMO._cvgmParticlesAgePrev = cvgmParticlesAgePrev_;

	pcvgmParticlesAgeCurr_->setTo(0);
	cPAMO._cvgmParticleAnglesCurr = cvgmParticleAngleCurr_;
	cPAMO._cvgmParticleOrbDescriptorsCurr = *pcvgmParticleOrbDescriptorsCurr_;
	cPAMO._cvgmParticleResponsesCurr = *pcvgmParticleResponsesCurr_;
	cPAMO._cvgmParticlesVelocityCurr = *pcvgmParticlesVelocityCurr_;
	cPAMO._cvgmParticlesAgeCurr = *pcvgmParticlesAgeCurr_;

	cPAMO._fRho = .75f;
	cPAMO._usMatchThreshold = usMatchThreshold_;
	cPAMO._usHalfSize = usHalfSize_;
	cPAMO._sSearchRange = sSearchRange_;
	cPAMO._psPatternX = psPatternX_;
	cPAMO._psPatternY = psPatternY_;

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
	return uDeleted;

}




}//semidense
}//device
}//btl