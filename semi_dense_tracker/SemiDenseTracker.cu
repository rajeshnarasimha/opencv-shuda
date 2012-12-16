#include <thrust/sort.h>

#include <opencv2/gpu/gpumat.hpp>
#include <opencv2/gpu/device/common.hpp>
#include <opencv2/gpu/device/utility.hpp>
#include <opencv2/gpu/device/functional.hpp>

#define GRAY

namespace btl { namespace device {  namespace semidense  {

__device__ void devUpdateMaxConstrast(const uchar3& Color_, const uchar3& Center_, float* pfConMax_ ){
	float fC = abs(Center_.x - Color_.x);
	*pfConMax_ = *pfConMax_ > fC? *pfConMax_ :fC;
	fC = abs(Center_.y - Color_.y);
	*pfConMax_ = *pfConMax_ > fC? *pfConMax_ :fC;
	fC = abs(Center_.z - Color_.z);
	*pfConMax_ = *pfConMax_ > fC? *pfConMax_ :fC;
}

__device__ void devUpdateMaxConstrast(const uchar3& Color_, const float& fCenter_, float* pfConMax_ ){
	float fC = abs(fCenter_ - (Color_.x + Color_.y + Color_.z)/3.f );
	*pfConMax_ = *pfConMax_ > fC? *pfConMax_ :fC;
}

__device__ float devCalcMaxContrast(const cv::gpu::DevMem2D_<uchar3>& cvgmImage_, const int r, const int c ){
	const uchar3& Center = cvgmImage_.ptr(r)[c];
#ifdef GRAY
	float fCenter = ( Center.x + Center.y + Center.z )/3.f;
#endif
	float fConMax = -1.f; 
	uchar3 Color;

	Color = cvgmImage_.ptr(r-3)[c  ];//1
#ifdef GRAY
	devUpdateMaxConstrast(Color, fCenter, &fConMax);
#else
	devUpdateMaxConstrast(Color, Center, &fConMax);
#endif

	Color = cvgmImage_.ptr(r-3)[c+1];//2
#ifdef GRAY
	devUpdateMaxConstrast(Color, fCenter, &fConMax);
#else
	devUpdateMaxConstrast(Color, Center, &fConMax);
#endif
	
	Color = cvgmImage_.ptr(r-2)[c+2];//3
#ifdef GRAY
	devUpdateMaxConstrast(Color, fCenter, &fConMax);
#else
	devUpdateMaxConstrast(Color, Center, &fConMax);
#endif
	
	Color = cvgmImage_.ptr(r-1)[c+3];//4
#ifdef GRAY
	devUpdateMaxConstrast(Color, fCenter, &fConMax);
#else
	devUpdateMaxConstrast(Color, Center, &fConMax);
#endif

	Color = cvgmImage_.ptr(r  )[c+3];//5
#ifdef GRAY
	devUpdateMaxConstrast(Color, fCenter, &fConMax);
#else
	devUpdateMaxConstrast(Color, Center, &fConMax);
#endif

	Color = cvgmImage_.ptr(r+1)[c+3];//6
#ifdef GRAY
	devUpdateMaxConstrast(Color, fCenter, &fConMax);
#else
	devUpdateMaxConstrast(Color, Center, &fConMax);
#endif

	Color = cvgmImage_.ptr(r+2)[c+2];//7
#ifdef GRAY
	devUpdateMaxConstrast(Color, fCenter, &fConMax);
#else
	devUpdateMaxConstrast(Color, Center, &fConMax);
#endif

	Color = cvgmImage_.ptr(r+3)[c+1];//8
#ifdef GRAY
	devUpdateMaxConstrast(Color, fCenter, &fConMax);
#else
	devUpdateMaxConstrast(Color, Center, &fConMax);
#endif

	Color = cvgmImage_.ptr(r+3)[c  ];//9
#ifdef GRAY
	devUpdateMaxConstrast(Color, fCenter, &fConMax);
#else
	devUpdateMaxConstrast(Color, Center, &fConMax);
#endif

	Color= cvgmImage_.ptr(r+3)[c-1];//10
#ifdef GRAY
	devUpdateMaxConstrast(Color, fCenter, &fConMax);
#else
	devUpdateMaxConstrast(Color, Center, &fConMax);
#endif

	Color= cvgmImage_.ptr(r+2)[c-2];//11
#ifdef GRAY
	devUpdateMaxConstrast(Color, fCenter, &fConMax);
#else
	devUpdateMaxConstrast(Color, Center, &fConMax);
#endif

	Color= cvgmImage_.ptr(r+1)[c-3];//12
#ifdef GRAY
	devUpdateMaxConstrast(Color, fCenter, &fConMax);
#else
	devUpdateMaxConstrast(Color, Center, &fConMax);
#endif

	Color= cvgmImage_.ptr(r  )[c-3];//13
#ifdef GRAY
	devUpdateMaxConstrast(Color, fCenter, &fConMax);
#else
	devUpdateMaxConstrast(Color, Center, &fConMax);
#endif

	Color= cvgmImage_.ptr(r-1)[c-3];//14
#ifdef GRAY
	devUpdateMaxConstrast(Color, fCenter, &fConMax);
#else
	devUpdateMaxConstrast(Color, Center, &fConMax);
#endif

	Color= cvgmImage_.ptr(r-2)[c-2];//15
#ifdef GRAY
	devUpdateMaxConstrast(Color, fCenter, &fConMax);
#else
	devUpdateMaxConstrast(Color, Center, &fConMax);
#endif

	Color= cvgmImage_.ptr(r-3)[c-1];//16
#ifdef GRAY
	devUpdateMaxConstrast(Color, fCenter, &fConMax);
#else
	devUpdateMaxConstrast(Color, Center, &fConMax);
#endif
	
	return fConMax;
}

__global__ void kernelCalcMaxContrast(const cv::gpu::DevMem2D_<uchar3> cvgmImage_, const unsigned char ucContrastThreshold_, cv::gpu::DevMem2D_<float> cvgmContrast_ ){
	const int c = threadIdx.x + blockIdx.x * blockDim.x + 3;
    const int r = threadIdx.y + blockIdx.y * blockDim.y + 3;

	if( c < 3 || c > cvgmImage_.cols - 4 || r < 3 || r > cvgmImage_.rows - 4 ) return;

	float& fC = cvgmContrast_.ptr(r)[c];
	fC = devCalcMaxContrast(cvgmImage_, r, c);
	fC = fC > ucContrastThreshold_? fC:0;
}

void cudaCalcMaxContrast(const cv::gpu::GpuMat& cvgmImage_, const unsigned char ucContrastThreshold_, cv::gpu::GpuMat* pcvgmContrast_){
	dim3 block(32, 8);

    dim3 grid;
    grid.x = cv::gpu::divUp(cvgmImage_.cols - 6, block.x); //6 is the size-1 of the Bresenham circle
    grid.y = cv::gpu::divUp(cvgmImage_.rows - 6, block.y);

	kernelCalcMaxContrast<<<grid, block>>>(cvgmImage_, ucContrastThreshold_, *pcvgmContrast_);
}

// given two pixels in the diameter
// is it smaller than MinContrast? if yes, then MinContrast will be updated
__device__ void devUpdateMinContrast( const uchar3& uc3Color1_, const uchar3& uc3Color2_, const float& fCenter_, float* pfMinContrast_){
	float fC = .5f * abs( 2.f * fCenter_ - (uc3Color1_.x + uc3Color1_.y + uc3Color1_.z  + uc3Color2_.x + uc3Color2_.y + uc3Color2_.z )/3.f);
	*pfMinContrast_ = *pfMinContrast_ < fC? *pfMinContrast_:fC;
}
__device__ float devCalcMinDiameterContrast(const cv::gpu::DevMem2D_<uchar3>& cvgmImage_, int r, int c){
	const uchar3& Center = cvgmImage_.ptr(r)[c];
	float fCenter = (Center.x + Center.y + Center.z)/3.f;
	//float fColor1, fColor2;
	float fConMin =300.f; 
	//float fC;
	uchar3 Color1, Color2;

	Color1 = cvgmImage_.ptr(r-3)[c  ];//1
	Color2 = cvgmImage_.ptr(r+3)[c  ];//9
	devUpdateMinContrast( Color1, Color2, fCenter, &fConMin );
	
	Color1 = cvgmImage_.ptr(r-3)[c+1];//2
	Color2 = cvgmImage_.ptr(r+3)[c-1];//10
	devUpdateMinContrast( Color1, Color2, fCenter, &fConMin );

	Color1 = cvgmImage_.ptr(r-2)[c+2];//3
	Color2 = cvgmImage_.ptr(r+2)[c-2];//11
	devUpdateMinContrast( Color1, Color2, fCenter, &fConMin );

	Color1 = cvgmImage_.ptr(r-1)[c+3];//4
	Color2 = cvgmImage_.ptr(r+1)[c-3];//12
	devUpdateMinContrast( Color1, Color2, fCenter, &fConMin );

	Color1 = cvgmImage_.ptr(r  )[c+3];//5
	Color2 = cvgmImage_.ptr(r  )[c-3];//13
	devUpdateMinContrast( Color1, Color2, fCenter, &fConMin );

	Color1 = cvgmImage_.ptr(r+1)[c+3];//6
	Color2 = cvgmImage_.ptr(r-1)[c-3];//14
	devUpdateMinContrast( Color1, Color2, fCenter, &fConMin );

	Color1 = cvgmImage_.ptr(r+2)[c+2];//7
	Color2 = cvgmImage_.ptr(r-2)[c-2];//15
	devUpdateMinContrast( Color1, Color2, fCenter, &fConMin );

	Color1 = cvgmImage_.ptr(r+3)[c+1];//8
	Color2 = cvgmImage_.ptr(r-3)[c-1];//16
	devUpdateMinContrast( Color1, Color2, fCenter, &fConMin );

	return fConMin;
}
// given two pixels in the diameter
// is it smaller than MinContrast? if yes, then MinContrast will be updated
__device__ void devUpdateMinContrast2( const uchar3& uc3Color1_, const uchar3& uc3Color2_, const uchar3& uc3Center_, float* pfMinContrast_){
	float fM = -1.f;
	float fC;
	fC = .5f * abs( 2.f * uc3Center_.x - uc3Color1_.x - uc3Color2_.x );
	fM = fM > fC ? fM : fC;
	fC = .5f * abs( 2.f * uc3Center_.y - uc3Color1_.y - uc3Color2_.y );
	fM = fM > fC ? fM : fC;
	fC = .5f * abs( 2.f * uc3Center_.z - uc3Color1_.z - uc3Color2_.z );
	fM = fM > fC ? fM : fC;
	*pfMinContrast_ = *pfMinContrast_ < fM? *pfMinContrast_:fM;

}
__device__ float devCalcMinDiameterContrast2(const cv::gpu::DevMem2D_<uchar3>& cvgmImage_, int r, int c){
	const uchar3& Center = cvgmImage_.ptr(r)[c];
	//float fColor1, fColor2;
	float fConMin =300.f; 
	//float fC;
	uchar3 Color1, Color2;

	Color1 = cvgmImage_.ptr(r-3)[c  ];//1
	Color2 = cvgmImage_.ptr(r+3)[c  ];//9
	devUpdateMinContrast2( Color1, Color2, Center, &fConMin );
	
	Color1 = cvgmImage_.ptr(r-3)[c+1];//2
	Color2 = cvgmImage_.ptr(r+3)[c-1];//10
	devUpdateMinContrast2( Color1, Color2, Center, &fConMin );

	Color1 = cvgmImage_.ptr(r-2)[c+2];//3
	Color2 = cvgmImage_.ptr(r+2)[c-2];//11
	devUpdateMinContrast2( Color1, Color2, Center, &fConMin );

	Color1 = cvgmImage_.ptr(r-1)[c+3];//4
	Color2 = cvgmImage_.ptr(r+1)[c-3];//12
	devUpdateMinContrast2( Color1, Color2, Center, &fConMin );

	Color1 = cvgmImage_.ptr(r  )[c+3];//5
	Color2 = cvgmImage_.ptr(r  )[c-3];//13
	devUpdateMinContrast2( Color1, Color2, Center, &fConMin );

	Color1 = cvgmImage_.ptr(r+1)[c+3];//6
	Color2 = cvgmImage_.ptr(r-1)[c-3];//14
	devUpdateMinContrast2( Color1, Color2, Center, &fConMin );

	Color1 = cvgmImage_.ptr(r+2)[c+2];//7
	Color2 = cvgmImage_.ptr(r-2)[c-2];//15
	devUpdateMinContrast2( Color1, Color2, Center, &fConMin );

	Color1 = cvgmImage_.ptr(r+3)[c+1];//8
	Color2 = cvgmImage_.ptr(r-3)[c-1];//16
	devUpdateMinContrast2( Color1, Color2, Center, &fConMin );

	return fConMin;
}

__global__ void kernelCalcMinDiameterContrast(const cv::gpu::DevMem2D_<uchar3> cvgmImage_, cv::gpu::DevMem2D_<float> cvgmContrast_ ){
	const int c = threadIdx.x + blockIdx.x * blockDim.x + 3;
    const int r = threadIdx.y + blockIdx.y * blockDim.y + 3;

	if( c < 3 || c > cvgmImage_.cols - 4 || r < 3 || r > cvgmImage_.rows - 4 ) return;
	cvgmContrast_.ptr(r)[c] = devCalcMinDiameterContrast2(cvgmImage_, r, c );
}

void cudaCalcMinDiameterContrast(const cv::gpu::GpuMat& cvgmImage_, cv::gpu::GpuMat* pcvgmContrast_){
	dim3 block(32, 8);

    dim3 grid;
    grid.x = cv::gpu::divUp(cvgmImage_.cols - 6, block.x); //6 is the size-1 of the Bresenham circle
    grid.y = cv::gpu::divUp(cvgmImage_.rows - 6, block.y);

	kernelCalcMinDiameterContrast<<<grid, block>>>(cvgmImage_, *pcvgmContrast_);

}

__device__ unsigned int _devuCounter = 0;

__global__ void kernelCalcSaliency(const cv::gpu::DevMem2D_<uchar3> cvgmImage_, const unsigned char ucContrastThreshold_, const float fSaliencyThreshold_, 
	cv::gpu::DevMem2D_<float> cvgmSaliency_, cv::gpu::DevMem2D_<short2> cvgmKeyPointLocations_){
	const int c = threadIdx.x + blockIdx.x * blockDim.x + 3;
    const int r = threadIdx.y + blockIdx.y * blockDim.y + 3;

	if( c < 3 || c > cvgmImage_.cols - 4 || r < 3 || r > cvgmImage_.rows - 4 ) return;
	float& fSaliency = cvgmSaliency_.ptr(r)[c];

	float fMaxContrast = devCalcMaxContrast(cvgmImage_, r, c );
	//fSaliency = fMaxContrast;
	if(fMaxContrast > ucContrastThreshold_){
		fSaliency = devCalcMinDiameterContrast(cvgmImage_, r, c )/fMaxContrast;
		if (fSaliency > fSaliencyThreshold_){
			const unsigned int nIdx = atomicInc(&_devuCounter, (unsigned int)(-1));

            if (nIdx < cvgmKeyPointLocations_.cols)
				cvgmKeyPointLocations_.ptr(0)[nIdx] = make_short2(c, r);
		}
		else
			fSaliency = 0.f;
	}
	else
		fSaliency = 0.f;
	return;
}

//return the No. of Salient pixels above fSaliencyThreshold_
unsigned int cudaCalcSaliency(const cv::gpu::GpuMat& cvgmImage_, const unsigned char ucContrastThreshold_, const float& fSaliencyThreshold_, cv::gpu::GpuMat* pcvgmSaliency_, cv::gpu::GpuMat* pcvgmKeyPointLocations_){
	void* pCounter;
    cudaSafeCall( cudaGetSymbolAddress(&pCounter, _devuCounter) );

	dim3 block(32, 8);
    dim3 grid;
    grid.x = cv::gpu::divUp(cvgmImage_.cols - 6, block.x); //6 is the size-1 of the Bresenham circle
    grid.y = cv::gpu::divUp(cvgmImage_.rows - 6, block.y);

	kernelCalcSaliency<<<grid, block>>>(cvgmImage_, ucContrastThreshold_, fSaliencyThreshold_, *pcvgmSaliency_, *pcvgmKeyPointLocations_);
	cudaSafeCall( cudaGetLastError() );
    cudaSafeCall( cudaDeviceSynchronize() );

    unsigned int uCount;
    cudaSafeCall( cudaMemcpy(&uCount, pCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

    return uCount;
}

__device__ void devGetFastDescriptor(const cv::gpu::DevMem2D_<uchar3>& cvgmImage_, const int r, const int c, int4* pDescriptor_ ){
	pDescriptor_->x = pDescriptor_->y = pDescriptor_->z = pDescriptor_->w = 0;
	uchar3 Color;
	Color = cvgmImage_.ptr(r-3)[c  ];//1
	pDescriptor_->x += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->x = pDescriptor_->x << 8;
	Color = cvgmImage_.ptr(r-3)[c+1];//2
	pDescriptor_->x += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->x = pDescriptor_->x << 8;
	Color = cvgmImage_.ptr(r-2)[c+2];//3
	pDescriptor_->x += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->x = pDescriptor_->x << 8;
	Color = cvgmImage_.ptr(r-1)[c+3];//4
	pDescriptor_->x += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 


	Color = cvgmImage_.ptr(r  )[c+3];//5
	pDescriptor_->y += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->y = pDescriptor_->y << 8;
	Color = cvgmImage_.ptr(r+1)[c+3];//6
	pDescriptor_->y += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->y = pDescriptor_->y << 8;
	Color = cvgmImage_.ptr(r+2)[c+2];//7
	pDescriptor_->y += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->y = pDescriptor_->y << 8;
	Color = cvgmImage_.ptr(r+3)[c+1];//8
	pDescriptor_->y += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 

	Color = cvgmImage_.ptr(r+3)[c  ];//9
	pDescriptor_->z += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->z = pDescriptor_->z << 8;
	Color= cvgmImage_.ptr(r+3)[c-1];//10
	pDescriptor_->z += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->z = pDescriptor_->z << 8;
	Color= cvgmImage_.ptr(r+2)[c-2];//11
	pDescriptor_->z += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->z = pDescriptor_->z << 8;
	Color= cvgmImage_.ptr(r+1)[c-3];//12
	pDescriptor_->z += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	
	Color= cvgmImage_.ptr(r  )[c-3];//13
	pDescriptor_->w += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->w = pDescriptor_->w << 8;
	Color= cvgmImage_.ptr(r-1)[c-3];//14
	pDescriptor_->w += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->w = pDescriptor_->w << 8;
	Color= cvgmImage_.ptr(r-2)[c-2];//15
	pDescriptor_->w += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->w = pDescriptor_->w << 8;
	Color= cvgmImage_.ptr(r-3)[c-1];//16
	pDescriptor_->w += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	return;
}

///////////////////////////////////////////////////////////////////////////
// kernelNonMaxSupression
// supress all other corners in 3x3 area only keep the strongest corner
//
__global__ void kernelNonMaxSupression(const cv::gpu::DevMem2D_<uchar3> cvgmImage_, const short2* ps2KeyPointLoc_,const int nCount_, cv::gpu::PtrStepSzf cvgmScore_, short2* ps2LocFinal_, float* pfResponseFinal_)
{
    const int nKeyPointIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (nKeyPointIdx < nCount_)
    {
        short2 s2Location = ps2KeyPointLoc_[nKeyPointIdx];

        float& fScore = cvgmScore_(s2Location.y, s2Location.x);
		//check whether the current corner is the max in 3x3 local area
        bool bIsMax =
            fScore > cvgmScore_(s2Location.y - 1, s2Location.x - 1) &&
            fScore > cvgmScore_(s2Location.y - 1, s2Location.x    ) &&
            fScore > cvgmScore_(s2Location.y - 1, s2Location.x + 1) &&

            fScore > cvgmScore_(s2Location.y    , s2Location.x - 1) &&
            fScore > cvgmScore_(s2Location.y    , s2Location.x + 1) &&

            fScore > cvgmScore_(s2Location.y + 1, s2Location.x - 1) &&
            fScore > cvgmScore_(s2Location.y + 1, s2Location.x    ) &&
            fScore > cvgmScore_(s2Location.y + 1, s2Location.x + 1);

        if (bIsMax){
            const unsigned int nIdx = atomicInc(&_devuCounter, (unsigned int)(-1));
            ps2LocFinal_[nIdx] = s2Location;
            pfResponseFinal_[nIdx] = fScore;
        }
		else{
			fScore = 0.f;
		}
    }
	return;
}

unsigned int cudaNonMaxSupression(const cv::gpu::GpuMat& cvgmImage_, const cv::gpu::GpuMat& cvgmKeyPointLocation_, const unsigned int uMaxSalientPoints_, cv::gpu::GpuMat* pcvgmSaliency_, short2* ps2devLocations_, float* pfdevResponse_){
	void* pCounter;
    cudaSafeCall( cudaGetSymbolAddress(&pCounter, _devuCounter) );

    dim3 block(256);
    dim3 grid;
    grid.x = cv::gpu::divUp(uMaxSalientPoints_, block.x);

    cudaSafeCall( cudaMemset(pCounter, 0, sizeof(unsigned int)) );

    kernelNonMaxSupression<<<grid, block>>>(cvgmImage_, cvgmKeyPointLocation_.ptr<short2>(), uMaxSalientPoints_, *pcvgmSaliency_, ps2devLocations_, pfdevResponse_);
    cudaSafeCall( cudaGetLastError() );
    cudaSafeCall( cudaDeviceSynchronize() );

    unsigned int uFinalCount;
    cudaSafeCall( cudaMemcpy(&uFinalCount, pCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

    return uFinalCount;
}

void thrustSort(short2* pnLoc_, float* pfResponse_, const unsigned int nCorners_)
{
    thrust::device_ptr<short2> loc_ptr(pnLoc_);
    thrust::device_ptr<float> response_ptr(pfResponse_);
    thrust::sort_by_key(response_ptr, response_ptr + nCorners_, loc_ptr, thrust::greater<float>());
    return;
}

__global__ void kernelFastDescriptors(cv::gpu::DevMem2D_<uchar3> cvgmImage_, const short2* ps2KeyPointLoc_, const unsigned int uMaxSailentPoints_, int4* pn4devDescriptor_ )
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 110)

    const int nKeyPointIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nKeyPointIdx < uMaxSailentPoints_){
		short2 s2Location = ps2KeyPointLoc_[nKeyPointIdx];
		int4 n4Descriptor;
		devGetFastDescriptor(cvgmImage_,s2Location.y,s2Location.x,&n4Descriptor );
		pn4devDescriptor_[nKeyPointIdx] = n4Descriptor;
	}
#endif
}

void cudaFastDescriptors(const cv::gpu::GpuMat& cvgmImage_, unsigned int uFinalSalientPoints_, cv::gpu::GpuMat* pcvgmKeyPointsLocations_, cv::gpu::GpuMat* pcvgmParticlesDescriptors_){
	dim3 block(256);
    dim3 grid;
    grid.x = cv::gpu::divUp(uFinalSalientPoints_, block.x);

	kernelFastDescriptors<<<grid, block>>>(cvgmImage_, pcvgmKeyPointsLocations_->ptr<short2>(), uFinalSalientPoints_, pcvgmParticlesDescriptors_->ptr<int4>());
    cudaSafeCall( cudaGetLastError() );
    cudaSafeCall( cudaDeviceSynchronize() );
}
__device__ short2 operator + (const short2 s2O1_, const short2 s2O2_){
	return make_short2(s2O1_.x + s2O2_.x,s2O1_.y + s2O2_.y);
}
__device__ short2 operator - (const short2 s2O1_, const short2 s2O2_){
	return make_short2(s2O1_.x - s2O2_.x,s2O1_.y - s2O2_.y);
}

__device__ short2 operator * (const float fO1_, const short2 s2O2_){
	return make_short2( __float2int_rn(fO1_* s2O2_.x),__float2int_rn( fO1_ * s2O2_.y));
}



__device__ float dL1(const int4& n4Descriptor1_, const int4& n4Descriptor2_){
	return 255.f;
}

__device__ float devMatch(short2* ps2Loc_, const cv::gpu::DevMem2D_<uchar3>& cvgmImage_, const cv::gpu::DevMem2D_<float>& cvgmScore_, int4* pn4Descriptor_){
	float fResponse = 0.f;
	short2 s2Loc;
	float fMinDist = 300.f;
	for(short r = -3; r < 4; r++ )
	for(short c = -3; c < 4; c++ ){
		s2Loc = *ps2Loc_ + make_short2( c, r ); 
		fResponse = cvgmScore_.ptr(s2Loc.y)[s2Loc.x];
		if( fResponse > 0 ){
			int4 n4Des; 
			devGetFastDescriptor(cvgmImage_,s2Loc.y,s2Loc.x,&n4Des);
			float fDist = dL1(n4Des,*pn4Descriptor_);
			if ( fDist < 76 ){
				if (  fMinDist > fDist ){
					fMinDist = fDist;
					*pn4Descriptor_ = n4Des;
				}
			}
		}//if sailent corner exits
	}//for for
	if(fMinDist < 300.f)
		return fResponse;
	else
		return -1.f;
}

__global__ void kernerlPredictAndMatch(const unsigned int uMaxSailentPoints_, const cv::gpu::DevMem2D_<uchar3> cvgmImage_,const cv::gpu::DevMem2D_<float> cvgmScore_, short2* ps2KeyPointLoc_, short2* ps2ParticlesVelocity_, int4* pn4Descriptor_, uchar* pucAge_, float* pfResponse_ ){
	const int nKeyPointIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (nKeyPointIdx >= uMaxSailentPoints_) return;
	//predict the next position
	short2 s2PredictNewLoc = ps2KeyPointLoc_[nKeyPointIdx] + ps2ParticlesVelocity_[nKeyPointIdx];
	float fResponse = devMatch( &s2PredictNewLoc, cvgmImage_, cvgmScore_, &(pn4Descriptor_[nKeyPointIdx]));
	if( fResponse > 0 ){
		ps2ParticlesVelocity_[nKeyPointIdx] = s2PredictNewLoc - ps2KeyPointLoc_[nKeyPointIdx];//update velocity
		pucAge_[nKeyPointIdx] ++;//update age
		ps2KeyPointLoc_[nKeyPointIdx] = s2PredictNewLoc;//update location
		pfResponse_[nKeyPointIdx] = fResponse;//update response
	}
	else{
		pfResponse_[nKeyPointIdx] = 0.f;
		atomicInc(&_devuCounter, (unsigned int)(-1));
	}//lost
	return;
}
unsigned int cudaPredictAndMatch(const unsigned int uFinalSalientPoints_, const cv::gpu::GpuMat& cvgmImage_,const cv::gpu::GpuMat& cvgmSaliency_, cv::gpu::GpuMat& cvgmFinalKeyPointsLocations_,cv::gpu::GpuMat& cvgmFinalKeyPointsResponse_,cv::gpu::GpuMat& cvgmParticlesAge_,cv::gpu::GpuMat& cvgmParticlesVelocity_, cv::gpu::GpuMat& cvgmParticlesDescriptors_){
	void* pCounter;
    cudaSafeCall( cudaGetSymbolAddress(&pCounter, _devuCounter) );
	cudaSafeCall( cudaMemset(pCounter, 0, sizeof(unsigned int)) );
	dim3 block(256);
    dim3 grid;
    grid.x = cv::gpu::divUp(uFinalSalientPoints_, block.x);

	kernerlPredictAndMatch<<<grid, block>>>(uFinalSalientPoints_, cvgmImage_, cvgmSaliency_, cvgmFinalKeyPointsLocations_.ptr<short2>(), cvgmParticlesVelocity_.ptr<short2>(), cvgmParticlesDescriptors_.ptr<int4>(), cvgmParticlesAge_.ptr<uchar>(), cvgmFinalKeyPointsResponse_.ptr<float>() );
    cudaSafeCall( cudaGetLastError() );
    cudaSafeCall( cudaDeviceSynchronize() );

    unsigned int uDeleted ;
    cudaSafeCall( cudaMemcpy(&uDeleted, pCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	return uDeleted;
}

__global__ void kernerlCollectParticles( const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, const unsigned int uTotalParticles_, cv::gpu::DevMem2D_<float> cvgmParticleResponses_){
	const int nKeyPointIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (nKeyPointIdx >= uTotalParticles_) return;

	const short2& s2Loc = ps2KeyPointsLocations_[nKeyPointIdx];
	cvgmParticleResponses_.ptr(s2Loc.y)[s2Loc.x] = pfKeyPointsResponse_[nKeyPointIdx];
}

void cudaCollectParticles(const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, const unsigned int uTotalParticles_, cv::gpu::GpuMat* pcvgmParticleResponses_){
	dim3 block(256);
    dim3 grid;
    grid.x = cv::gpu::divUp(uTotalParticles_, block.x);
	kernerlCollectParticles<<<grid, block>>>( ps2KeyPointsLocations_, pfKeyPointsResponse_, uTotalParticles_, *pcvgmParticleResponses_);
	return;
}

class CPredictAndMatch{
public:
	cv::gpu::DevMem2D_<uchar3> _cvgmBlurredPrev;
	cv::gpu::DevMem2D_<float>  _cvgmParticleResponsesPrev;
	cv::gpu::DevMem2D_<uchar>  _cvgmParticlesAgePrev;
	cv::gpu::DevMem2D_<short2> _cvgmParticlesVelocityPrev;
	
	cv::gpu::DevMem2D_<uchar3> _cvgmBlurredCurr;
	cv::gpu::DevMem2D_<float>  _cvgmParticleResponsesCurr;
	cv::gpu::DevMem2D_<uchar>  _cvgmParticlesAgeCurr;
	cv::gpu::DevMem2D_<short2> _cvgmParticlesVelocityCurr;

	float _fRho;



	__device__ __forceinline__ void operator () (){
		const int c = threadIdx.x + blockIdx.x * blockDim.x + 3;
		const int r = threadIdx.y + blockIdx.y * blockDim.y + 3;

		if( c < 3 || c > _cvgmBlurredPrev.cols - 4 || r < 3 || r > _cvgmBlurredPrev.rows - 4 ) return;
		//if IsParticle( PixelLocation, cvgmParitclesResponse(i) )
		if(_cvgmParticleResponsesPrev.ptr(r)[c] > 0.2f){
			//A) PredictLocation = PixelLocation + ParticleVelocity(i, PixelLocation);
			short2 s2PredictLoc = make_short2(c,r) + _cvgmParticlesVelocityPrev.ptr(r)[c];
			//B) ActualLocation = Match(PredictLocation, cvgmBlurred(i),cvgmBlurred(i+1));
			int4 n4DesPrev;	devGetFastDescriptor(_cvgmBlurredPrev,r,c,&n4DesPrev);
			float fResponse = devMatch( &s2PredictLoc, _cvgmBlurredCurr, _cvgmParticleResponsesCurr, &n4DesPrev );
		
			if( fResponse > 0 ){
				_cvgmParticlesVelocityCurr.ptr(s2PredictLoc.y)[s2PredictLoc.x] = _fRho * (s2PredictLoc - _cvgmParticlesVelocityPrev.ptr(r)[c]) + (1.f - _fRho)* _cvgmParticlesVelocityPrev.ptr(r)[c];//update velocity
				_cvgmParticlesAgeCurr.ptr(s2PredictLoc.y)[s2PredictLoc.x] = _cvgmParticlesAgePrev.ptr(s2PredictLoc.y)[s2PredictLoc.x] + 1; //update age
				_cvgmParticleResponsesCurr.ptr(s2PredictLoc.y)[s2PredictLoc.x] = -fResponse; //update response and location //marked as matched and it will be corrected in NoMaxAndCollection
			}
			else{//C) if no match found 
				atomicInc(&_devuCounter, (unsigned int)(-1));//deleted particle counter increase by 1
			}//lost
		}
		return;
	}
};//class CPredictAndMatch



__global__ void kernelPredictAndMatch(CPredictAndMatch cPAM_){
	cPAM_ ();
}
unsigned int cudaTrack(const cv::gpu::GpuMat& cvgmBlurredPrev_, const cv::gpu::GpuMat& cvgmParticleResponsesPrev_,const cv::gpu::GpuMat& cvgmParticlesAgePrev_,const cv::gpu::GpuMat& cvgmParticlesVelocityPrev_, const cv::gpu::GpuMat& cvgmBlurredCurr_,cv::gpu::GpuMat* pcvgmParticleResponsesCurr_,cv::gpu::GpuMat* pcvgmParticlesAgeCurr_,cv::gpu::GpuMat* pcvgmParticlesVelocityCurr_){
	dim3 block(32,8);
	dim3 grid;
	grid.x = cv::gpu::divUp(cvgmBlurredPrev_.cols - 6, block.x); //6 is the size-1 of the Bresenham circle
    grid.y = cv::gpu::divUp(cvgmBlurredPrev_.rows - 6, block.y);

	CPredictAndMatch cPAM;
	cPAM._cvgmBlurredCurr = cvgmBlurredCurr_;
	cPAM._cvgmBlurredPrev = cvgmBlurredPrev_;
	cPAM._cvgmParticleResponsesPrev = cvgmParticleResponsesPrev_;
	cPAM._cvgmParticlesVelocityPrev = cvgmParticlesVelocityPrev_;
	cPAM._cvgmParticlesAgePrev = cvgmParticlesAgePrev_;

	cPAM._cvgmParticleResponsesCurr = *pcvgmParticleResponsesCurr_;
	cPAM._cvgmParticlesVelocityCurr = *pcvgmParticlesVelocityCurr_;
	cPAM._cvgmParticlesAgeCurr = *pcvgmParticlesAgeCurr_;

	cPAM._fRho = .75f;

	void* pCounter;
    cudaSafeCall( cudaGetSymbolAddress(&pCounter, _devuCounter) );
	cudaSafeCall( cudaMemset(pCounter, 0, sizeof(unsigned int)) );

	kernelPredictAndMatch<<<grid, block>>>(cPAM);
	cudaSafeCall( cudaGetLastError() );
    cudaSafeCall( cudaDeviceSynchronize() );

    unsigned int uDeleted ;
    cudaSafeCall( cudaMemcpy(&uDeleted, pCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	return uDeleted;
}


__device__ unsigned int _devuNewlyAddedCounter = 0;

struct SMatchCollectionAndNonMaxSupression{
	
	cv::gpu::DevMem2D_<float> _cvgmScore;
	
	short2* _ps2KeyPointLocation;
	unsigned int _uTotal;

	unsigned int _uNewlyAddedCount;
	short2* _ps2NewlyAddedKeyPointLocation; 
	float* _pfNewlyAddedKeyPointResponse;
	
	unsigned int _uMatchedCount;
	short2* _ps2MatchedKeyPointLocation; 
	float* _pfMatchedKeyPointResponse;

	__device__ __forceinline__ void operator () (){
		const int nKeyPointIdx = threadIdx.x + blockIdx.x * blockDim.x;
		if (nKeyPointIdx >= _uTotal) return;
		short2 s2Location = _ps2KeyPointLocation[nKeyPointIdx];
		float& fScore = _cvgmScore(s2Location.y, s2Location.x);
		//if the pixel has been identified as matched, store it as the keypoint
		if(fScore < 0){
			const unsigned int nIdx = atomicInc(&_devuCounter, (unsigned int)(-1));
			_ps2MatchedKeyPointLocation[nIdx] = s2Location;
			_pfMatchedKeyPointResponse[nIdx] = -fScore;
			_cvgmScore(s2Location.y, s2Location.x) = -fScore;
			return;
		}
		else{
			//check whether the current corner is the max in 3x3 local area
			bool bIsMax =
				fScore > abs(_cvgmScore(s2Location.y - 1, s2Location.x - 1)) &&
				fScore > abs(_cvgmScore(s2Location.y - 1, s2Location.x    )) &&
				fScore > abs(_cvgmScore(s2Location.y - 1, s2Location.x + 1)) &&

				fScore > abs(_cvgmScore(s2Location.y    , s2Location.x - 1)) &&
				fScore > abs(_cvgmScore(s2Location.y    , s2Location.x + 1)) &&

				fScore > abs(_cvgmScore(s2Location.y + 1, s2Location.x - 1)) &&
				fScore > abs(_cvgmScore(s2Location.y + 1, s2Location.x    )) &&
				fScore > abs(_cvgmScore(s2Location.y + 1, s2Location.x + 1));

			if (bIsMax){
				const unsigned int nIdx = atomicInc(&_devuNewlyAddedCounter , (unsigned int)(-1));
				_ps2NewlyAddedKeyPointLocation[nIdx] = s2Location;
				_pfNewlyAddedKeyPointResponse[nIdx] = fScore;
			}
			else{
				fScore = 0.f;
			}
		}
		return;
	}//operator()
};//SMatchCollectionAndNonMaxSupression



__global__ void kernelMatchCollectionAndNonMaxSupression(SMatchCollectionAndNonMaxSupression sMCNMS_)
{
    sMCNMS_ ();
	return;
}

unsigned int cudaMatchCollectionAndNonMaxSupression(const cv::gpu::GpuMat& cvgmKeyPointLocation_, unsigned int* puMaxSalientPoints_, cv::gpu::GpuMat* pcvgmParticleResponsesCurr_, short2* ps2devMatchedKeyPointLocations_, float* pfdevMatchedKeyPointResponse_, short2* ps2devNewlyAddedKeyPointLocations_, float* pfdevNewlyAddedKeyPointResponse_){
	void* pNewlyAddedCounter,*pMatchCounter;
    cudaSafeCall( cudaGetSymbolAddress(&pMatchCounter, _devuCounter) );
	cudaSafeCall( cudaGetSymbolAddress(&pNewlyAddedCounter, _devuNewlyAddedCounter) );
	cudaSafeCall( cudaMemset(pMatchCounter, 0, sizeof(unsigned int)) );
	cudaSafeCall( cudaMemset(pNewlyAddedCounter, 0, sizeof(unsigned int)) );
    
	dim3 block(256);
    dim3 grid;
    grid.x = cv::gpu::divUp(*puMaxSalientPoints_, block.x);

	SMatchCollectionAndNonMaxSupression sMCNMS;
	sMCNMS._uTotal = *puMaxSalientPoints_;
	sMCNMS._cvgmScore = *pcvgmParticleResponsesCurr_;
	sMCNMS._pfMatchedKeyPointResponse = pfdevMatchedKeyPointResponse_;
	sMCNMS._ps2MatchedKeyPointLocation= ps2devMatchedKeyPointLocations_;
	sMCNMS._pfNewlyAddedKeyPointResponse = pfdevNewlyAddedKeyPointResponse_;
	sMCNMS._ps2NewlyAddedKeyPointLocation= ps2devNewlyAddedKeyPointLocations_;

    kernelMatchCollectionAndNonMaxSupression<<<grid, block>>>(sMCNMS);
    cudaSafeCall( cudaGetLastError() );
    cudaSafeCall( cudaDeviceSynchronize() );

    unsigned int uNewlyAddedCount,uMatchedCount;
    cudaSafeCall( cudaMemcpy(&uNewlyAddedCount, pNewlyAddedCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	cudaSafeCall( cudaMemcpy(&uMatchedCount,    pMatchCounter,      sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	*puMaxSalientPoints_ = uMatchedCount;
    return uNewlyAddedCount;
}

}//semidense
}//device
}//btl
