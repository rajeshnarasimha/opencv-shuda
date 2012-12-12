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
	float fColor1, fColor2;
	float fConMin =300.f; 
	float fC;
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

__global__ void kernelCalcMinDiameterContrast(const cv::gpu::DevMem2D_<uchar3> cvgmImage_, cv::gpu::DevMem2D_<float> cvgmContrast_ ){
	const int c = threadIdx.x + blockIdx.x * blockDim.x + 3;
    const int r = threadIdx.y + blockIdx.y * blockDim.y + 3;

	if( c < 3 || c > cvgmImage_.cols - 4 || r < 3 || r > cvgmImage_.rows - 4 ) return;
	cvgmContrast_.ptr(r)[c] = devCalcMinDiameterContrast(cvgmImage_, r, c );
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
__device__ float devGetFastDescriptor(const cv::gpu::DevMem2D_<uchar3>& cvgmImage_, const int r, const int c, int4* pDescriptor_ ){
	pDescriptor_->x = pDescriptor_->y = pDescriptor_->z = pDescriptor_->w = 0;
	uchar3 Color;
	Color = cvgmImage_.ptr(r-3)[c  ];//1
	pDescriptor_->x += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->x << 8;
	Color = cvgmImage_.ptr(r-3)[c+1];//2
	pDescriptor_->x += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->x << 8;
	Color = cvgmImage_.ptr(r-2)[c+2];//3
	pDescriptor_->x += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->x << 8;
	Color = cvgmImage_.ptr(r-1)[c+3];//4
	pDescriptor_->x += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 


	Color = cvgmImage_.ptr(r  )[c+3];//5
	pDescriptor_->y += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->y << 8;
	Color = cvgmImage_.ptr(r+1)[c+3];//6
	pDescriptor_->y += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->y << 8;
	Color = cvgmImage_.ptr(r+2)[c+2];//7
	pDescriptor_->y += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->y << 8;
	Color = cvgmImage_.ptr(r+3)[c+1];//8
	pDescriptor_->y += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 

	Color = cvgmImage_.ptr(r+3)[c  ];//9
	pDescriptor_->z += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->z << 8;
	Color= cvgmImage_.ptr(r+3)[c-1];//10
	pDescriptor_->z += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->z << 8;
	Color= cvgmImage_.ptr(r+2)[c-2];//11
	pDescriptor_->z += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->z << 8;
	Color= cvgmImage_.ptr(r+1)[c-3];//12
	pDescriptor_->z += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	
	Color= cvgmImage_.ptr(r  )[c-3];//13
	pDescriptor_->w += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->w << 8;
	Color= cvgmImage_.ptr(r-1)[c-3];//14
	pDescriptor_->w += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->w << 8;
	Color= cvgmImage_.ptr(r-2)[c-2];//15
	pDescriptor_->w += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->w << 8;
	Color= cvgmImage_.ptr(r-3)[c-1];//16
	pDescriptor_->w += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	
}

///////////////////////////////////////////////////////////////////////////
// kernelNonMaxSupression
// supress all other corners in 3x3 area only keep the strongest corner
//
__global__ void kernelNonMaxSupression(const cv::gpu::DevMem2D_<uchar3> cvgmImage_, const short2* ps2KeyPointLoc_,const int nCount_, const cv::gpu::PtrStepSzi cvgmScore_, short2* ps2LocFinal_, float* pfResponseFinal_)
{
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 110)

    const int nKeyPointIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (nKeyPointIdx < nCount_)
    {
        short2 s2Location = ps2KeyPointLoc_[nKeyPointIdx];

        float fScore = cvgmScore_(s2Location.y, s2Location.x);
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

        if (bIsMax)
        {
            const unsigned int nIdx = atomicInc(&_devuCounter, (unsigned int)(-1));
            ps2LocFinal_[nIdx] = s2Location;
            pfResponseFinal_[nIdx] = fScore;
			/*int4 f4Descriptor;
			devGetFastDescriptor(cvgmImage_,s2Location.y,s2Location.x,&f4Descriptor );
			pf4devDescriptor_[nIdx] = f4Descriptor;*/
        }
    }

    #endif
}

unsigned int cudaNonMaxSupression(const cv::gpu::GpuMat& cvgmImage_, const cv::gpu::GpuMat& cvgmKeyPointLocation_, const unsigned int uMaxSalientPoints_, const cv::gpu::GpuMat& cvgmSaliency_, short2* ps2devLocations_, float* pfdevResponse_){
	void* pCounter;
    cudaSafeCall( cudaGetSymbolAddress(&pCounter, _devuCounter) );

    dim3 block(256);
    dim3 grid;
    grid.x = cv::gpu::divUp(uMaxSalientPoints_, block.x);

    cudaSafeCall( cudaMemset(pCounter, 0, sizeof(unsigned int)) );

    kernelNonMaxSupression<<<grid, block>>>(cvgmImage_, cvgmKeyPointLocation_.ptr<short2>(), uMaxSalientPoints_, cvgmSaliency_, ps2devLocations_, pfdevResponse_);
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



}//semidense
}//device
}//btl
