#include <thrust/sort.h>

#include <opencv2/gpu/gpumat.hpp>
#include <opencv2/gpu/device/common.hpp>
#include <opencv2/gpu/device/utility.hpp>
#include <opencv2/gpu/device/functional.hpp>

#include "CudaHelper.hpp"
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
__device__ void devUpdateMaxConstrast(const uchar& Color_, const float& fCenter_, float* pfConMax_ ){
	float fC = abs(fCenter_ - Color_ );
	*pfConMax_ = *pfConMax_ > fC? *pfConMax_ :fC;
}
__device__ float devCalcMaxContrast(const cv::gpu::DevMem2D_<uchar>& cvgmImage_, const int r, const int c ){
	const uchar& Center = cvgmImage_.ptr(r)[c];
	float fCenter = Center;
	float fConMax = -1.f; 
	uchar Color;

	Color = cvgmImage_.ptr(r-3)[c  ];//1
	devUpdateMaxConstrast(Color, fCenter, &fConMax);

	Color = cvgmImage_.ptr(r-3)[c+1];//2
	devUpdateMaxConstrast(Color, fCenter, &fConMax);
	
	Color = cvgmImage_.ptr(r-2)[c+2];//3
	devUpdateMaxConstrast(Color, fCenter, &fConMax);
	
	Color = cvgmImage_.ptr(r-1)[c+3];//4
	devUpdateMaxConstrast(Color, fCenter, &fConMax);

	Color = cvgmImage_.ptr(r  )[c+3];//5
	devUpdateMaxConstrast(Color, fCenter, &fConMax);

	Color = cvgmImage_.ptr(r+1)[c+3];//6
	devUpdateMaxConstrast(Color, fCenter, &fConMax);

	Color = cvgmImage_.ptr(r+2)[c+2];//7
	devUpdateMaxConstrast(Color, fCenter, &fConMax);

	Color = cvgmImage_.ptr(r+3)[c+1];//8
	devUpdateMaxConstrast(Color, fCenter, &fConMax);

	Color = cvgmImage_.ptr(r+3)[c  ];//9
	devUpdateMaxConstrast(Color, fCenter, &fConMax);

	Color= cvgmImage_.ptr(r+3)[c-1];//10
	devUpdateMaxConstrast(Color, fCenter, &fConMax);

	Color= cvgmImage_.ptr(r+2)[c-2];//11
	devUpdateMaxConstrast(Color, fCenter, &fConMax);

	Color= cvgmImage_.ptr(r+1)[c-3];//12
	devUpdateMaxConstrast(Color, fCenter, &fConMax);

	Color= cvgmImage_.ptr(r  )[c-3];//13
	devUpdateMaxConstrast(Color, fCenter, &fConMax);

	Color= cvgmImage_.ptr(r-1)[c-3];//14
	devUpdateMaxConstrast(Color, fCenter, &fConMax);

	Color= cvgmImage_.ptr(r-2)[c-2];//15
	devUpdateMaxConstrast(Color, fCenter, &fConMax);

	Color= cvgmImage_.ptr(r-3)[c-1];//16
	devUpdateMaxConstrast(Color, fCenter, &fConMax);
	
	return fConMax;
}
/*
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
}*/

__global__ void kernelCalcMaxContrast(const cv::gpu::DevMem2D_<uchar> cvgmImage_, const unsigned char ucContrastThreshold_, cv::gpu::DevMem2D_<float> cvgmContrast_ ){
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
__device__ void devUpdateMinContrast( const uchar& ucColor1_, const uchar& ucColor2_, const float& fCenter_, float* pfMinContrast_){
	float fC = .5f * abs( 2.f * fCenter_ - ucColor1_  - ucColor2_);
	*pfMinContrast_ = *pfMinContrast_ < fC? *pfMinContrast_:fC;
}
__device__ float devCalcMinDiameterContrast(const cv::gpu::DevMem2D_<uchar>& cvgmImage_, int r, int c){
	const uchar& Center = cvgmImage_.ptr(r)[c];
	float fCenter = Center;
	//float fColor1, fColor2;
	float fConMin =300.f; 
	//float fC;
	uchar Color1, Color2;

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
__device__ void devUpdateMinContrastColor( const uchar3& uc3Color1_, const uchar3& uc3Color2_, const uchar3& uc3Center_, float* pfMinContrast_){
	float fM = -1.f;
	float fC;
	fC = .5f * abs( 2.f * uc3Center_.x - uc3Color1_.x - uc3Color2_.x );
	fM = fM > fC ? fM : fC;
	fC = .5f * abs( 2.f * uc3Center_.y - uc3Color1_.y - uc3Color2_.y );
	fM = fM > fC ? fM : fC;
	fC = .5f * abs( 2.f * uc3Center_.z - uc3Color1_.z - uc3Color2_.z );
	fM = fM > fC ? fM : fC;
	*pfMinContrast_ = *pfMinContrast_ < fM? *pfMinContrast_:fM;
	return;
}
__device__ float devCalcMinDiameterContrast2(const cv::gpu::DevMem2D_<uchar3>& cvgmImage_, int r, int c){
	const uchar3& Center = cvgmImage_.ptr(r)[c];
	//float fColor1, fColor2;
	float fConMin =300.f; 
	//float fC;
	uchar3 Color1, Color2;

	Color1 = cvgmImage_.ptr(r-3)[c  ];//1
	Color2 = cvgmImage_.ptr(r+3)[c  ];//9
	devUpdateMinContrastColor( Color1, Color2, Center, &fConMin );
	
	Color1 = cvgmImage_.ptr(r-3)[c+1];//2
	Color2 = cvgmImage_.ptr(r+3)[c-1];//10
	devUpdateMinContrastColor( Color1, Color2, Center, &fConMin );

	Color1 = cvgmImage_.ptr(r-2)[c+2];//3
	Color2 = cvgmImage_.ptr(r+2)[c-2];//11
	devUpdateMinContrastColor( Color1, Color2, Center, &fConMin );

	Color1 = cvgmImage_.ptr(r-1)[c+3];//4
	Color2 = cvgmImage_.ptr(r+1)[c-3];//12
	devUpdateMinContrastColor( Color1, Color2, Center, &fConMin );

	Color1 = cvgmImage_.ptr(r  )[c+3];//5
	Color2 = cvgmImage_.ptr(r  )[c-3];//13
	devUpdateMinContrastColor( Color1, Color2, Center, &fConMin );

	Color1 = cvgmImage_.ptr(r+1)[c+3];//6
	Color2 = cvgmImage_.ptr(r-1)[c-3];//14
	devUpdateMinContrastColor( Color1, Color2, Center, &fConMin );

	Color1 = cvgmImage_.ptr(r+2)[c+2];//7
	Color2 = cvgmImage_.ptr(r-2)[c-2];//15
	devUpdateMinContrastColor( Color1, Color2, Center, &fConMin );

	Color1 = cvgmImage_.ptr(r+3)[c+1];//8
	Color2 = cvgmImage_.ptr(r-3)[c-1];//16
	devUpdateMinContrastColor( Color1, Color2, Center, &fConMin );

	return fConMin;
}

__global__ void kernelCalcMinDiameterContrast(const cv::gpu::DevMem2D_<uchar> cvgmImage_, cv::gpu::DevMem2D_<float> cvgmContrast_ ){
	const int c = threadIdx.x + blockIdx.x * blockDim.x;
    const int r = threadIdx.y + blockIdx.y * blockDim.y;
	if( c < 0 || c >= cvgmImage_.cols || r < 0 || r >= cvgmImage_.rows ) return; //falling out the image
	if( c < 3 || c > cvgmImage_.cols - 4 || r < 3 || r > cvgmImage_.rows - 4 ) { cvgmContrast_.ptr(r)[c] = 0.f; return;} // brim

	cvgmContrast_.ptr(r)[c] = devCalcMinDiameterContrast(cvgmImage_, r, c );  //effective domain
}

void cudaCalcMinDiameterContrast(const cv::gpu::GpuMat& cvgmImage_, cv::gpu::GpuMat* pcvgmContrast_){
	dim3 block(32, 8);

    dim3 grid;
    grid.x = cv::gpu::divUp(cvgmImage_.cols - 6, block.x); //6 is the size-1 of the Bresenham circle
    grid.y = cv::gpu::divUp(cvgmImage_.rows - 6, block.y);

	kernelCalcMinDiameterContrast<<<grid, block>>>(cvgmImage_, *pcvgmContrast_);
}



__global__ void kernelCalcSaliency(const cv::gpu::DevMem2D_<uchar> cvgmImage_, const unsigned short usHalfSizeRound_, 
								   const unsigned char ucContrastThreshold_, const float fSaliencyThreshold_, 
								   cv::gpu::DevMem2D_<float> cvgmSaliency_, cv::gpu::DevMem2D_<short2> cvgmKeyPointLocations_){
	const int c = threadIdx.x + blockIdx.x * blockDim.x;
    const int r = threadIdx.y + blockIdx.y * blockDim.y;

	if( c < 0 || c >= cvgmImage_.cols || r < 0 || r >= cvgmImage_.rows ) return; //falling out the image
	float& fSaliency = cvgmSaliency_.ptr(r)[c]; 
	fSaliency = 0.f;

	if( c < usHalfSizeRound_ || c >= cvgmImage_.cols - usHalfSizeRound_ || r < usHalfSizeRound_ || r >= cvgmImage_.rows - usHalfSizeRound_ ) return; 

	//calc saliency scores
	float fMaxContrast = devCalcMaxContrast(cvgmImage_, r, c );
	if(fMaxContrast <= ucContrastThreshold_) return;
	fSaliency = devCalcMinDiameterContrast(cvgmImage_, r, c )/fMaxContrast;
	if (fSaliency < fSaliencyThreshold_) { fSaliency = 0.f; return; } //if lower than the saliency threshold
																	  //the saliency score is truncated into 0.f;
	//record the location of the pixel where the saliency is above the threshold
	const unsigned int nIdx = atomicInc(&_devuCounter, (unsigned int)(-1));
    if (nIdx < cvgmKeyPointLocations_.cols)
		cvgmKeyPointLocations_.ptr(0)[nIdx] = make_short2(c, r);
	return;
}

//return the No. of Salient pixels above fSaliencyThreshold_
unsigned int cudaCalcSaliency(const cv::gpu::GpuMat& cvgmImage_, const unsigned short usHalfSizeRound_,
							  const unsigned char ucContrastThreshold_, const float& fSaliencyThreshold_, 
							  cv::gpu::GpuMat* pcvgmSaliency_, cv::gpu::GpuMat* pcvgmKeyPointLocations_){
	void* pCounter;
    cudaSafeCall( cudaGetSymbolAddress(&pCounter, _devuCounter) );

	dim3 block(32, 8);
    dim3 grid;
    grid.x = cv::gpu::divUp(cvgmImage_.cols - 6, block.x); //6 is the size-1 of the Bresenham circle
    grid.y = cv::gpu::divUp(cvgmImage_.rows - 6, block.y);

	kernelCalcSaliency<<<grid, block>>>(cvgmImage_, usHalfSizeRound_, ucContrastThreshold_, fSaliencyThreshold_, *pcvgmSaliency_, *pcvgmKeyPointLocations_);
	cudaSafeCall( cudaGetLastError() );
    cudaSafeCall( cudaDeviceSynchronize() );

    unsigned int uCount;
    cudaSafeCall( cudaMemcpy(&uCount, pCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

    return uCount;
}


__device__ void devGetFastDescriptor(const cv::gpu::DevMem2D_<uchar>& cvgmImage_, const int r, const int c, int4* pDescriptor_ ){
	pDescriptor_->x = pDescriptor_->y = pDescriptor_->z = pDescriptor_->w = 0;
	uchar Color;
	Color = cvgmImage_.ptr(r-3)[c  ];//1
	pDescriptor_->x += Color; 
	pDescriptor_->x = pDescriptor_->x << 8;
	Color = cvgmImage_.ptr(r-6)[c+2];//2 B6
	pDescriptor_->x += Color; 
	pDescriptor_->x = pDescriptor_->x << 8;
	Color = cvgmImage_.ptr(r-2)[c+2];//3
	pDescriptor_->x += Color; 
	pDescriptor_->x = pDescriptor_->x << 8;
	Color = cvgmImage_.ptr(r-2)[c+6];//4 B6
	pDescriptor_->x += Color; 


	Color = cvgmImage_.ptr(r  )[c+3];//5
	pDescriptor_->y += Color; 
	pDescriptor_->y = pDescriptor_->y << 8;
	Color = cvgmImage_.ptr(r+2)[c+6];//6 B6
	pDescriptor_->y += Color; 
	pDescriptor_->y = pDescriptor_->y << 8;
	Color = cvgmImage_.ptr(r+2)[c+2];//7
	pDescriptor_->y += Color; 
	pDescriptor_->y = pDescriptor_->y << 8;
	Color = cvgmImage_.ptr(r+6)[c+2];//8 B6
	pDescriptor_->y += Color; 

	Color = cvgmImage_.ptr(r+3)[c  ];//9
	pDescriptor_->z += Color; 
	pDescriptor_->z = pDescriptor_->z << 8;
	Color= cvgmImage_.ptr(r+6)[c-2];//10 B6
	pDescriptor_->z += Color; 
	pDescriptor_->z = pDescriptor_->z << 8;
	Color= cvgmImage_.ptr(r+2)[c-2];//11
	pDescriptor_->z += Color; 
	pDescriptor_->z = pDescriptor_->z << 8;
	Color= cvgmImage_.ptr(r+2)[c-6];//12 B6
	pDescriptor_->z += Color; 
	
	Color= cvgmImage_.ptr(r  )[c-3];//13
	pDescriptor_->w += Color; 
	pDescriptor_->w = pDescriptor_->w << 8;
	Color= cvgmImage_.ptr(r-2)[c-6];//14 B6
	pDescriptor_->w += Color; 
	pDescriptor_->w = pDescriptor_->w << 8;
	Color= cvgmImage_.ptr(r-2)[c-2];//15
	pDescriptor_->w += Color; 
	pDescriptor_->w = pDescriptor_->w << 8;
	Color= cvgmImage_.ptr(r-6)[c-2];//16 B6
	pDescriptor_->w += Color; 
	return;
}
//single ring
__device__ void devGetFastDescriptor1(const cv::gpu::DevMem2D_<uchar>& cvgmImage_, const int r, const int c, int4* pDescriptor_ ){
	pDescriptor_->x = pDescriptor_->y = pDescriptor_->z = pDescriptor_->w = 0;
	uchar Color;
	Color = cvgmImage_.ptr(r-3)[c  ];//1
	pDescriptor_->x += Color; 
	pDescriptor_->x = pDescriptor_->x << 8;
	Color = cvgmImage_.ptr(r-3)[c+1];//2
	pDescriptor_->x += Color; 
	pDescriptor_->x = pDescriptor_->x << 8;
	Color = cvgmImage_.ptr(r-2)[c+2];//3
	pDescriptor_->x += Color; 
	pDescriptor_->x = pDescriptor_->x << 8;
	Color = cvgmImage_.ptr(r-1)[c+3];//4
	pDescriptor_->x += Color; 


	Color = cvgmImage_.ptr(r  )[c+3];//5
	pDescriptor_->y += Color; 
	pDescriptor_->y = pDescriptor_->y << 8;
	Color = cvgmImage_.ptr(r+1)[c+3];//6
	pDescriptor_->y += Color; 
	pDescriptor_->y = pDescriptor_->y << 8;
	Color = cvgmImage_.ptr(r+2)[c+2];//7
	pDescriptor_->y += Color; 
	pDescriptor_->y = pDescriptor_->y << 8;
	Color = cvgmImage_.ptr(r+3)[c+1];//8
	pDescriptor_->y += Color; 

	Color = cvgmImage_.ptr(r+3)[c  ];//9
	pDescriptor_->z += Color; 
	pDescriptor_->z = pDescriptor_->z << 8;
	Color= cvgmImage_.ptr(r+3)[c-1];//10
	pDescriptor_->z += Color; 
	pDescriptor_->z = pDescriptor_->z << 8;
	Color= cvgmImage_.ptr(r+2)[c-2];//11
	pDescriptor_->z += Color; 
	pDescriptor_->z = pDescriptor_->z << 8;
	Color= cvgmImage_.ptr(r+1)[c-3];//12
	pDescriptor_->z += Color; 
	
	Color= cvgmImage_.ptr(r  )[c-3];//13
	pDescriptor_->w += Color; 
	pDescriptor_->w = pDescriptor_->w << 8;
	Color= cvgmImage_.ptr(r-1)[c-3];//14
	pDescriptor_->w += Color; 
	pDescriptor_->w = pDescriptor_->w << 8;
	Color= cvgmImage_.ptr(r-2)[c-2];//15
	pDescriptor_->w += Color; 
	pDescriptor_->w = pDescriptor_->w << 8;
	Color= cvgmImage_.ptr(r-3)[c-1];//16
	pDescriptor_->w += Color; 
	return;
}

///////////////////////////////////////////////////////////////////////////
// kernelNonMaxSupression
// supress all other corners in 3x3 area only keep the strongest corner
//
__global__ void kernelNonMaxSupression(const short2* ps2KeyPointLoc_,const int nCount_,const cv::gpu::PtrStepSzf cvgmSaliency_, short2* ps2LocFinal_, float* pfResponseFinal_)
{
    const int nKeyPointIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (nKeyPointIdx >= nCount_) return;
    short2 s2Location = ps2KeyPointLoc_[nKeyPointIdx];
	if( s2Location.x < 1 || s2Location.x >= cvgmSaliency_.cols - 1 || s2Location.y < 1 || s2Location.y >= cvgmSaliency_.rows - 1 ) return;
    const float& fScore = cvgmSaliency_(s2Location.y, s2Location.x);
	//check whether the current corner is the max in 3x3 local area
    bool bIsMax =
        fScore > cvgmSaliency_(s2Location.y - 1, s2Location.x - 1) &&
        fScore > cvgmSaliency_(s2Location.y - 1, s2Location.x    ) &&
        fScore > cvgmSaliency_(s2Location.y - 1, s2Location.x + 1) &&

        fScore > cvgmSaliency_(s2Location.y    , s2Location.x - 1) &&
        fScore > cvgmSaliency_(s2Location.y    , s2Location.x + 1) &&

        fScore > cvgmSaliency_(s2Location.y + 1, s2Location.x - 1) &&
        fScore > cvgmSaliency_(s2Location.y + 1, s2Location.x    ) &&
        fScore > cvgmSaliency_(s2Location.y + 1, s2Location.x + 1);

    if (bIsMax){
        const unsigned int nIdx = atomicInc(&_devuCounter, (unsigned int)(-1));
        ps2LocFinal_[nIdx] = s2Location;
        pfResponseFinal_[nIdx] = fScore;
    }
	/*else{
		fScore = 0.f;
	}*/
	return;
}
/*
input:
 cvgmKeyPointLocation_: 1 row array of key point (salient point) locations
 uMaxSalientPoints_: the total # of salient points 
 pcvgmSaliency_: store the frame of saliency score
returned values
 ps2devLocations_: store the non-max supressed key point (salient point) locations
 pfdevResponse_: store the non-max supressed key point (sailent point) strength score
*/
unsigned int cudaNonMaxSupression(const cv::gpu::GpuMat& cvgmKeyPointLocation_, const unsigned int uMaxSalientPoints_, 
	const cv::gpu::GpuMat& cvgmSaliency_, short2* ps2devLocations_, float* pfdevResponse_){
	void* pCounter;
    cudaSafeCall( cudaGetSymbolAddress(&pCounter, _devuCounter) );

    dim3 block(256);
    dim3 grid;
    grid.x = cv::gpu::divUp(uMaxSalientPoints_, block.x);

    cudaSafeCall( cudaMemset(pCounter, 0, sizeof(unsigned int)) );

    kernelNonMaxSupression<<<grid, block>>>(cvgmKeyPointLocation_.ptr<short2>(), uMaxSalientPoints_, cvgmSaliency_, ps2devLocations_, pfdevResponse_);
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

__global__ void kernelFastDescriptors(cv::gpu::DevMem2D_<uchar> cvgmImage_, const short2* ps2KeyPointLoc_, const unsigned int uMaxSailentPoints_, int4* pn4devDescriptor_ )
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








__global__ void kernerlCollectParticles( const cv::gpu::DevMem2D_<uchar> cvgmImage_,const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, const unsigned int uTotalParticles_, 
	cv::gpu::DevMem2D_<float> cvgmParticleResponses_, cv::gpu::DevMem2D_<int4> cvgmParticleDescriptors_){
	const int nKeyPointIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (nKeyPointIdx >= uTotalParticles_) return;

	const short2& s2Loc = ps2KeyPointsLocations_[nKeyPointIdx];
	if( s2Loc.x < 8 || s2Loc.x >= cvgmImage_.cols - 8 || s2Loc.y < 8 || s2Loc.y >= cvgmImage_.rows - 8 ) return;

	cvgmParticleResponses_.ptr(s2Loc.y)[s2Loc.x] = pfKeyPointsResponse_[nKeyPointIdx];
	int4 n4Desc;
	devGetFastDescriptor(cvgmImage_,s2Loc.y,s2Loc.x,&n4Desc);
	cvgmParticleDescriptors_.ptr(s2Loc.y)[s2Loc.x] = n4Desc;
}

__global__ void kernerlCollectParticles( const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, const unsigned int uTotalParticles_, 
	cv::gpu::DevMem2D_<float> cvgmParticleResponses_){
	const int nKeyPointIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (nKeyPointIdx >= uTotalParticles_) return;

	const short2& s2Loc = ps2KeyPointsLocations_[nKeyPointIdx];
	cvgmParticleResponses_.ptr(s2Loc.y)[s2Loc.x] = pfKeyPointsResponse_[nKeyPointIdx];
}
/*
collect all key points and key point response and set a frame of saliency frame
input values:
  ps2KeyPointsLocations_: 
returned values:
  pcvgmParticleResponses_: a frame of saliency response
*/
void cudaCollectParticles(const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, const unsigned int uTotalParticles_, 
	cv::gpu::GpuMat* pcvgmParticleResponses_, cv::gpu::GpuMat* pcvgmParticleDescriptor_, const cv::gpu::GpuMat& cvgmImage_/*=cv::gpu::GpuMat()*/ ){
	if(uTotalParticles_ == 0) return;
	dim3 block(256);
    dim3 grid;
    grid.x = cv::gpu::divUp(uTotalParticles_, block.x);
	if(pcvgmParticleDescriptor_)
		kernerlCollectParticles<<<grid, block>>>( cvgmImage_, ps2KeyPointsLocations_, pfKeyPointsResponse_, uTotalParticles_, *pcvgmParticleResponses_, *pcvgmParticleDescriptor_);
	else
		kernerlCollectParticles<<<grid, block>>>( ps2KeyPointsLocations_, pfKeyPointsResponse_, uTotalParticles_, *pcvgmParticleResponses_);
	return;
}

class CPredictAndMatch{
public:
	cv::gpu::DevMem2D_<int4>   _cvgmParticleDescriptorsPrev;
	cv::gpu::DevMem2D_<float>  _cvgmParticleResponsesPrev;
	cv::gpu::DevMem2D_<uchar>  _cvgmParticlesAgePrev;
	cv::gpu::DevMem2D_<short2> _cvgmParticlesVelocityPrev;
	
	cv::gpu::DevMem2D_<uchar>  _cvgmImageCurr;
	cv::gpu::DevMem2D_<int4>   _cvgmParticleDescriptorsCurr;
	cv::gpu::DevMem2D_<float>  _cvgmParticleResponsesCurr;
	cv::gpu::DevMem2D_<uchar>  _cvgmParticlesAgeCurr;
	cv::gpu::DevMem2D_<short2> _cvgmParticlesVelocityCurr;

	float _fRho;

	float _fMatchThreshold;
	short _sSearchRange;
	unsigned short _usHalfSizeRound;
/*calc the distance of two descriptors, the distance is ranged from 0. to 255.
*/
__device__ float dL1(const int4& n4Descriptor1_, const int4& n4Descriptor2_){
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
	fDist /= 16.f;
	return fDist;
}
/*search in a n x n (search area) area round ps2Loc_ in current frame for the most similar descriptor
  Input: 
	1.fMatchThreshold_: the difference of two descriptors
	2.sSearchRange_: the radius of the searching area
	3.n4DesPrev_: the descriptor of the previous frame
	4.ps2Loc_: the location of predicted position in current frame
  Output:
	1.ps2Loc_: the location of the best matched descriptor in current frame with previous frame descriptor
	2.pn4DesCurr_: the descriptor of the best matched point in current frame
*/
	__device__ __forceinline__ float devMatch(const float& fMatchThreshold_, const short sSearchRange_,
		const int4& n4DesPrev_, short2* ps2Loc_, int4 *pn4DesCurr_){
		float fResponse = 0.f;
		float fBestMatchedResponse;
		short2 s2Loc,s2BestLoc;
		int4 n4BestDescriptor;
		float fMinDist = 300.f;
		//search for the 7x7 neighbourhood for 
		for(short r = -sSearchRange_; r <= sSearchRange_; r++ ){
			for(short c = -sSearchRange_; c <= sSearchRange_; c++ ){
				s2Loc = *ps2Loc_ + make_short2( c, r ); 
				if(s2Loc.x < _usHalfSizeRound || s2Loc.x >= _cvgmImageCurr.cols - _usHalfSizeRound || s2Loc.y < _usHalfSizeRound || s2Loc.y >= _cvgmImageCurr.rows - _usHalfSizeRound ) continue;
				fResponse = _cvgmParticleResponsesCurr.ptr(s2Loc.y)[s2Loc.x];
				if( fResponse > 0.1f ){
					int4 n4Des; 
					devGetFastDescriptor(_cvgmImageCurr,s2Loc.y,s2Loc.x,&n4Des);
					float fDist = dL1(n4Des,n4DesPrev_);
					if ( fDist < fMatchThreshold_ ){
						if (  fMinDist > fDist ){
							fMinDist = fDist;
							fBestMatchedResponse = fResponse;
							s2BestLoc = s2Loc;
							n4BestDescriptor = n4Des;
						}
					}
				}//if sailent corner exits
			}//for 
		}//for
		if(fMinDist < 300.f){
			*ps2Loc_ = s2BestLoc;
			*pn4DesCurr_ = n4BestDescriptor;
			return fBestMatchedResponse;
		}
		else{
			return -1.f;
		}
	}//devMatch

	__device__ __forceinline__ void operator () (){
		const int c = threadIdx.x + blockIdx.x * blockDim.x;
		const int r = threadIdx.y + blockIdx.y * blockDim.y;

		if( c < 3 || c >= _cvgmImageCurr.cols - 4 || r < 3 || r >= _cvgmImageCurr.rows - 4 ) return;

		//if IsParticle( PixelLocation, cvgmParitclesResponse(i) )
		if(_cvgmParticleResponsesPrev.ptr(r)[c] < 0.2f) return;
		//A) PredictLocation = PixelLocation + ParticleVelocity(i, PixelLocation);
		short2 s2PredictLoc = make_short2(c,r);// + _cvgmParticlesVelocityPrev.ptr(r)[c];
		//B) ActualLocation = Match(PredictLocation, cvgmBlurred(i),cvgmBlurred(i+1));
		
		//;	devGetFastDescriptor(_cvgmBlurredPrev,r,c,&n4DesPrev);
		const int4& n4DesPrev = _cvgmParticleDescriptorsPrev.ptr(r)[c];
		int4 n4DesCur;
		float fResponse = devMatch( _fMatchThreshold, _sSearchRange, n4DesPrev, &s2PredictLoc, &n4DesCur );
		
		if( fResponse > 0.1f ){
			atomicInc(&_devuNewlyAddedCounter, (unsigned int)(-1));//deleted particle counter increase by 1

			_cvgmParticleDescriptorsCurr.ptr(s2PredictLoc.y)[s2PredictLoc.x]=n4DesCur;
			_cvgmParticlesVelocityCurr.ptr(s2PredictLoc.y)[s2PredictLoc.x] = _fRho * (s2PredictLoc - make_short2(c,r)) + (1.f - _fRho)* _cvgmParticlesVelocityPrev.ptr(r)[c];//update velocity
			_cvgmParticlesAgeCurr.ptr     (s2PredictLoc.y)[s2PredictLoc.x] = _cvgmParticlesAgePrev.ptr(r)[c] + 1; //update age
			_cvgmParticleResponsesCurr.ptr(s2PredictLoc.y)[s2PredictLoc.x] = -fResponse; //update response and location //marked as matched and it will be corrected in NoMaxAndCollection
		}
		else{//C) if no match found 
			_cvgmParticlesVelocityPrev.ptr(r)[c] = make_short2(0,0);
			atomicInc(&_devuCounter, (unsigned int)(-1));//deleted particle counter increase by 1
		}//lost
		
		return;
	}
};//class CPredictAndMatch



__global__ void kernelPredictAndMatch(CPredictAndMatch cPAM_){
	cPAM_ ();
}
unsigned int cudaTrack(float fMatchThreshold_, const short sSearchRange_, 
	const cv::gpu::GpuMat& cvgmParticleDescriptorsPrev_, const cv::gpu::GpuMat& cvgmParticleResponsesPrev_,const cv::gpu::GpuMat& cvgmParticlesAgePrev_,
	const cv::gpu::GpuMat& cvgmParticlesVelocityPrev_, const cv::gpu::GpuMat& cvgmBlurredCurr_,
	cv::gpu::GpuMat* pcvgmParticleResponsesCurr_,cv::gpu::GpuMat* pcvgmParticlesAgeCurr_,cv::gpu::GpuMat* pcvgmParticlesVelocityCurr_,cv::gpu::GpuMat* pcvgmParticleDescriptorsCurr_){
	dim3 block(32,8);
	dim3 grid;
	grid.x = cv::gpu::divUp(cvgmBlurredCurr_.cols - 6, block.x); //6 is the size-1 of the Bresenham circle
    grid.y = cv::gpu::divUp(cvgmBlurredCurr_.rows - 6, block.y);

	CPredictAndMatch cPAM;
	cPAM._cvgmImageCurr = cvgmBlurredCurr_;
	cPAM._cvgmParticleDescriptorsPrev = cvgmParticleDescriptorsPrev_;
	cPAM._cvgmParticleResponsesPrev = cvgmParticleResponsesPrev_;
	cPAM._cvgmParticlesVelocityPrev = cvgmParticlesVelocityPrev_;
	cPAM._cvgmParticlesAgePrev = cvgmParticlesAgePrev_;

	cPAM._cvgmParticleDescriptorsCurr = *pcvgmParticleDescriptorsCurr_;
	cPAM._cvgmParticleResponsesCurr = *pcvgmParticleResponsesCurr_;
	cPAM._cvgmParticlesVelocityCurr = *pcvgmParticlesVelocityCurr_;
	cPAM._cvgmParticlesAgeCurr = *pcvgmParticlesAgeCurr_;

	cPAM._fRho = .75f;
	cPAM._fMatchThreshold = fMatchThreshold_;
	cPAM._sSearchRange = sSearchRange_;
	cPAM._usHalfSizeRound = 7;//the half size of fast descriptor is 6, but it doesnot rotate, therefore 7 is enough to avoid memory voilation

	void* pCounter;
    cudaSafeCall( cudaGetSymbolAddress(&pCounter, _devuCounter) );
	cudaSafeCall( cudaMemset(pCounter, 0, sizeof(unsigned int)) );

	void* pCounterMatch;
    cudaSafeCall( cudaGetSymbolAddress(&pCounterMatch, _devuNewlyAddedCounter) );
	cudaSafeCall( cudaMemset(pCounterMatch, 0, sizeof(unsigned int)) );

	kernelPredictAndMatch<<<grid, block>>>(cPAM);
	cudaSafeCall( cudaGetLastError() );
    cudaSafeCall( cudaDeviceSynchronize() );

    unsigned int uDeleted ;
    cudaSafeCall( cudaMemcpy(&uDeleted, pCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	unsigned int uMatched ;
    cudaSafeCall( cudaMemcpy(&uMatched, pCounterMatch, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	return uDeleted;
}



struct SMatchedAndNewlyAddedKeyPointsCollection{
	
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
		if(fScore < 0.f){
			const unsigned int nIdx = atomicInc(&_devuCounter, (unsigned int)(-1));
			_ps2MatchedKeyPointLocation[nIdx] = s2Location;
			_pfMatchedKeyPointResponse[nIdx] = -fScore;
			_cvgmScore(s2Location.y, s2Location.x) = -fScore; 
		}
		else if(fScore > 0.0001f){
			const unsigned int nIdx = atomicInc(&_devuNewlyAddedCounter , (unsigned int)(-1));
			_ps2NewlyAddedKeyPointLocation[nIdx] = s2Location;
			_pfNewlyAddedKeyPointResponse[nIdx] = fScore;
		}
		return;
	}//operator()
};//SMatchCollectionAndNonMaxSupression

__global__ void kernelMatchedAndNewlyAddedKeyPointsCollection(SMatchedAndNewlyAddedKeyPointsCollection sMCNMS_){
    sMCNMS_ ();
	return;
}
//after track, all key points in current frame are collected into 1.matched key point group 2.newly added key point group
unsigned int cudaMatchedAndNewlyAddedKeyPointsCollection(cv::gpu::GpuMat& cvgmKeyPointLocation_, 
	unsigned int* puMaxSalientPoints_, cv::gpu::GpuMat* pcvgmParticleResponsesCurr_, short2* ps2devMatchedKeyPointLocations_, 
	float* pfdevMatchedKeyPointResponse_, short2* ps2devNewlyAddedKeyPointLocations_, float* pfdevNewlyAddedKeyPointResponse_){
	void* pNewlyAddedCounter,*pMatchCounter;
    cudaSafeCall( cudaGetSymbolAddress(&pMatchCounter, _devuCounter) );
	cudaSafeCall( cudaGetSymbolAddress(&pNewlyAddedCounter, _devuNewlyAddedCounter) );
	cudaSafeCall( cudaMemset(pMatchCounter, 0, sizeof(unsigned int)) );
	cudaSafeCall( cudaMemset(pNewlyAddedCounter, 0, sizeof(unsigned int)) );
    
	dim3 block(256);
    dim3 grid;
    grid.x = cv::gpu::divUp(*puMaxSalientPoints_, block.x);

	SMatchedAndNewlyAddedKeyPointsCollection sMCNMS;
	sMCNMS._ps2KeyPointLocation = cvgmKeyPointLocation_.ptr<short2>();
	sMCNMS._uTotal = *puMaxSalientPoints_;
	sMCNMS._cvgmScore = *pcvgmParticleResponsesCurr_;
	sMCNMS._pfMatchedKeyPointResponse = pfdevMatchedKeyPointResponse_;
	sMCNMS._ps2MatchedKeyPointLocation= ps2devMatchedKeyPointLocations_;
	sMCNMS._pfNewlyAddedKeyPointResponse = pfdevNewlyAddedKeyPointResponse_;
	sMCNMS._ps2NewlyAddedKeyPointLocation= ps2devNewlyAddedKeyPointLocations_;

    kernelMatchedAndNewlyAddedKeyPointsCollection<<<grid, block>>>(sMCNMS);
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
