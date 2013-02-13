#include <thrust/sort.h>

#include <opencv2/gpu/gpumat.hpp>
#include <opencv2/gpu/device/common.hpp>
#include <opencv2/gpu/device/utility.hpp>
#include <opencv2/gpu/device/functional.hpp>

#include "CudaHelper.hpp"
//#include "pcl/vector_math.hpp"
//using namespace pcl::device;

#define GRAY
__device__ unsigned int _devuCounter = 0;

__device__ unsigned int _devuNewlyAddedCounter = 0;

__device__ unsigned int _devuOther = 0;

__device__ unsigned int _devuTest1 = 0;


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
    cudaSafeCall( cudaMemset(pCounter, 0, sizeof(unsigned int)) );
	
	dim3 block(32, 8);
    dim3 grid;
    grid.x = cv::gpu::divUp(cvgmImage_.cols, block.x); //6 is the size-1 of the Bresenham circle
    grid.y = cv::gpu::divUp(cvgmImage_.rows, block.y);

	kernelCalcSaliency<<<grid, block>>>(cvgmImage_, usHalfSizeRound_, ucContrastThreshold_, fSaliencyThreshold_, *pcvgmSaliency_, *pcvgmKeyPointLocations_);
	cudaSafeCall( cudaGetLastError() );
    cudaSafeCall( cudaDeviceSynchronize() );

    unsigned int uCount;
    cudaSafeCall( cudaMemcpy(&uCount, pCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

    return uCount;
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
    cudaSafeCall( cudaMemset(pCounter, 0, sizeof(unsigned int)) );

    dim3 block(256);
    dim3 grid;
    grid.x = cv::gpu::divUp(uMaxSalientPoints_, block.x);

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

void thrustSort(float2* pnLoc_, float* pfResponse_, const unsigned int nCorners_)
{
    thrust::device_ptr<float2> loc_ptr(pnLoc_);
    thrust::device_ptr<float> response_ptr(pfResponse_);
    thrust::sort_by_key(response_ptr, response_ptr + nCorners_, loc_ptr, thrust::greater<float>());
    return;
}


#define SHARE
#ifdef SHARE
	#define HALFROUND 9
	#define TWO_HALFROUND 18 // HALFROUND*2
	#define WHOLE_WIDTH 51 // BLOCKDIM_X + HALFROUND*2
	#define WHOLE_HEIGHT 51 // BLOCKDIM_Y + HALFROUND*2
	#define BLOCKDIM_X 32
	#define BLOCKDIM_Y 32 
#endif
struct SFastDescriptor {
	//input
	cv::gpu::DevMem2D_<uchar> cvgmImage_;
	unsigned int uTotalParticles_;      
	unsigned int usHalfPatchSizeRound_; //must equal to HALFROUND

	const short2* ps2KeyPointsLocations_;
	const float* pfKeyPointsResponse_; 

	//output for assign and input for ()
	cv::gpu::DevMem2D_<float> cvgmParticleResponses_;
	//output
	cv::gpu::DevMem2D_<int4> cvgmParticleDescriptors_;
#ifdef SHARE
	__device__ void assign(){
		const int nKeyPointIdx = threadIdx.x + blockIdx.x * blockDim.x;
		if (nKeyPointIdx >= uTotalParticles_) return;

		const short2& s2Loc = ps2KeyPointsLocations_[nKeyPointIdx];
		if( s2Loc.x < usHalfPatchSizeRound_ || s2Loc.x >= cvgmParticleResponses_.cols - usHalfPatchSizeRound_ || s2Loc.y < usHalfPatchSizeRound_ || s2Loc.y >= cvgmParticleResponses_.rows - usHalfPatchSizeRound_ ) return;

		cvgmParticleResponses_.ptr(s2Loc.y)[s2Loc.x] = pfKeyPointsResponse_[nKeyPointIdx];
		return;
	}

	//shared memory
	__device__ int4 devGetFastDescriptor(const uchar* pImage_, const int r, const int c ){
	int4 n4Descriptor;
	n4Descriptor.x = n4Descriptor.y = n4Descriptor.z = n4Descriptor.w = 0;
	uchar Color;
	Color = *(pImage_ + (r-3)*( WHOLE_WIDTH ) + c); //1
	n4Descriptor.x += Color; 
	n4Descriptor.x = n4Descriptor.x << 8;
	Color = *(pImage_ + (r-6)*( WHOLE_WIDTH ) + c + 2); //2 B6
	n4Descriptor.x += Color; 
	n4Descriptor.x = n4Descriptor.x << 8;
	Color = *(pImage_ + (r-2)*( WHOLE_WIDTH ) + c + 2); //3
	n4Descriptor.x += Color; 
	n4Descriptor.x = n4Descriptor.x << 8;
	Color = *(pImage_ + (r-2)*( WHOLE_WIDTH ) + c + 6); //4 B6
	n4Descriptor.x += Color; 

	Color = *(pImage_ + (r)*( WHOLE_WIDTH ) + c + 3); //5
	n4Descriptor.y += Color; 
	n4Descriptor.y = n4Descriptor.y << 8;
	Color = *(pImage_ + (r+2)*( WHOLE_WIDTH ) + c + 6); //6 B6
	n4Descriptor.y += Color; 
	n4Descriptor.y = n4Descriptor.y << 8;
	Color = *(pImage_ + (r+2)*( WHOLE_WIDTH ) + c + 2); //7
	n4Descriptor.y += Color; 
	n4Descriptor.y = n4Descriptor.y << 8;
	Color = *(pImage_ + (r+6)*( WHOLE_WIDTH ) + c + 2); //8 B6
	n4Descriptor.y += Color; 

	Color = *(pImage_ + (r+3)*( WHOLE_WIDTH ) + c ); //9
	n4Descriptor.z += Color; 
	n4Descriptor.z = n4Descriptor.z << 8;
	Color = *(pImage_ + (r+6)*( WHOLE_WIDTH ) + c-2 ); //10 B6
	n4Descriptor.z += Color; 
	n4Descriptor.z = n4Descriptor.z << 8;
	Color = *(pImage_ + (r+2)*( WHOLE_WIDTH ) + c-2 ); //11
	n4Descriptor.z += Color; 
	n4Descriptor.z = n4Descriptor.z << 8;
	Color = *(pImage_ + (r+2)*( WHOLE_WIDTH ) + c-6 ); //12
	n4Descriptor.z += Color; 
	
	Color = *(pImage_ + (r)*( WHOLE_WIDTH ) + c-3 ); //13
	n4Descriptor.w += Color; 
	n4Descriptor.w = n4Descriptor.w << 8;
	Color = *(pImage_ + (r-2)*( WHOLE_WIDTH ) + c-6 ); //14 B6
	n4Descriptor.w += Color; 
	n4Descriptor.w = n4Descriptor.w << 8;
	Color = *(pImage_ + (r-2)*( WHOLE_WIDTH ) + c-2 ); //15
	n4Descriptor.w += Color; 
	n4Descriptor.w = n4Descriptor.w << 8;
	Color = *(pImage_ + (r-6)*( WHOLE_WIDTH ) + c-2 ); //16 B6
	n4Descriptor.w += Color; 
	return n4Descriptor;
}
	__device__ void operator () (){
		// assert ( usHalfPatchSizeRound_ == HALFROUND )z

		const int nGrid_C = blockIdx.x * blockDim.x;
		const int nGrid_R = blockIdx.y * blockDim.y;
		const int c = threadIdx.x + nGrid_C;
		const int r = threadIdx.y + nGrid_R;
	
		__shared__ uchar _sImage[  WHOLE_WIDTH  * WHOLE_HEIGHT  ];

		bool bOutterUpY = r >= 0;
		bool bOutterDownY = r < cvgmImage_.rows;
		bool bOutterLeftX = c >= 0;
		bool bOutterRightX = c < cvgmImage_.cols;

		bool bInnerDownY  = r >= HALFROUND;
		bool bInnerUpY    = r < cvgmImage_.rows - HALFROUND;
		bool bInnerLeftX  = c >= HALFROUND;
		bool bInnerRightX = c < cvgmImage_.cols - HALFROUND;

		bool bThreadLeftX = threadIdx.x < HALFROUND;
		bool bThreadRightX = threadIdx.x >= BLOCKDIM_X - HALFROUND;
		bool bThreadUpY = threadIdx.y < HALFROUND;
		bool bThreadDownY = threadIdx.y >= BLOCKDIM_Y - HALFROUND;
		 //{ __syncthreads(); return; }
		//copy image to shared memory
			//up left
		if( bThreadLeftX && bThreadUpY && bOutterDownY && bOutterRightX)
			_sImage[threadIdx.x + threadIdx.y*WHOLE_WIDTH ] = cvgmImage_.ptr(r - HALFROUND )[c - HALFROUND ];
		//left
		if( bThreadLeftX && bOutterDownY && bInnerLeftX && bOutterRightX )
			_sImage[threadIdx.x + (threadIdx.y + HALFROUND)*WHOLE_WIDTH ] = cvgmImage_.ptr(r )[c- HALFROUND ];
		//down left
		if( bThreadLeftX && bThreadDownY && bInnerDownY && bOutterRightX )
			_sImage[threadIdx.x + (threadIdx.y + TWO_HALFROUND )*WHOLE_WIDTH ] = cvgmImage_.ptr(r+HALFROUND )[c- HALFROUND ];

		//up
		if( bThreadUpY && bInnerUpY && bOutterDownY && bOutterRightX )
			_sImage[threadIdx.x + HALFROUND + (threadIdx.y )*WHOLE_WIDTH ] = cvgmImage_.ptr(r - HALFROUND )[c];
		//center
		if( bInnerLeftX && bInnerRightX && bInnerUpY && bInnerDownY )
			_sImage[threadIdx.x + HALFROUND + ( threadIdx.y + HALFROUND ) *WHOLE_WIDTH ] = cvgmImage_.ptr(r )[c];
		//down
		if( bThreadDownY && bInnerDownY && bOutterRightX )
			_sImage[threadIdx.x + HALFROUND + (threadIdx.y + TWO_HALFROUND )*WHOLE_WIDTH ] = cvgmImage_.ptr(r + HALFROUND )[c];

		//up right
		if( bThreadRightX && bThreadUpY && bInnerUpY && bOutterDownY && bInnerRightX )
			_sImage[threadIdx.x + TWO_HALFROUND + threadIdx.y*WHOLE_WIDTH ] = cvgmImage_.ptr(r - HALFROUND )[c + HALFROUND ];
		//right
		if( bThreadRightX && bOutterUpY && bOutterDownY && bInnerRightX )
			_sImage[threadIdx.x + TWO_HALFROUND + (threadIdx.y + HALFROUND)*WHOLE_WIDTH ] = cvgmImage_.ptr(r )[c + HALFROUND ];
		//down right
		if( bThreadRightX && bThreadDownY && bOutterUpY && bInnerDownY && bInnerRightX )
			_sImage[threadIdx.x + TWO_HALFROUND + ( threadIdx.y + TWO_HALFROUND )*WHOLE_WIDTH ] = cvgmImage_.ptr( r + HALFROUND )[ c + HALFROUND ];
	
		// synchronize threads in this block
		__syncthreads();

		if( bInnerLeftX && bInnerRightX && bInnerUpY && bInnerDownY ){
			short2 s2Loc = make_short2( c, r );
			if ( cvgmParticleResponses_.ptr(r)[c] < 0.02f ) return;
			//int4 n4Desc; devGetFastDescriptor(s2Loc.y,s2Loc.x,&n4Desc);
			int4 n4DescShare = devGetFastDescriptor(_sImage,threadIdx.y+HALFROUND,threadIdx.x+HALFROUND);
			cvgmParticleDescriptors_.ptr(s2Loc.y)[s2Loc.x] = n4DescShare;//n4Desc;
		}
		return;
	}
#else
	__device__ int4 devGetFastDescriptor( const int r, const int c ){
		int4 n4Descriptor;
		n4Descriptor.x = n4Descriptor.y = n4Descriptor.z = n4Descriptor.w = 0;
		uchar Color;
		Color = cvgmImage_.ptr(r-3)[c  ];//1
		n4Descriptor.x += Color; 
		n4Descriptor.x = n4Descriptor.x << 8;
		Color = cvgmImage_.ptr(r-6)[c+2];//2 B6
		n4Descriptor.x += Color; 
		n4Descriptor.x = n4Descriptor.x << 8;
		Color = cvgmImage_.ptr(r-2)[c+2];//3
		n4Descriptor.x += Color; 
		n4Descriptor.x = n4Descriptor.x << 8;
		Color = cvgmImage_.ptr(r-2)[c+6];//4 B6
		n4Descriptor.x += Color; 


		Color = cvgmImage_.ptr(r  )[c+3];//5
		n4Descriptor.y += Color; 
		n4Descriptor.y = n4Descriptor.y << 8;
		Color = cvgmImage_.ptr(r+2)[c+6];//6 B6
		n4Descriptor.y += Color; 
		n4Descriptor.y = n4Descriptor.y << 8;
		Color = cvgmImage_.ptr(r+2)[c+2];//7
		n4Descriptor.y += Color; 
		n4Descriptor.y = n4Descriptor.y << 8;
		Color = cvgmImage_.ptr(r+6)[c+2];//8 B6
		n4Descriptor.y += Color; 

		Color = cvgmImage_.ptr(r+3)[c  ];//9
		n4Descriptor.z += Color; 
		n4Descriptor.z = n4Descriptor.z << 8;
		Color= cvgmImage_.ptr(r+6)[c-2];//10 B6
		n4Descriptor.z += Color; 
		n4Descriptor.z = n4Descriptor.z << 8;
		Color= cvgmImage_.ptr(r+2)[c-2];//11
		n4Descriptor.z += Color; 
		n4Descriptor.z = n4Descriptor.z << 8;
		Color= cvgmImage_.ptr(r+2)[c-6];//12 B6
		n4Descriptor.z += Color; 
	
		Color= cvgmImage_.ptr(r  )[c-3];//13
		n4Descriptor.w += Color; 
		n4Descriptor.w = n4Descriptor.w << 8;
		Color= cvgmImage_.ptr(r-2)[c-6];//14 B6
		n4Descriptor.w += Color; 
		n4Descriptor.w = n4Descriptor.w << 8;
		Color= cvgmImage_.ptr(r-2)[c-2];//15
		n4Descriptor.w += Color; 
		n4Descriptor.w = n4Descriptor.w << 8;
		Color= cvgmImage_.ptr(r-6)[c-2];//16 B6
		n4Descriptor.w += Color; 
		return;
	}
	//single ring
	__device__ int4 devGetFastDescriptor1( const int r, const int c ){
		int4 n4Descriptor;
		n4Descriptor.x = n4Descriptor.y = n4Descriptor.z = n4Descriptor.w = 0;
		uchar Color;
		Color = cvgmImage_.ptr(r-3)[c  ];//1
		n4Descriptor.x += Color; 
		n4Descriptor.x = n4Descriptor.x << 8;
		Color = cvgmImage_.ptr(r-3)[c+1];//2
		n4Descriptor.x += Color; 
		n4Descriptor.x = n4Descriptor.x << 8;
		Color = cvgmImage_.ptr(r-2)[c+2];//3
		n4Descriptor.x += Color; 
		n4Descriptor.x = n4Descriptor.x << 8;
		Color = cvgmImage_.ptr(r-1)[c+3];//4
		n4Descriptor.x += Color; 


		Color = cvgmImage_.ptr(r  )[c+3];//5
		n4Descriptor.y += Color; 
		n4Descriptor.y = n4Descriptor.y << 8;
		Color = cvgmImage_.ptr(r+1)[c+3];//6
		n4Descriptor.y += Color; 
		n4Descriptor.y = n4Descriptor.y << 8;
		Color = cvgmImage_.ptr(r+2)[c+2];//7
		n4Descriptor.y += Color; 
		n4Descriptor.y = n4Descriptor.y << 8;
		Color = cvgmImage_.ptr(r+3)[c+1];//8
		n4Descriptor.y += Color; 

		Color = cvgmImage_.ptr(r+3)[c  ];//9
		n4Descriptor.z += Color; 
		n4Descriptor.z = n4Descriptor.z << 8;
		Color= cvgmImage_.ptr(r+3)[c-1];//10
		n4Descriptor.z += Color; 
		n4Descriptor.z = n4Descriptor.z << 8;
		Color= cvgmImage_.ptr(r+2)[c-2];//11
		n4Descriptor.z += Color; 
		n4Descriptor.z = n4Descriptor.z << 8;
		Color= cvgmImage_.ptr(r+1)[c-3];//12
		n4Descriptor.z += Color; 
	
		Color= cvgmImage_.ptr(r  )[c-3];//13
		n4Descriptor.w += Color; 
		n4Descriptor.w = n4Descriptor.w << 8;
		Color= cvgmImage_.ptr(r-1)[c-3];//14
		n4Descriptor.w += Color; 
		n4Descriptor.w = n4Descriptor.w << 8;
		Color= cvgmImage_.ptr(r-2)[c-2];//15
		n4Descriptor.w += Color; 
		n4Descriptor.w = n4Descriptor.w << 8;
		Color= cvgmImage_.ptr(r-3)[c-1];//16
		n4Descriptor.w += Color; 
		return;
	}
	__device__ void normal(){
		// assert ( usHalfPatchSizeRound_ == HALFROUND )z
		const int nKeyPointIdx = threadIdx.x + blockIdx.x * blockDim.x;
		if (nKeyPointIdx >= uTotalParticles_) return;

		const short2& s2Loc = ps2KeyPointsLocations_[nKeyPointIdx];
		if( s2Loc.x < usHalfPatchSizeRound_ || s2Loc.x >= cvgmImage_.cols - usHalfPatchSizeRound_ || s2Loc.y < usHalfPatchSizeRound_ || s2Loc.y >= cvgmImage_.rows - usHalfPatchSizeRound_ ) return;

		cvgmParticleResponses_.ptr(s2Loc.y)[s2Loc.x] = pfKeyPointsResponse_[nKeyPointIdx];
		cvgmParticleDescriptors_.ptr(s2Loc.y)[s2Loc.x] = devGetFastDescriptor(s2Loc.y,s2Loc.x);
		return;
	}
#endif
};

#ifdef SHARE
__global__ void kernelAssignKeyPoint(SFastDescriptor sFD){
	sFD.assign();
}
#endif

__global__ void kernelExtractAllDescriptorFast(SFastDescriptor sFD){
#ifdef SHARE
	sFD();
#else
	sFD.normal();
#endif
}
/*
collect all key points and key point response and set a frame of saliency frame
input values:
  ps2KeyPointsLocations_: 
returned values:
  pcvgmParticleResponses_: a frame of saliency response
*/
void cudaExtractAllDescriptorFast(const cv::gpu::GpuMat& cvgmImage_, 
								  const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, 
								  const unsigned int uTotalParticles_, const unsigned int usHalfPatchSize_,  
								  cv::gpu::GpuMat* pcvgmParticleResponses_, cv::gpu::GpuMat* pcvgmParticleDescriptor_ ){
	cudaEvent_t     start, stop;
    cudaSafeCall( cudaEventCreate( &start ) );
    cudaSafeCall( cudaEventCreate( &stop ) );
    cudaSafeCall( cudaEventRecord( start, 0 ) );

	if(uTotalParticles_ == 0) return;
	
	struct SFastDescriptor sFD;
	//input
	sFD.cvgmImage_ = cvgmImage_;
	sFD.uTotalParticles_ = uTotalParticles_;
	sFD.usHalfPatchSizeRound_ = unsigned int(usHalfPatchSize_*1.5);

	sFD.ps2KeyPointsLocations_ = ps2KeyPointsLocations_;
	sFD.pfKeyPointsResponse_ = pfKeyPointsResponse_;
	//output
	sFD.cvgmParticleResponses_ = *pcvgmParticleResponses_;
	sFD.cvgmParticleDescriptors_ = *pcvgmParticleDescriptor_;
#ifdef SHARE
	dim3 block(256);
    dim3 grid;
    grid.x = cv::gpu::divUp(uTotalParticles_, block.x);
	
	kernelAssignKeyPoint<<<grid, block>>>( sFD );
	cudaSafeCall( cudaGetLastError() );
	
	block.x = BLOCKDIM_X;
	block.y = BLOCKDIM_Y;
	grid.x = cv::gpu::divUp(cvgmImage_.cols, block.x);
	grid.y = cv::gpu::divUp(cvgmImage_.rows, block.y);
	
	kernelExtractAllDescriptorFast<<<grid, block>>>( sFD );
	cudaSafeCall( cudaGetLastError() );
	cudaSafeCall( cudaDeviceSynchronize() );
#else
	dim3 block(256);
    dim3 grid;
    grid.x = cv::gpu::divUp(uTotalParticles_, block.x);
	kernelExtractAllDescriptorFast<<<grid, block>>>( sFD );
	cudaSafeCall( cudaGetLastError() );
	cudaSafeCall( cudaDeviceSynchronize() );
#endif
	cudaSafeCall( cudaEventRecord( stop, 0 ) );
    cudaSafeCall( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    cudaSafeCall( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    printf( "Extract Fast:  %3.1f ms\n", elapsedTime );

    cudaSafeCall( cudaEventDestroy( start ) );
    cudaSafeCall( cudaEventDestroy( stop ) );

	return;
}

class CPredictAndMatch{
public:
	cv::gpu::DevMem2D_<int4>   _cvgmParticleDescriptorsPrev;
	cv::gpu::DevMem2D_<float>  _cvgmParticleResponsesPrev;
	
	cv::gpu::DevMem2D_<int4>   _cvgmParticleDescriptorCurrTmp;
	cv::gpu::DevMem2D_<float>  _cvgmSaliencyCurr;

	cv::gpu::DevMem2D_<float>  _cvgmMinMatchDistance;
	cv::gpu::DevMem2D_<short2> _cvgmMatchedLocationPrev;

	short _sSearchRange;
	float _fMatchThreshold;
	unsigned short _usHalfSize;
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
	__device__ __forceinline__ float devMatch(const float& fMatchThreshold_, 
											  const int4& n4DesPrev_, const short2 s2PredicLoc_, short2* ps2BestLoc_){
		float fResponse = 0.f;
		short2 s2Loc;
		float fMinDist = 300.f;
		//search for the 7x7 neighbourhood for 
		for(short r = -_sSearchRange; r <= _sSearchRange; r++ ){
			for(short c = -_sSearchRange; c <= _sSearchRange; c++ ){
				s2Loc = s2PredicLoc_ + make_short2( c, r ); 
				if(s2Loc.x < _usHalfSizeRound || s2Loc.x >= _cvgmParticleResponsesPrev.cols - _usHalfSizeRound || s2Loc.y < _usHalfSizeRound || s2Loc.y >= _cvgmParticleResponsesPrev.rows - _usHalfSizeRound ) continue;
				fResponse = _cvgmSaliencyCurr.ptr(s2Loc.y)[s2Loc.x];
				if( fResponse > 0.1f ){
					int4 n4Des = _cvgmParticleDescriptorCurrTmp.ptr(s2Loc.y)[s2Loc.x]; 
					float fDist = dL1(n4Des,n4DesPrev_);
					if ( fDist < fMatchThreshold_ ){
						if (  fMinDist > fDist ){
							fMinDist = fDist;
							*ps2BestLoc_ = s2Loc;
						}
					}
				}//if sailent corner exits
			}//for 
		}//for
		return fMinDist;
	}//devMatch

	__device__ __forceinline__ void operator () (){
		const int c = threadIdx.x + blockIdx.x * blockDim.x;
		const int r = threadIdx.y + blockIdx.y * blockDim.y;

		if( c < _usHalfSizeRound || c >= _cvgmParticleResponsesPrev.cols - _usHalfSizeRound || r < _usHalfSizeRound || r >= _cvgmParticleResponsesPrev.rows - _usHalfSizeRound ) return;
		if(_cvgmParticleResponsesPrev.ptr(r)[c] < 0.1f) return;
		const int4& n4DesPrev = _cvgmParticleDescriptorsPrev.ptr(r)[c];
		
		short2 s2BestLoc; 
		const float fDist = devMatch( _fMatchThreshold, n4DesPrev, make_short2(c,r), &s2BestLoc );
		
		if( fDist < 299.f ){ //300.f is the max distance
			const float& fMin = _cvgmMinMatchDistance.ptr(s2BestLoc.y)[s2BestLoc.x];//competing for the same memory
			atomicInc(&_devuOther, (unsigned int)(-1));//deleted particle counter increase by 1
			if(fMin > 299.f) {//it has NEVER been matched before.
				atomicInc(&_devuNewlyAddedCounter, (unsigned int)(-1));//deleted particle counter increase by 1
				_cvgmMinMatchDistance     .ptr(s2BestLoc.y)[s2BestLoc.x] = fDist;
				_cvgmMatchedLocationPrev  .ptr(s2BestLoc.y)[s2BestLoc.x] = make_short2(c,r);
			}
			else{//it has been matched 
				//double match means one of them will be removed
				atomicInc(&_devuCounter, (unsigned int)(-1));//deleted particle counter increase by 1
				if ( fMin > fDist ){//record it if it is a better match than previous match
					_cvgmMinMatchDistance     .ptr(s2BestLoc.y)[s2BestLoc.x] = fDist;
					_cvgmMatchedLocationPrev  .ptr(s2BestLoc.y)[s2BestLoc.x] = make_short2(c,r);
				}//if
			}//else
		}//if
		else{//C) if no match found 
			atomicInc(&_devuCounter, (unsigned int)(-1));//deleted particle counter increase by 1
		}//lost
		return;
	}
};//class CPredictAndMatch

__global__ void kernelPredictAndMatch(CPredictAndMatch cPAM_){
	cPAM_ ();
}
unsigned int cudaTrackFast(float fMatchThreshold_, const unsigned short usHalfSize_, const short sSearchRange_, 
							const cv::gpu::GpuMat& cvgmParticleDescriptorPrev_, const cv::gpu::GpuMat& cvgmParticleResponsesPrev_, 
							const cv::gpu::GpuMat& cvgmParticleDescriptorCurrTmp_, const cv::gpu::GpuMat& cvgmSaliencyCurr_, 
							cv::gpu::GpuMat* pcvgmMinMatchDistance_,
							cv::gpu::GpuMat* pcvgmMatchedLocationPrev_){

	cudaEvent_t     start, stop;
    cudaSafeCall( cudaEventCreate( &start ) );
    cudaSafeCall( cudaEventCreate( &stop ) );
    cudaSafeCall( cudaEventRecord( start, 0 ) );

	dim3 block(32,8);
	dim3 grid;
	grid.x = cv::gpu::divUp(cvgmParticleResponsesPrev_.cols - 6, block.x); //6 is the size-1 of the Bresenham circle
    grid.y = cv::gpu::divUp(cvgmParticleResponsesPrev_.rows - 6, block.y);

	CPredictAndMatch cPAM;
	cPAM._cvgmParticleDescriptorsPrev = cvgmParticleDescriptorPrev_;
	cPAM._cvgmParticleResponsesPrev = cvgmParticleResponsesPrev_;

	cPAM._cvgmParticleDescriptorCurrTmp = cvgmParticleDescriptorCurrTmp_;
	cPAM._cvgmSaliencyCurr = cvgmSaliencyCurr_;


	pcvgmMinMatchDistance_->setTo(300.f);
	cPAM._cvgmMinMatchDistance = *pcvgmMinMatchDistance_;
	pcvgmMatchedLocationPrev_->setTo(cv::Scalar::all(0));
	cPAM._cvgmMatchedLocationPrev = *pcvgmMatchedLocationPrev_; 

	cPAM._fMatchThreshold = fMatchThreshold_;
	cPAM._usHalfSize = usHalfSize_;
	cPAM._usHalfSizeRound = (unsigned short)(usHalfSize_*1.5);
	cPAM._sSearchRange = sSearchRange_;

	void* pCounter;
    cudaSafeCall( cudaGetSymbolAddress(&pCounter, _devuCounter) );
	cudaSafeCall( cudaMemset(pCounter, 0, sizeof(unsigned int)) );

	void* pCounterMatch;
    cudaSafeCall( cudaGetSymbolAddress(&pCounterMatch, _devuNewlyAddedCounter) );
	cudaSafeCall( cudaMemset(pCounterMatch, 0, sizeof(unsigned int)) );

	void* pCounterOther;
    cudaSafeCall( cudaGetSymbolAddress(&pCounterOther, _devuOther) );
	cudaSafeCall( cudaMemset(pCounterOther, 0, sizeof(unsigned int)) );

	kernelPredictAndMatch<<<grid, block>>>(cPAM);
	cudaSafeCall( cudaGetLastError() );
    cudaSafeCall( cudaDeviceSynchronize() );

    unsigned int uDeleted ;
    cudaSafeCall( cudaMemcpy(&uDeleted, pCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	unsigned int uMatched ;
    cudaSafeCall( cudaMemcpy(&uMatched, pCounterMatch, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	unsigned int uOther ;
    cudaSafeCall( cudaMemcpy(&uOther, pCounterOther, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

	cudaSafeCall( cudaEventRecord( stop, 0 ) );
    cudaSafeCall( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    cudaSafeCall( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    printf( "Track Fast:  %3.1f ms\n", elapsedTime );

    cudaSafeCall( cudaEventDestroy( start ) );
    cudaSafeCall( cudaEventDestroy( stop ) );

	return uMatched;
}//cudaTrack

struct SMatchedAndNewlyAddedKeyPointsCollection{
	
	cv::gpu::DevMem2D_<float> _cvgmSaliency;
	cv::gpu::DevMem2D_<int4>  _cvgmParticleDescriptorCurrTmp;

	cv::gpu::DevMem2D_<short2> _cvgmParticleVelocityPrev;
	cv::gpu::DevMem2D_<uchar>  _cvgmParticleAgePrev;
	cv::gpu::DevMem2D_<short2> _cvgmParticleVelocityCurr;
	cv::gpu::DevMem2D_<uchar>  _cvgmParticleAgeCurr;
	cv::gpu::DevMem2D_<float>  _cvgmParticleResponseCurr;
	cv::gpu::DevMem2D_<int4>   _cvgmParticleDescriptorCurr;

	cv::gpu::DevMem2D_<short2> _cvgmMatchedLocationPrev;
	cv::gpu::DevMem2D_<float>  _cvgmMinMatchDistance;

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
		_cvgmParticleDescriptorCurr.ptr(r)[c] = make_int4(0,0,0,0);
		const float& fResponse = _cvgmSaliency.ptr(r)[c];

		if( fResponse < 0.1f ) return; 

		if(_cvgmMinMatchDistance.ptr(r)[c] > 299.f ){
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
			_cvgmParticleAgeCurr	   .ptr(r)[c] = _cvgmParticleAgePrev.ptr(s2PrevLoc.y)[s2PrevLoc.x] + 1; //update age
		}
		return;
	}//operator()
};//SCollectUnMatchedKeyPoints
__global__ void kernelMatchedAndNewlyAddedKeyPointsCollection(SMatchedAndNewlyAddedKeyPointsCollection sCUMKP_){
	sCUMKP_ ();
}

__global__ void kernerlAddNewParticlesFast( const unsigned int uTotalParticles_,   
										const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, 
										const cv::gpu::DevMem2D_<int4> cvgmParticleDescriptorTmp_,
										cv::gpu::DevMem2D_<float> cvgmParticleResponse_, cv::gpu::DevMem2D_<int4> cvgmParticleDescriptor_){

	const int nKeyPointIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (nKeyPointIdx >= uTotalParticles_) return;

	const short2& s2Loc = ps2KeyPointsLocations_[nKeyPointIdx];
	cvgmParticleResponse_.ptr(s2Loc.y)[s2Loc.x] = pfKeyPointsResponse_[nKeyPointIdx];
	cvgmParticleDescriptor_.ptr(s2Loc.y)[s2Loc.x] = cvgmParticleDescriptorTmp_.ptr(s2Loc.y)[s2Loc.x]; 
	return;
}

/*


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
}*/
//after track, all key points in current frame are collected into 1.matched key point group 2.newly added key point group
void cudaCollectKeyPointsFast(unsigned int uTotalParticles_, unsigned int uMaxNewKeyPoints_, const float fRho_,
												 const cv::gpu::GpuMat& cvgmSaliency_, 
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
	SMatchedAndNewlyAddedKeyPointsCollection sCUMKP;

	cudaEvent_t     start, stop;
    cudaSafeCall( cudaEventCreate( &start ) );
    cudaSafeCall( cudaEventCreate( &stop ) );
    cudaSafeCall( cudaEventRecord( start, 0 ) );

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
	kernelMatchedAndNewlyAddedKeyPointsCollection<<<grid, block>>>(sCUMKP);
	cudaSafeCall( cudaGetLastError() );

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
	kernerlAddNewParticlesFast<<<grid, block>>>(uNewlyAdded, pcvgmNewlyAddedKeyPointLocation_->ptr<short2>(), pcvgmNewlyAddedKeyPointResponse_->ptr<float>(),
											sCUMKP._cvgmParticleDescriptorCurrTmp ,
											sCUMKP._cvgmParticleResponseCurr, sCUMKP._cvgmParticleDescriptorCurr);

	
	cudaSafeCall( cudaEventRecord( stop, 0 ) );
    cudaSafeCall( cudaEventSynchronize( stop ) );
    float   elapsedTime;
    cudaSafeCall( cudaEventElapsedTime( &elapsedTime, start, stop ) );
    printf( "Collect Fast:  %3.1f ms\n", elapsedTime );

    cudaSafeCall( cudaEventDestroy( start ) );
    cudaSafeCall( cudaEventDestroy( stop ) );



    return;
}






}//semidense
}//device
}//btl
