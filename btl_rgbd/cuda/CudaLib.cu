
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/devmem2d.hpp>
#include "cv/common.hpp" //copied from opencv
#include "../OtherUtil.hpp"
#include <math_constants.h>
#include "pcl/limits.hpp"
#include "pcl/device.hpp"
#include <vector>

namespace btl{ namespace device
{

__global__ void kernelTestFloat3(const cv::gpu::DevMem2D_<float3> cvgmIn_, cv::gpu::DevMem2D_<float3> cvgmOut_)
{
	const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;
	if (nX >= cvgmIn_.cols && nY >= cvgmIn_.rows) return;
	const float3& in = cvgmIn_.ptr(nY)[nX];
	float3& out  = cvgmOut_.ptr(nY)[nX];
	out.x = out.y = out.z = (in.x + in.y + in.z)/3;
}
void cudaTestFloat3( const cv::gpu::GpuMat& cvgmIn_, cv::gpu::GpuMat* pcvgmOut_ )
{
	pcvgmOut_->create(cvgmIn_.size(),CV_32FC3);
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(cvgmIn_.cols, block.x), cv::gpu::divUp(cvgmIn_.rows, block.y));
	//run kernel
	kernelTestFloat3<<<grid,block>>>( cvgmIn_,*pcvgmOut_ );
	cudaSafeCall ( cudaGetLastError () );
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//depth to disparity
__global__ void kernelInverse(const cv::gpu::DevMem2Df cvgmIn_, cv::gpu::DevMem2Df cvgmOut_){
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;
	if (nX >= cvgmIn_.cols && nY >= cvgmIn_.rows) return;
	if(fabsf(cvgmIn_.ptr(nY)[nX]) > 0.f )
		cvgmOut_.ptr(nY)[nX] = 1.f/cvgmIn_.ptr(nY)[nX];
	else
		cvgmOut_.ptr(nY)[nX] = pcl::device::numeric_limits<float>::quiet_NaN();
}//kernelInverse

void cudaDepth2Disparity( const cv::gpu::GpuMat& cvgmDepth_, cv::gpu::GpuMat* pcvgmDisparity_ ){
	//not necessary as pcvgmDisparity has been allocated in VideoSourceKinect()
	//pcvgmDisparity_->create(cvgmDepth_.size(),CV_32F);
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(cvgmDepth_.cols, block.x), cv::gpu::divUp(cvgmDepth_.rows, block.y));
	//run kernel
	kernelInverse<<<grid,block>>>( cvgmDepth_,*pcvgmDisparity_ );
	cudaSafeCall ( cudaGetLastError () );
}//cudaDepth2Disparity

__global__ void kernelInverse2(const cv::gpu::DevMem2Df cvgmIn_, cv::gpu::DevMem2Df cvgmOut_){
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;
	if (nX >= cvgmIn_.cols && nY >= cvgmIn_.rows) return;
	if(fabsf(cvgmIn_.ptr(nY)[nX]) > 0.f )
		cvgmOut_.ptr(nY)[nX] = 1000.f/cvgmIn_.ptr(nY)[nX];
	else
		cvgmOut_.ptr(nY)[nX] = pcl::device::numeric_limits<float>::quiet_NaN();
}//kernelInverse

void cudaDepth2Disparity2( const cv::gpu::GpuMat& cvgmDepth_, cv::gpu::GpuMat* pcvgmDisparity_ ){
	//convert the depth from mm to m
	//not necessary as pcvgmDisparity has been allocated in VideoSourceKinect()
	//pcvgmDisparity_->create(cvgmDepth_.size(),CV_32F);
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(cvgmDepth_.cols, block.x), cv::gpu::divUp(cvgmDepth_.rows, block.y));
	//run kernel
	kernelInverse2<<<grid,block>>>( cvgmDepth_,*pcvgmDisparity_ );
	cudaSafeCall ( cudaGetLastError () );
}//cudaDepth2Disparity


void cudaDisparity2Depth( const cv::gpu::GpuMat& cvgmDisparity_, cv::gpu::GpuMat* pcvgmDepth_ ){
	pcvgmDepth_->create(cvgmDisparity_.size(),CV_32F);
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(cvgmDisparity_.cols, block.x), cv::gpu::divUp(cvgmDisparity_.rows, block.y));
	//run kernel
	kernelInverse<<<grid,block>>>( cvgmDisparity_,*pcvgmDepth_ );
	cudaSafeCall ( cudaGetLastError () );
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//global constant used by kernelUnprojectIR() and cudaUnProjectIR()
__constant__ float _aIRCameraParameter[4];// 1/f_x, 1/f_y, u, v for IR camera; constant memory declaration
//
__global__ void kernelUnprojectIRCVmmCVm(const cv::gpu::DevMem2Df cvgmDepth_,
	cv::gpu::DevMem2D_<float3> cvgmIRWorld_) {
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;

	if (nX >= cvgmIRWorld_.cols && nY >= cvgmIRWorld_.rows) return;

	const float& fDepth = cvgmDepth_.ptr(nY)[nX];
	float3& temp = cvgmIRWorld_.ptr(nY)[nX];
		
	if(400.f < fDepth && fDepth < 4000.f ){ //truncate, fDepth is captured from openni and always > 0
		temp.z = fDepth /1000.f;//convert to meter z 5 million meter is added according to experience. as the OpenNI
		//coordinate system is defined w.r.t. the camera plane which is 0.5 centimeters in front of the camera center
		temp.x = (nX - _aIRCameraParameter[2]) * _aIRCameraParameter[0] * temp.z;
		temp.y = (nY - _aIRCameraParameter[3]) * _aIRCameraParameter[1] * temp.z;
	}//if within 0.4m - 4m
	else{
		temp.x = temp.y = temp.z = pcl::device::numeric_limits<float>::quiet_NaN();
	}//else

	return;
}//kernelUnprojectIRCVCV

void cudaUnprojectIRCVCV(const cv::gpu::GpuMat& cvgmDepth_ ,
const float& fFxIR_, const float& fFyIR_, const float& uIR_, const float& vIR_, 
cv::gpu::GpuMat* pcvgmIRWorld_ )
{
	//constant definition
	size_t sN = sizeof(float) * 4;
	float* const pIRCameraParameters = (float*) malloc( sN );
	pIRCameraParameters[0] = 1.f/fFxIR_;
	pIRCameraParameters[1] = 1.f/fFyIR_;
	pIRCameraParameters[2] = uIR_;
	pIRCameraParameters[3] = vIR_;
	cudaSafeCall( cudaMemcpyToSymbol(_aIRCameraParameter, pIRCameraParameters, sN) );
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(cvgmDepth_.cols, block.x), cv::gpu::divUp(cvgmDepth_.rows, block.y));
	//run kernel
    kernelUnprojectIRCVmmCVm<<<grid,block>>>( cvgmDepth_,*pcvgmIRWorld_ );
	cudaSafeCall ( cudaGetLastError () );
	//release temporary pointers
	free(pIRCameraParameters);
	return;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//global constant used by kernelUnprojectIR() and cudaTransformIR2RGB()
__constant__ float _aR[9];
__constant__ float _aRT[3];
__global__ void kernelTransformIR2RGBCVmCVm(const cv::gpu::DevMem2D_<float3> cvgmIRWorld_, cv::gpu::DevMem2D_<float3> cvgmRGBWorld_){
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;

	if (nX >= cvgmRGBWorld_.cols || nY >= cvgmRGBWorld_.rows) return;

	float3& rgbWorld = cvgmRGBWorld_.ptr(nY)[nX];
	const float3& irWorld  = cvgmIRWorld_ .ptr(nY)[nX];
	if( 0.4f < irWorld.z && irWorld.z < 4.f ) {
		//_aR[0] [1] [2] //row major
		//   [3] [4] [5]
		//   [6] [7] [8]
		//_aT[0]
		//   [1]
		//   [2]
		//  pRGB_ = _aR * ( pIR_ - _aT )
		//  	  = _aR * pIR_ - _aR * _aT
		//  	  = _aR * pIR_ - _aRT
		rgbWorld.x = _aR[0] * irWorld.x + _aR[1] * irWorld.y + _aR[2] * irWorld.z - _aRT[0];
		rgbWorld.y = _aR[3] * irWorld.x + _aR[4] * irWorld.y + _aR[5] * irWorld.z - _aRT[1];
		rgbWorld.z = _aR[6] * irWorld.x + _aR[7] * irWorld.y + _aR[8] * irWorld.z - _aRT[2];
	}//if irWorld.z within 0.4m-4m
	else{
		rgbWorld.x = rgbWorld.y = rgbWorld.z = pcl::device::numeric_limits<float>::quiet_NaN();
	}//set NaN
	return;
}//kernelTransformIR2RGB
void cudaTransformIR2RGBCVCV(const cv::gpu::GpuMat& cvgmIRWorld_, const float* aR_, const float* aRT_, cv::gpu::GpuMat* pcvgmRGBWorld_){
	cudaSafeCall( cudaMemcpyToSymbol(_aR,  aR_,  9*sizeof(float)) );
	cudaSafeCall( cudaMemcpyToSymbol(_aRT, aRT_, 3*sizeof(float)) );
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(pcvgmRGBWorld_->cols, block.x), cv::gpu::divUp(pcvgmRGBWorld_->rows, block.y));
	//run kernel
    kernelTransformIR2RGBCVmCVm<<<grid,block>>>( cvgmIRWorld_,*pcvgmRGBWorld_ );
	cudaSafeCall ( cudaGetLastError () );
	return;
}//cudaTransformIR2RGB
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//global constant used by kernelProjectRGB() and cudaProjectRGB()
__constant__ float _aRGBCameraParameter[4]; //fFxRGB_,fFyRGB_,uRGB_,vRGB_
__global__ void kernelProjectRGBCVmCVm(const cv::gpu::DevMem2D_<float3> cvgmRGBWorld_, cv::gpu::DevMem2Df cvgmAligned_){
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;
	// cvgmAligned_ must be preset to zero;
	if (nX >= cvgmRGBWorld_.cols || nY >= cvgmRGBWorld_.rows) return;
	const float3& rgbWorld = cvgmRGBWorld_.ptr(nY)[nX];
	if( 0.4f < rgbWorld.z  &&  rgbWorld.z  < 4.f ){
		// get 2D image projection in RGB image of the XYZ in the world
		int nXAligned = __float2int_rn( _aRGBCameraParameter[0] * rgbWorld.x / rgbWorld.z + _aRGBCameraParameter[2] );
		int nYAligned = __float2int_rn( _aRGBCameraParameter[1] * rgbWorld.y / rgbWorld.z + _aRGBCameraParameter[3] );
		//if outside image return;
		if ( nXAligned < 0 || nXAligned >= cvgmRGBWorld_.cols || nYAligned < 0 || nYAligned >= cvgmRGBWorld_.rows )	return;
		
		float fPt = cvgmAligned_.ptr(nYAligned)[nXAligned];
		if(isnan<float>(fPt)){
			cvgmAligned_.ptr(nYAligned)[nXAligned] = rgbWorld.z;
		}//if havent been asigned
		else{
			fPt = (fPt+ rgbWorld.z)/2.f;
		}//if it does use the average 
	}//if within 0.4m-4m
	//else is not required
	//the cvgmAligned_ must be preset to NaN
	return;
}//kernelProjectRGB
void cudaProjectRGBCVCV(const cv::gpu::GpuMat& cvgmRGBWorld_, 
const float& fFxRGB_, const float& fFyRGB_, const float& uRGB_, const float& vRGB_, 
cv::gpu::GpuMat* pcvgmAligned_ ){
	pcvgmAligned_->setTo(std::numeric_limits<float>::quiet_NaN());
	//constant definition
	size_t sN = sizeof(float) * 4;
	float* const pRGBCameraParameters = (float*) malloc( sN );
	pRGBCameraParameters[0] = fFxRGB_;
	pRGBCameraParameters[1] = fFyRGB_;
	pRGBCameraParameters[2] = uRGB_;
	pRGBCameraParameters[3] = vRGB_;
	cudaSafeCall( cudaMemcpyToSymbol(_aRGBCameraParameter, pRGBCameraParameters, sN) );
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(cvgmRGBWorld_.cols, block.x), cv::gpu::divUp(cvgmRGBWorld_.rows, block.y));
	//run kernel
    kernelProjectRGBCVmCVm<<<grid,block>>>( cvgmRGBWorld_,*pcvgmAligned_ );
	cudaSafeCall ( cudaGetLastError () );
	//release temporary pointers
	free(pRGBCameraParameters);
	return;
}//cudaProjectRGBCVCV()
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//const float sigma_color = 30;     //in mm
//const float sigma_space = 4;     // in pixels
__constant__ float _aSigma2InvHalf[2]; //sigma_space2_inv_half,sigma_color2_inv_half

__global__ void kernelBilateral (const cv::gpu::DevMem2Df src, cv::gpu::DevMem2Df dst )
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= src.cols || y >= src.rows)  return;

    const int R = 2;//static_cast<int>(sigma_space * 1.5);
    const int D = R * 2 + 1;

    float fValueCentre = src.ptr (y)[x];
	//if fValueCentre is NaN
	if(fValueCentre!=fValueCentre) return; 

    int tx = min (x - D/2 + D, src.cols - 1);
    int ty = min (y - D/2 + D, src.rows - 1);

    float sum1 = 0;
    float sum2 = 0;

    for (int cy = max (y - D/2, 0); cy < ty; ++cy)
    for (int cx = max (x - D/2, 0); cx < tx; ++cx){
        float  fValueNeighbour = src.ptr (cy)[cx];
		//if fValueNeighbour is NaN
		if(fValueNeighbour!=fValueNeighbour) continue; 
        float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        float color2 = (fValueCentre - fValueNeighbour) * (fValueCentre - fValueNeighbour);
        float weight = __expf (-(space2 * _aSigma2InvHalf[0] + color2 * _aSigma2InvHalf[1]) );

        sum1 += fValueNeighbour * weight;
        sum2 += weight;
    }//for for each pixel in neigbbourhood

    dst.ptr (y)[x] = sum1/sum2;
	return;
}//kernelBilateral

void cudaBilateralFiltering(const cv::gpu::GpuMat& cvgmSrc_, const float& fSigmaSpace_, const float& fSigmaColor_, cv::gpu::GpuMat* pcvgmDst_ )
{
	pcvgmDst_->setTo(std::numeric_limits<float>::quiet_NaN());
	//constant definition
	size_t sN = sizeof(float) * 2;
	float* const pSigma = (float*) malloc( sN );
	pSigma[0] = 0.5f / (fSigmaSpace_ * fSigmaSpace_);
	pSigma[1] = 0.5f / (fSigmaColor_ * fSigmaColor_);
	cudaSafeCall( cudaMemcpyToSymbol(_aSigma2InvHalf, pSigma, sN) );
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(cvgmSrc_.cols, block.x), cv::gpu::divUp(cvgmSrc_.rows, block.y));
	//run kernel
    kernelBilateral<<<grid,block>>>( cvgmSrc_,*pcvgmDst_ );
	cudaSafeCall ( cudaGetLastError () );
	//release temporary pointers
	free(pSigma);
	return;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelPyrDown (const cv::gpu::DevMem2Df cvgmSrc_, cv::gpu::DevMem2Df cvgmDst_, float fSigmaColor_ )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cvgmDst_.cols || y >= cvgmDst_.rows) return;

    const int D = 5;

    float center = cvgmSrc_.ptr (2 * y)[2 * x];
	if( center!=center ){
		cvgmDst_.ptr (y)[x] = pcl::device::numeric_limits<float>::quiet_NaN();
		return;
	}//if center is NaN
    int tx = min (2 * x - D / 2 + D, cvgmSrc_.cols - 1);
    int ty = min (2 * y - D / 2 + D, cvgmSrc_.rows - 1);
    int cy = max (0, 2 * y - D / 2);

    float sum = 0;
    int count = 0;

    for (; cy < ty; ++cy)
    for (int cx = max (0, 2 * x - D / 2); cx < tx; ++cx) {
        float val = cvgmSrc_.ptr (cy)[cx];
        if (fabsf (val - center) < 3 * fSigmaColor_){//
			sum += val;
			++count;
        } //if within 3*fSigmaColor_
    }//for each pixel in the neighbourhood
    cvgmDst_.ptr (y)[x] = sum / count;
}//kernelPyrDown()
void cudaPyrDown (const cv::gpu::GpuMat& cvgmSrc_, const float& fSigmaColor_, cv::gpu::GpuMat* pcvgmDst_)
{
	dim3 block (32, 8);
	dim3 grid (cv::gpu::divUp (pcvgmDst_->cols, block.x), cv::gpu::divUp (pcvgmDst_->rows, block.y));
	kernelPyrDown<<<grid, block>>>(cvgmSrc_, *pcvgmDst_, fSigmaColor_);
	cudaSafeCall ( cudaGetLastError () );
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelUnprojectRGBCVmCVm (const cv::gpu::DevMem2Df cvgmDepths_, const unsigned short uScale_, cv::gpu::DevMem2D_<float3> cvgmPts_ )
{
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;

    if (nX >= cvgmPts_.cols || nY >= cvgmPts_.rows)  return;

	float3& pt = cvgmPts_.ptr(nY)[nX];
	const float fDepth = cvgmDepths_.ptr(nY)[nX];

	if( 0.4f < fDepth && fDepth < 4.f ){
		pt.z = fDepth;
		pt.x = ( nX*uScale_  - _aRGBCameraParameter[2] ) * _aRGBCameraParameter[0] * pt.z; //_aRGBCameraParameter[0] is 1.f/fFxRGB_
		pt.y = ( nY*uScale_  - _aRGBCameraParameter[3] ) * _aRGBCameraParameter[1] * pt.z; 
	}
	else {
		pt.x = pt.y = pt.z = pcl::device::numeric_limits<float>::quiet_NaN();
	}
}
void unprojectRGBCVm ( const cv::gpu::GpuMat& cvgmDepths_, 
	const float& fFxRGB_,const float& fFyRGB_,const float& uRGB_, const float& vRGB_, unsigned int uLevel_, 
	cv::gpu::GpuMat* pcvgmPts_ )
{
	unsigned short uScale = 1<< uLevel_;
	pcvgmPts_->setTo(0);
	//constant definition
	size_t sN = sizeof(float) * 4;
	float* const pRGBCameraParameters = (float*) malloc( sN );
	pRGBCameraParameters[0] = 1.f/fFxRGB_;
	pRGBCameraParameters[1] = 1.f/fFyRGB_;
	pRGBCameraParameters[2] = uRGB_;
	pRGBCameraParameters[3] = vRGB_;
	cudaSafeCall( cudaMemcpyToSymbol(_aRGBCameraParameter, pRGBCameraParameters, sN) );
	
	dim3 block (32, 8);
	dim3 grid (cv::gpu::divUp (pcvgmPts_->cols, block.x), cv::gpu::divUp (pcvgmPts_->rows, block.y));
	kernelUnprojectRGBCVmCVm<<<grid, block>>>(cvgmDepths_, uScale, *pcvgmPts_ );
	cudaSafeCall ( cudaGetLastError () );
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelFastNormalEstimation (const cv::gpu::DevMem2D_<float3> cvgmPts_, cv::gpu::DevMem2D_<float3> cvgmNls_ )
{
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;

    if (nX >= cvgmPts_.cols || nY >= cvgmPts_.rows ) return;
	float3& fN = cvgmNls_.ptr(nY)[nX];
	if (nX == cvgmPts_.cols - 1 || nY >= cvgmPts_.rows - 1 ){
		fN.x = fN.y = fN.z = pcl::device::numeric_limits<float>::quiet_NaN();
		return;
	}
	const float3& pt = cvgmPts_.ptr(nY)[nX];
	const float3& pt1= cvgmPts_.ptr(nY)[nX+1]; //right 
	const float3& pt2= cvgmPts_.ptr(nY+1)[nX]; //down

	if(isnan<float>(pt.z) ||isnan<float>(pt1.z) ||isnan<float>(pt2.z) ){
		fN.x = fN.y = fN.z = pcl::device::numeric_limits<float>::quiet_NaN();
		return;
	}//if input or its neighour is NaN,
	float3 v1;
	v1.x = pt1.x-pt.x;
	v1.y = pt1.y-pt.y;
	v1.z = pt1.z-pt.z;
	float3 v2;
	v2.x = pt2.x-pt.x;
	v2.y = pt2.y-pt.y;
	v2.z = pt2.z-pt.z;
	//n = v1 x v2 cross product
	float3 n;
	n.x = v1.y*v2.z - v1.z*v2.y;
	n.y = v1.z*v2.x - v1.x*v2.z;
	n.z = v1.x*v2.y - v1.y*v2.x;
	//normalization
	float norm = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);

	if( norm < 1.0e-10 ) {
		fN.x = fN.y = fN.z = pcl::device::numeric_limits<float>::quiet_NaN();
		return;
	}//set as NaN,
	n.x /= norm;
	n.y /= norm;
	n.z /= norm;

	if( -n.x*pt.x - n.y*pt.y - n.z*pt.z <0 ){ //this gives (0-pt).dot( n ); 
		fN.x = -n.x;
		fN.y = -n.y;
		fN.z = -n.z;
	}//if facing away from the camera
	else{
		fN.x = n.x;
		fN.y = n.y;
		fN.z = n.z;
	}//else
	return;
}

void cudaFastNormalEstimation(const cv::gpu::GpuMat& cvgmPts_, cv::gpu::GpuMat* pcvgmNls_ )
{
	pcvgmNls_->setTo(0);
	dim3 block (32, 8);
	dim3 grid (cv::gpu::divUp (cvgmPts_.cols, block.x), cv::gpu::divUp (cvgmPts_.rows, block.y));
	kernelFastNormalEstimation<<<grid, block>>>(cvgmPts_, *pcvgmNls_ );
	cudaSafeCall ( cudaGetLastError () );
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//GL is always measured in meters
__global__ void kernelNormalSetRotationAxisCVmGL (const cv::gpu::DevMem2D_<float3> cvgmNlsCV_, cv::gpu::DevMem2D_<float3> cvgmAAs_ )
{
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;
    if (nX >= cvgmNlsCV_.cols || nY >= cvgmNlsCV_.rows ) return;
	const float3& Nl = cvgmNlsCV_.ptr(nY)[nX];
	float3& fRA = cvgmAAs_.ptr(nY)[nX];
	if(isnan<float>(Nl.x)||isnan<float>(Nl.y)) {
		fRA.x=fRA.y=fRA.z=pcl::device::numeric_limits<float>::quiet_NaN();
		return;
	}//if is NaN
	//Assuming both vectors v1, v2 are of equal magnitude, 
	//a unique rotation R about the origin exists satisfying R.z-axis = Nl.
	//It is most easily expressed in axis-angle representation.
	//First, normalise the two source vectors, then compute w = z-axis ?Nl (z-axis 0,0,1) Nl (x,-y,-z)
	//Normalise again for the axis: w' = w / |w|
	//Take the arcsine of the magnitude for the angle: 
	//q = asin(|w|)

	//float3 n;
	//n.x = Nl.y; //because of cv-convention
	//n.y = Nl.x;
	//n.z =  0;
	//normalization
	float norm = sqrtf(Nl.x*Nl.x + Nl.y*Nl.y );
	if(norm >0.f){
		fRA.x = Nl.y/norm;
		fRA.y = Nl.x/norm;
		fRA.z = asinf(norm)*180.f/CUDART_PI_F;//convert to degree
	}else{
		fRA.x=fRA.y=fRA.z=pcl::device::numeric_limits<float>::quiet_NaN();
	}

	return;
}//kernelNormalCVSetRotationAxisGL()

void cudaNormalSetRotationAxisCVGL(const cv::gpu::GpuMat& cvgmNlsCV_, cv::gpu::GpuMat* pcvgmAAs_ )
{
	pcvgmAAs_->setTo(0);
	dim3 block (32, 8);
	dim3 grid (cv::gpu::divUp (cvgmNlsCV_.cols, block.x), cv::gpu::divUp (cvgmNlsCV_.rows, block.y));
	kernelNormalSetRotationAxisCVmGL<<<grid, block>>>(cvgmNlsCV_, *pcvgmAAs_ );
	cudaSafeCall ( cudaGetLastError () );
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__constant__ ushort _aNormalHistorgarmParams[3];
__global__ void kernelNormalHistogramKernelCV (const cv::gpu::DevMem2D_<float3> cvgmNlsCV_, const float fNormalBinSize_, cv::gpu::DevMem2D_<short> cvgmBinIdx_ ){

	int nX = threadIdx.x + blockIdx.x * blockDim.x;
    int nY = threadIdx.y + blockIdx.y * blockDim.y;
	if (nX >= cvgmNlsCV_.cols || nY >= cvgmNlsCV_.rows)  return;
	const float3& nl = cvgmNlsCV_.ptr (nY)[nX];
	if( isnan<float>(nl.x)||isnan<float>(nl.y)||isnan<float>(nl.z) ) return;

	ushort usX,usY,usZ;
	usX = __float2int_rd( nl.x / fNormalBinSize_ )+_aNormalHistorgarmParams[0];//0:usSamplesElevationZ_
	usY = __float2int_rd( nl.y / fNormalBinSize_ )+_aNormalHistorgarmParams[0];
	usZ = __float2int_rd(-nl.z / fNormalBinSize_ ); //because of cv-convention
	cvgmBinIdx_.ptr(nY)[nX]= usZ*_aNormalHistorgarmParams[2]+ usY*_aNormalHistorgarmParams[1]+ usX;//2:usLevel 1:usWidth
}//kernelNormalHistogramKernelCV()
void cudaNormalHistogramCV(const cv::gpu::GpuMat& cvgmNlsCV_, const unsigned short usSamplesAzimuth_, const unsigned short usSamplesElevationZ_, 
	const unsigned short usWidth_,const unsigned short usLevel_,  const float fNormalBinSize_, cv::gpu::GpuMat* pcvgmBinIdx_){
	//constant definition
	size_t sN = sizeof(ushort) * 3;
	ushort* const pNormal = (ushort*) malloc( sN );
	pNormal[0] = usSamplesElevationZ_;
	pNormal[1] = usWidth_;
	pNormal[2] = usLevel_;
	cudaSafeCall( cudaMemcpyToSymbol(_aNormalHistorgarmParams, pNormal, sN) );
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(cvgmNlsCV_.cols, block.x), cv::gpu::divUp(cvgmNlsCV_.rows, block.y));
	kernelNormalHistogramKernelCV<<<grid,block>>>(cvgmNlsCV_,fNormalBinSize_,*pcvgmBinIdx_);
	cudaSafeCall ( cudaGetLastError () );
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelScaleDepthCVmCVm (cv::gpu::DevMem2Df cvgmDepth_, const pcl::device::Intr sCameraIntrinsics_)
{
    int nX = threadIdx.x + blockIdx.x * blockDim.x;
    int nY = threadIdx.y + blockIdx.y * blockDim.y;

    if (nX >= cvgmDepth_.cols || nY >= cvgmDepth_.rows)  return;

    float& fDepth = cvgmDepth_.ptr(nY)[nX];
    float fTanX = (nX - sCameraIntrinsics_.cx) / sCameraIntrinsics_.fx;
    float fTanY = (nY - sCameraIntrinsics_.cy) / sCameraIntrinsics_.fy;
    float fSec = sqrtf (fTanX*fTanX + fTanY*fTanY + 1);
    fDepth *= fSec; //meters
}//kernelScaleDepthCVmCVm()
//scaleDepth is to transform raw depth into scaled depth which is the distance from the 3D point to the camera centre
//     *---* 3D point
//     |  / 
//raw  | /scaled depth
//depth|/
//     * camera center
//
void scaleDepthCVmCVm(unsigned short usPyrLevel_, const float fFx_, const float fFy_, const float u_, const float v_, cv::gpu::GpuMat* pcvgmDepth_){
	pcl::device::Intr sCameraIntrinsics(fFx_,fFy_,u_,v_);
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(pcvgmDepth_->cols, block.x), cv::gpu::divUp(pcvgmDepth_->rows, block.y));
	kernelScaleDepthCVmCVm<<< grid,block >>>(*pcvgmDepth_,sCameraIntrinsics(usPyrLevel_));
	cudaSafeCall ( cudaGetLastError () );
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__constant__ float _aRwTrans[9];//row major 
__constant__ float _aTw[3]; 
__global__ void kernelTransformLocalToWorldCVCV(cv::gpu::DevMem2D_<float3> cvgmPts_, cv::gpu::DevMem2D_<float3> cvgmNls_){ 
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
    int nY = threadIdx.y + blockIdx.y * blockDim.y;
    if (nX >= cvgmPts_.cols || nY >= cvgmPts_.rows)  return;
	//convert Pts
	float3& Pt = cvgmPts_.ptr(nY)[nX];
	float3 PtTmp; 
	//PtTmp = X_c - Tw
	PtTmp.x = Pt.x - _aTw[0];
	PtTmp.y = Pt.y - _aTw[1];
	PtTmp.z = Pt.z - _aTw[2];
	//Pt = RwTrans * PtTmp
	Pt.x = _aRwTrans[0]*PtTmp.x + _aRwTrans[1]*PtTmp.y + _aRwTrans[2]*PtTmp.z;
	Pt.y = _aRwTrans[3]*PtTmp.x + _aRwTrans[4]*PtTmp.y + _aRwTrans[5]*PtTmp.z;
	Pt.z = _aRwTrans[6]*PtTmp.x + _aRwTrans[7]*PtTmp.y + _aRwTrans[8]*PtTmp.z;
	//convert Nls
	float3& Nl = cvgmNls_.ptr(nY)[nX];
	float3 NlTmp;
	//Nlw = RwTrans*Nlc
	NlTmp.x = _aRwTrans[0]*Nl.x + _aRwTrans[1]*Nl.y + _aRwTrans[2]*Nl.z;
	NlTmp.y = _aRwTrans[3]*Nl.x + _aRwTrans[4]*Nl.y + _aRwTrans[5]*Nl.z;
	NlTmp.z = _aRwTrans[6]*Nl.x + _aRwTrans[7]*Nl.y + _aRwTrans[8]*Nl.z;
	Nl = NlTmp;
}//kernelTransformLocalToWorld()
void transformLocalToWorldCVCV(const float* pRw_/*col major*/, const float* pTw_, cv::gpu::GpuMat* pcvgmPts_, cv::gpu::GpuMat* pcvgmNls_){
	size_t sN1 = sizeof(float) * 9;
	cudaSafeCall( cudaMemcpyToSymbol(_aRwTrans, pRw_, sN1) );
	size_t sN2 = sizeof(float) * 3;
	cudaSafeCall( cudaMemcpyToSymbol(_aTw, pTw_, sN2) );
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(pcvgmPts_->cols, block.x), cv::gpu::divUp(pcvgmPts_->rows, block.y));
	kernelTransformLocalToWorldCVCV<<<grid,block>>>(*pcvgmPts_,*pcvgmNls_);
	cudaSafeCall ( cudaGetLastError () );
}//transformLocalToWorld()
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//check if the normal is NaN whether the corresponding is vertex NaN;
__global__ void kernelCheck(const cv::gpu::DevMem2D_<float3> cvgmPts_,const cv::gpu::DevMem2D_<float3> cvgmNls_, cv::gpu::DevMem2D_<short> cvgmResults_){
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
    int nY = threadIdx.y + blockIdx.y * blockDim.y;
    if (nX >= cvgmPts_.cols || nY >= cvgmPts_.rows)  return;

	const float3& pt = cvgmPts_.ptr(nY)[nX];
	const float3& nl = cvgmNls_.ptr(nY)[nX];
	//if(pt.x!=pt.x && nl.x == nl.x)
	if(nl.x!=nl.x/* && pt.x == pt.x*/)
		cvgmResults_.ptr(nY)[nX] = 1;
}
void checkNVMap(const cv::gpu::GpuMat& cvgmPts_, const cv::gpu::GpuMat& cvgmNls_, cv::gpu::GpuMat* pcvgmResults_){
	pcvgmResults_->create(cvgmPts_.cols,cvgmPts_.rows,CV_16SC1);
	pcvgmResults_->setTo(0);
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(cvgmPts_.cols, block.x), cv::gpu::divUp(cvgmPts_.rows, block.y));
	kernelCheck<<<grid,block>>>( cvgmPts_,cvgmNls_,(*pcvgmResults_) );
	cudaSafeCall ( cudaGetLastError () );
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<bool normalize>
__global__ void kernelResizeMap (const cv::gpu::DevMem2D_<float3> cvgmSrc_, cv::gpu::DevMem2D_<float3> cvgmDst_)
{
	using namespace pcl::device;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= cvgmDst_.cols || y >= cvgmDst_.rows) return;

    float3 qnan; qnan.x = qnan.y = qnan.z = pcl::device::numeric_limits<float>::quiet_NaN ();

    int xs = x * 2;
    int ys = y * 2;

    float3 x00 = cvgmSrc_.ptr (ys + 0)[xs + 0];
    float3 x01 = cvgmSrc_.ptr (ys + 0)[xs + 1];
    float3 x10 = cvgmSrc_.ptr (ys + 1)[xs + 0];
    float3 x11 = cvgmSrc_.ptr (ys + 1)[xs + 1];

    if (isnan (x00.x) || isnan (x01.x) || isnan (x10.x) || isnan (x11.x))
    {
		cvgmDst_.ptr (y)[x] = qnan;
		return;
    }
    else
    {
		float3 n;

		n = (x00 + x01 + x10 + x11) / 4;

		if (normalize)
			n = normalized (n);

		cvgmDst_.ptr (y)[x] = n;
    }
}//kernelResizeMap()

void resizeMap (bool bNormalize_, const cv::gpu::GpuMat& cvgmSrc_, cv::gpu::GpuMat* pcvgmDst_ )
{
    int in_cols = cvgmSrc_.cols;
    int in_rows = cvgmSrc_.rows;

    int out_cols = in_cols / 2;
    int out_rows = in_rows / 2;

    pcvgmDst_->create (out_rows, out_cols,cvgmSrc_.type());

    dim3 block (32, 8);
    dim3 grid (cv::gpu::divUp (out_cols, block.x), cv::gpu::divUp (out_rows, block.y));
	if(bNormalize_)
		kernelResizeMap<true><<<grid, block>>>(cvgmSrc_, *pcvgmDst_);
	else
		kernelResizeMap<false><<<grid, block>>>(cvgmSrc_, *pcvgmDst_);
	cudaSafeCall ( cudaGetLastError () );
    //cudaSafeCall (cudaDeviceSynchronize ());
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelRgb2RGBA(const cv::gpu::DevMem2D_<uchar3> cvgmRGB_, uchar uA_, cv::gpu::DevMem2D_<uchar4> cvgmRGBA_){
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
    int nY = threadIdx.y + blockIdx.y * blockDim.y;
    if (nX >= cvgmRGB_.cols || nY >= cvgmRGB_.rows)  return;
	//convert Pts
	const uchar3& RGB = cvgmRGB_.ptr(nY)[nX];
	uchar4& RGBA = cvgmRGBA_.ptr(nY)[nX];
	RGBA.w = RGB.x;
	RGBA.x = RGB.y;
	RGBA.y = RGB.z;
	RGBA.z = uA_;
	return;
}
__global__ void kernelRgb2RGBAfloat(const cv::gpu::DevMem2D_<uchar3> cvgmRGB_, uchar uA_, cv::gpu::DevMem2D_<float4> cvgmRGBA_){
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
    int nY = threadIdx.y + blockIdx.y * blockDim.y;
    if (nX >= cvgmRGB_.cols || nY >= cvgmRGB_.rows)  return;
	//convert Pts
	const uchar3& RGB = cvgmRGB_.ptr(nY)[nX];
	float4& RGBA = cvgmRGBA_.ptr(nY)[nX];
	RGBA.w = RGB.x/255.f;
	RGBA.x = RGB.y/255.f;
	RGBA.y = RGB.z/255.f;
	RGBA.z = uA_/255.f;
	return;
}
void rgb2RGBA(const cv::gpu::GpuMat& cvgmRGB_, const uchar uA_, cv::gpu::GpuMat* pcvgmRGBA_){
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(cvgmRGB_.cols, block.x), cv::gpu::divUp(cvgmRGB_.rows, block.y));
	//different type
	if(CV_8UC4 == pcvgmRGBA_->type() )
	{	
		kernelRgb2RGBA<<<grid,block>>>(cvgmRGB_,uA_,*pcvgmRGBA_);
	}
	else if(CV_32FC4 == pcvgmRGBA_->type() )
	{
		kernelRgb2RGBAfloat<<<grid,block>>>(cvgmRGB_,uA_,*pcvgmRGBA_);
	}
	cudaSafeCall ( cudaGetLastError () );
}//rgb2RGBA()
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelBoundaryDetector(const float fThreshold, const cv::gpu::DevMem2D_<float3> cvgmPt_, const cv::gpu::DevMem2D_<float3> cvgmNl_, cv::gpu::DevMem2D_<uchar3> cvgmRGB_){
	using namespace pcl::device;
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
    int nY = threadIdx.y + blockIdx.y * blockDim.y;
    if (nX ==0 || nY == 0 || nX >= cvgmRGB_.cols-1 || nY >= cvgmRGB_.rows-1)  return;

	const float3& Pt = cvgmPt_.ptr(nY)[nX];
	
	if(isnan<float>(Pt.x)) return;

	const float3& Nl = cvgmNl_.ptr(nY)[nX];
	uchar3& RGB= cvgmRGB_.ptr(nY)[nX];

	short sCount=0;
	float fDistance;
	float3 PtNeighbour;
	PtNeighbour = cvgmPt_.ptr(nY)[nX-1];//Left
	fDistance = pcl::device::norm(PtNeighbour - Pt);	if(fDistance>fThreshold||isnan<float>(fDistance)) sCount++;
	PtNeighbour = cvgmPt_.ptr(nY)[nX+1];//Right
	fDistance = pcl::device::norm(PtNeighbour - Pt);	if(fDistance>fThreshold||isnan<float>(fDistance)) sCount++;
	PtNeighbour = cvgmPt_.ptr(nY-1)[nX-1];//UL
	fDistance = pcl::device::norm(PtNeighbour - Pt);	if(fDistance>fThreshold||isnan<float>(fDistance)) sCount++;
	PtNeighbour = cvgmPt_.ptr(nY-1)[nX+1];//UR
	fDistance = pcl::device::norm(PtNeighbour - Pt);	if(fDistance>fThreshold||isnan<float>(fDistance)) sCount++;
	PtNeighbour = cvgmPt_.ptr(nY-1)[nX];//Up
	fDistance = pcl::device::norm(PtNeighbour - Pt);	if(fDistance>fThreshold||isnan<float>(fDistance)) sCount++;
	PtNeighbour = cvgmPt_.ptr(nY+1)[nX-1];//DL
	fDistance = pcl::device::norm(PtNeighbour - Pt);	if(fDistance>fThreshold||isnan<float>(fDistance)) sCount++;
	PtNeighbour = cvgmPt_.ptr(nY+1)[nX+1];//DR
	fDistance = pcl::device::norm(PtNeighbour - Pt);	if(fDistance>fThreshold||isnan<float>(fDistance)) sCount++;
	PtNeighbour = cvgmPt_.ptr(nY+1)[nX];//Down
	fDistance = pcl::device::norm(PtNeighbour - Pt);	if(fDistance>fThreshold||isnan<float>(fDistance)) sCount++;

	if(sCount>2&&sCount<8){//if it is a border pixel
		RGB = RGB*0.5 + make_uchar3(255,0,0)*0.5;
	}
	return;
}
void boundaryDetector(const float fThreshold_, const cv::gpu::GpuMat& cvgmPt_, const cv::gpu::GpuMat& cvgmNl_, cv::gpu::GpuMat* pcvgmRGB_){
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(pcvgmRGB_->cols, block.x), cv::gpu::divUp(pcvgmRGB_->rows, block.y));
	kernelBoundaryDetector<<<grid,block>>>(fThreshold_,cvgmPt_,cvgmNl_,*pcvgmRGB_);
	cudaSafeCall ( cudaGetLastError () );
}//boundaryDetector()

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelShapeClassifier(const float fThreshold, const cv::gpu::DevMem2D_<float3> cvgmPt_, const cv::gpu::DevMem2D_<float3> cvgmNl_, cv::gpu::DevMem2D_<uchar3> cvgmRGB_){
	using namespace pcl::device;
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
    int nY = threadIdx.y + blockIdx.y * blockDim.y;
    if (nX ==0 || nY == 0 || nX >= cvgmRGB_.cols-1 || nY >= cvgmRGB_.rows-1)  return;

	const float3& Pt = cvgmPt_.ptr(nY)[nX];
	
	if(isnan<float>(Pt.x)) return;

	const float3& Nl = cvgmNl_.ptr(nY)[nX];
	uchar3& RGB= cvgmRGB_.ptr(nY)[nX];
	
	const float3& Left= cvgmPt_.ptr(nY)[nX-1];
	const float3& LeftN=cvgmNl_.ptr(nY)[nX-1];

	const float3& Right= cvgmPt_.ptr(nY)[nX+1];
	const float3& Up= cvgmPt_.ptr(nY-1)[nX];
	const float3& Down= cvgmPt_.ptr(nY+1)[nX-1];
	//line-line intersection
	//http://en.wikipedia.org/wiki/Line-line_intersection
	float3 OutProduct[3];
	outProductSelf<float3>( LeftN, OutProduct);
	
	{//if it is a border pixel
		RGB = RGB*0.5 + make_uchar3(0,255,0)*0.5;
	}
	return;
}//kernelShapeClassifier()
void shapeClassifier(const float fThreshold_, const cv::gpu::GpuMat& cvgmPt_, const cv::gpu::GpuMat& cvgmNl_, cv::gpu::GpuMat* pcvgmRGB_){
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(pcvgmRGB_->cols, block.x), cv::gpu::divUp(pcvgmRGB_->rows, block.y));
	kernelShapeClassifier<<<grid,block>>>(fThreshold_,cvgmPt_,cvgmNl_,*pcvgmRGB_);
	cudaSafeCall ( cudaGetLastError () );
}//boundaryDetector()

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelConvertZValue2Depth(const cv::gpu::DevMem2D_<float> cvgmZValue_, const float fNear_,  const float fFar_, cv::gpu::DevMem2D_<float> cvgmDepth_){
	using namespace pcl::device;
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
    int nY = threadIdx.y + blockIdx.y * blockDim.y;
    if (nX >= cvgmZValue_.cols || nY >= cvgmZValue_.rows)  return;

	const float& fZ = cvgmZValue_.ptr(nY)[nX];
	float& fDepth = cvgmDepth_.ptr(cvgmZValue_.rows-1-nY)[nX];

	/*if( abs(fZ-1.f) < 0.01f ) 
		fDepth = pcl::device::numeric_limits<float>::quiet_NaN();
	else*/
		fDepth = 2*fFar_*fNear_*1000.f / (fFar_ + fNear_ - (fFar_ - fNear_)*(2*fZ -1));
	//http://www.songho.ca/opengl/gl_projectionmatrix.html
		//fDepth = (fZ*fRange + fNear_)*1000.f;


	return;
}//kernelConvertZValue2Depth()
void cudaConvertZValue2Depth(const cv::gpu::GpuMat& cvgmZValue_, float fNear_, float fFar_, cv::gpu::GpuMat* pcvgmDepth_){
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(cvgmZValue_.cols, block.x), cv::gpu::divUp(cvgmZValue_.rows, block.y));
	kernelConvertZValue2Depth<<<grid,block>>>(cvgmZValue_,fNear_,fFar_,*pcvgmDepth_);
	cudaSafeCall ( cudaGetLastError () );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kernelConvertGL2CV(const cv::gpu::DevMem2D_<uchar3> cvgmRGB_, cv::gpu::DevMem2D_<uchar3> cvgmUndistRGB_){
	using namespace pcl::device;
	int nX = threadIdx.x + blockIdx.x * blockDim.x;
    int nY = threadIdx.y + blockIdx.y * blockDim.y;
    if (nX >= cvgmRGB_.cols || nY >= cvgmRGB_.rows)  return;

	const uchar3& fZ = cvgmRGB_.ptr(nY)[nX];
	cvgmUndistRGB_.ptr(cvgmRGB_.rows-1-nY)[nX] = fZ;

	return;
}//kernelConvertZValue2Depth()
void cudaConvertGL2CV(const cv::gpu::GpuMat cvgmRGB_, cv::gpu::GpuMat* pcvgmUndistRGB_){

	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(cvgmRGB_.cols, block.x), cv::gpu::divUp(cvgmRGB_.rows, block.y));
	kernelConvertGL2CV<<<grid,block>>>(cvgmRGB_,*pcvgmUndistRGB_);
	cudaSafeCall ( cudaGetLastError () );
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class T>
void cudaConvert(const cv::gpu::DevMem2D_<float3> cvgmSrc_, cv::gpu::DevMem2D_<float3> cvgmDst_){

}
}//device
}//btl
