/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/devmem2d.hpp>
#include "common.hpp" //copied from opencv

namespace btl
{
namespace cuda_util
{
__global__ void kernelTestFloat3(const cv::gpu::DevMem2D_<float3> cvgmIn_, cv::gpu::DevMem2D_<float3> cvgmOut_)
{
	const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;

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
}
//depth to disparity
__global__ void kernelInverse(const cv::gpu::DevMem2Df cvgmIn_, cv::gpu::DevMem2Df cvgmOut_)
{
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;

	if(fabsf(cvgmIn_.ptr(nY)[nX]) > 1.0e-38 )
		cvgmOut_.ptr(nY)[nX] = 1.f/cvgmIn_.ptr(nY)[nX];
	else
		cvgmOut_.ptr(nY)[nX] = 1.0e+38;
}

void cudaDepth2Disparity( const cv::gpu::GpuMat& cvgmDepth_, cv::gpu::GpuMat* pcvgmDisparity_ )
{
	pcvgmDisparity_->create(cvgmDepth_.size(),CV_32F);
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(cvgmDepth_.cols, block.x), cv::gpu::divUp(cvgmDepth_.rows, block.y));
	//run kernel
	kernelInverse<<<grid,block>>>( cvgmDepth_,*pcvgmDisparity_ );
}

void cudaDisparity2Depth( const cv::gpu::GpuMat& cvgmDisparity_, cv::gpu::GpuMat* pcvgmDepth_ )
{
	pcvgmDepth_->create(cvgmDisparity_.size(),CV_32F);
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(cvgmDisparity_.cols, block.x), cv::gpu::divUp(cvgmDisparity_.rows, block.y));
	//run kernel
	kernelInverse<<<grid,block>>>( cvgmDisparity_,*pcvgmDepth_ );
}

//global constant used by kernelUnprojectIR() and cudaUnProjectIR()
__constant__ double _aIRCameraParameter[4];// f_x, f_y, u, v for IR camera; constant memory declaration

__global__ void kernelUnprojectIR(const cv::gpu::DevMem2D_<unsigned short> cvgmDepth_,
	cv::gpu::DevMem2D_<float3> cvgmIRWorld_)
{
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;

	if (nX < cvgmIRWorld_.cols && nY < cvgmIRWorld_.rows)
    {
		float3& temp = cvgmIRWorld_.ptr(nY)[nX];
        temp.z = (cvgmDepth_.ptr(nY)[nX] + 5) /1000.f;//convert to meter z 5 million meter is added according to experience. as the OpenNI
		//coordinate system is defined w.r.t. the camera plane which is 0.5 centimeters in front of the camera center
		temp.x = (nX - _aIRCameraParameter[2]) / _aIRCameraParameter[0] * temp.z;
		temp.y = (nY - _aIRCameraParameter[3]) / _aIRCameraParameter[1] * temp.z;
    }
	return;
}

void cudaUnProjectIR(const cv::gpu::GpuMat& cvgmDepth_ ,
	const double& dFxIR_, const double& dFyIR_, const double& uIR_, const double& vIR_, 
	cv::gpu::GpuMat* pcvgmIRWorld_ )
{
	//constant definition
	size_t sN = sizeof(double) * 4;
	double* const pIRCameraParameters = (double*) malloc( sN );
	pIRCameraParameters[0] = dFxIR_;
	pIRCameraParameters[1] = dFyIR_;
	pIRCameraParameters[2] = uIR_;
	pIRCameraParameters[3] = vIR_;
	cudaSafeCall( cudaMemcpyToSymbol(_aIRCameraParameter, pIRCameraParameters, sN) );
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(cvgmDepth_.cols, block.x), cv::gpu::divUp(cvgmDepth_.rows, block.y));
	//run kernel
    kernelUnprojectIR<<<grid,block>>>( cvgmDepth_,*pcvgmIRWorld_ );
	//release temporary pointers
	free(pIRCameraParameters);
	return;
}
//global constant used by kernelUnprojectIR() and cudaTransformIR2RGB()
__constant__ double _aR[9];// f_x, f_y, u, v for IR camera; constant memory declaration
__constant__ double _aRT[3];

__global__ void kernelTransformIR2RGB(const cv::gpu::DevMem2D_<float3> cvgmIRWorld_, cv::gpu::DevMem2D_<float3> cvgmRGBWorld_)
{
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;

	if (nX < cvgmRGBWorld_.cols && nY < cvgmRGBWorld_.rows)
    {
		float3& rgbWorld = cvgmRGBWorld_.ptr(nY)[nX];
		const float3& irWorld  = cvgmIRWorld_ .ptr(nY)[nX];
		if( fabs( irWorld.z ) < 0.0001 )
		{
			rgbWorld.x = rgbWorld.y = rgbWorld.z = 0;
		}
		else
		{
			rgbWorld.x = _aR[0] * irWorld.x + _aR[1] * irWorld.y + _aR[2] * irWorld.z - _aRT[0];
			rgbWorld.y = _aR[3] * irWorld.x + _aR[4] * irWorld.y + _aR[5] * irWorld.z - _aRT[1];
			rgbWorld.z = _aR[6] * irWorld.x + _aR[7] * irWorld.y + _aR[8] * irWorld.z - _aRT[2];
		}
    }
	return;
}
void cudaTransformIR2RGB(const cv::gpu::GpuMat& cvgmIRWorld_, const double* aR_, const double* aRT_, cv::gpu::GpuMat* pcvgmRGBWorld_)
{
	cudaSafeCall( cudaMemcpyToSymbol(_aR,  aR_,  9*sizeof(double)) );
	cudaSafeCall( cudaMemcpyToSymbol(_aRT, aRT_, 3*sizeof(double)) );
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(pcvgmRGBWorld_->cols, block.x), cv::gpu::divUp(pcvgmRGBWorld_->rows, block.y));
	//run kernel
    kernelTransformIR2RGB<<<grid,block>>>( cvgmIRWorld_,*pcvgmRGBWorld_ );
	return;
}
//global constant used by kernelProjectRGB() and cudaProjectRGB()
__constant__ double _aRGBCameraParameter[4];

__global__ void kernelProjectRGB(const cv::gpu::DevMem2D_<float3> cvgmRGBWorld_, cv::gpu::DevMem2Df cvgmAligned_)
{
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;

	if (nX < cvgmRGBWorld_.cols && nY < cvgmRGBWorld_.rows)
    {
		const float3& rgbWorld = cvgmRGBWorld_.ptr(nY)[nX];
		if( fabsf( rgbWorld.z ) > 0.000001 )
		{
			// get 2D image projection in RGB image of the XYZ in the world
			int nXAligned = __float2int_rn( _aRGBCameraParameter[0] * rgbWorld.x / rgbWorld.z + _aRGBCameraParameter[2] );
			int nYAligned = __float2int_rn( _aRGBCameraParameter[1] * rgbWorld.y / rgbWorld.z + _aRGBCameraParameter[3] );
			if ( nXAligned >= 0 && nXAligned < cvgmRGBWorld_.cols && nYAligned >= 0 && nYAligned < cvgmRGBWorld_.rows )
			{
				cvgmAligned_.ptr(nYAligned)[nXAligned] = rgbWorld.z*1000;
				cvgmAligned_.ptr(nY)[nX] = rgbWorld.z*1000;
			}
		}
    }

	return;
}
void cudaProjectRGB(const cv::gpu::GpuMat& cvgmRGBWorld_, 
	const double& dFxRGB_, const double& dFyRGB_, const double& uRGB_, const double& vRGB_, 
	cv::gpu::GpuMat* pcvgmAligned_ )
{
		//constant definition
	size_t sN = sizeof(double) * 4;
	double* const pRGBCameraParameters = (double*) malloc( sN );
	pRGBCameraParameters[0] = dFxRGB_;
	pRGBCameraParameters[1] = dFyRGB_;
	pRGBCameraParameters[2] = uRGB_;
	pRGBCameraParameters[3] = vRGB_;
	cudaSafeCall( cudaMemcpyToSymbol(_aRGBCameraParameter, pRGBCameraParameters, sN) );
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(cvgmRGBWorld_.cols, block.x), cv::gpu::divUp(cvgmRGBWorld_.rows, block.y));
	//run kernel
    kernelProjectRGB<<<grid,block>>>( cvgmRGBWorld_,*pcvgmAligned_ );
	//release temporary pointers
	free(pRGBCameraParameters);
	return;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//const float sigma_color = 30;     //in mm
//const float sigma_space = 4.5;     // in pixels
__constant__ float _aSigma2InvHalf[2]; //sigma_space2_inv_half,sigma_color2_inv_half

__global__ void bilateralKernel (const cv::gpu::DevMem2Df src, cv::gpu::DevMem2Df dst )
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= src.cols || y >= src.rows)  return;

    const int R = 6;       //static_cast<int>(sigma_space * 1.5);
    const int D = R * 2 + 1;

    int value = src.ptr (y)[x];

    int tx = min (x - D / 2 + D, src.cols - 1);
    int ty = min (y - D / 2 + D, src.rows - 1);

    float sum1 = 0;
    float sum2 = 0;

    for (int cy = max (y - D / 2, 0); cy < ty; ++cy)
    for (int cx = max (x - D / 2, 0); cx < tx; ++cx)
    {
        int tmp = src.ptr (cy)[cx];

        float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        float color2 = (value - tmp) * (value - tmp);

        float weight = __expf (-(space2 * _aSigma2InvHalf[0] + color2 * _aSigma2InvHalf[1]) );

        sum1 += tmp * weight;
        sum2 += weight;
    }

    dst.ptr (y)[x] = __float2int_rn (sum1 / sum2);
	return;
}

void cudaBilateralFiltering(const cv::gpu::GpuMat& cvgmSrc_, const float& fSigmaSpace_, const float& fSigmaColor_, cv::gpu::GpuMat* pcvgmDst_ )
{
		//constant definition
	size_t sN = sizeof(float) * 2;
	double* const pSigma = (double*) malloc( sN );
	pSigma[0] = 0.5f / (fSigmaSpace_ * fSigmaSpace_);
	pSigma[1] = 0.5f / (fSigmaColor_ * fSigmaColor_);
	cudaSafeCall( cudaMemcpyToSymbol(_aSigma2InvHalf, pSigma, sN) );
	//define grid and block
	dim3 block(32, 8);
    dim3 grid(cv::gpu::divUp(cvgmSrc_.cols, block.x), cv::gpu::divUp(cvgmSrc_.rows, block.y));
	//run kernel
    kernelProjectRGB<<<grid,block>>>( cvgmSrc_,*pcvgmDst_ );
	//release temporary pointers
	free(pSigma);
	return;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}//cuda_util
}//btl
