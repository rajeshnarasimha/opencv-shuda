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

__global__ void kernelInverse(cv::gpu::DevMem2Df cvgmDepth_, cv::gpu::DevMem2Df cvgmDisparity_)
{
    const int nX = blockDim.x * blockIdx.x + threadIdx.x;
    const int nY = blockDim.y * blockIdx.y + threadIdx.y;

	if(fabsf(cvgmDepth_.ptr(nY)[nX]) > 1.0e-38 )
		cvgmDisparity_.ptr(nY)[nX] = 1.f/cvgmDepth_.ptr(nY)[nX];
	else
		cvgmDisparity_.ptr(nY)[nX] = 1.0e+38;
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

__global__ void kernelUnprojectIR(cv::gpu::DevMem2D_<unsigned short> cvgmDepth_,
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
	double* pIRCameraParameters = (double*) malloc( sN );
	*pIRCameraParameters++  = dFxIR_;
	*pIRCameraParameters++  = dFyIR_;
	*pIRCameraParameters++  = uIR_;
	*pIRCameraParameters    = vIR_;
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


}//cuda_util
}//btl
