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

namespace btl
{
namespace cuda_util
{

__global__ void kernelDepth2Disparity( const float *pDepth_, float *pDisparity_ ) {
    int nX = blockIdx.x;
	int nY = blockIdx.y; // built-in variable defined by CUDA.
	int nIdx = nX + nY*gridDim.x;
                          // cuda allow 2D blocks
	if(fabsf(pDepth_[nIdx]) > 1.0e-38 )
		pDisparity_[nIdx] = 1.f/pDepth_[nIdx] ;
	else
		pDisparity_[nIdx] = 1.0e+38;
}

__global__ void kernelDisparity2Depth( const float *pDisparity_, float *pDepth_ ) {
    int nX = blockIdx.x;
	int nY = blockIdx.y; // built-in variable defined by CUDA.
	int nIdx = nX + nY*gridDim.x;
                          // cuda allow 2D blocks
	if(fabsf(pDisparity_[nIdx]) > 1.0e-38 )
		pDepth_[nIdx] = 1.f/pDisparity_[nIdx] ;
	else
		pDepth_[nIdx] = 1.0e+38;
}


int cudaDepth2Disparity( const float* pDepth_, const int& nRow_, const int& nCol_, float *pDisparity_  )
{
	int nSize = nRow_*nCol_;
    // handler of GPU memory
    float *dev_pDepth, *dev_pDisparity;

    // allocate the memory on the GPU
	cudaMalloc( (void**)&dev_pDepth, nSize * sizeof(float) );
	cudaMalloc( (void**)&dev_pDisparity, nSize * sizeof(float) );

    // copy the arrays 'a' and 'b' to the GPU
	cudaMemcpy( dev_pDepth,     pDepth_,     nSize * sizeof(float), cudaMemcpyHostToDevice );

	dim3    blocks(nCol_,nRow_);
    kernelDepth2Disparity<<<blocks,1>>>( dev_pDepth, dev_pDisparity );

    // copy the array 'c' back from the GPU to the CPU
	cudaMemcpy( pDisparity_, dev_pDisparity, nSize * sizeof(float), cudaMemcpyDeviceToHost );

    // free the memory we allocated on the GPU
	cudaFree( dev_pDepth );
	cudaFree( dev_pDisparity );
   
    return 0;
}

int cudaDisparity2Depth( const float* pDisparity_, const int& nRow_, const int& nCol_, float *pDepth_  )
{
	int nSize = nRow_*nCol_;
    // handler of GPU memory
    float *dev_pDepth, *dev_pDisparity;

    // allocate the memory on the GPU
	cudaMalloc( (void**)&dev_pDepth, nSize * sizeof(float) );
	cudaMalloc( (void**)&dev_pDisparity, nSize * sizeof(float) );

    // copy the arrays 'a' and 'b' to the GPU
	cudaMemcpy( dev_pDisparity, pDisparity_, nSize * sizeof(float), cudaMemcpyHostToDevice );

	dim3    blocks(nCol_,nRow_);
	kernelDisparity2Depth<<<blocks,1>>>( dev_pDisparity, dev_pDepth );

    // copy the array 'c' back from the GPU to the CPU
	cudaMemcpy( pDepth_, dev_pDepth, nSize * sizeof(float), cudaMemcpyDeviceToHost );

    // free the memory we allocated on the GPU
	cudaFree( dev_pDepth );
	cudaFree( dev_pDisparity );
   
    return 0;
}

}//cuda_util
}//btl
