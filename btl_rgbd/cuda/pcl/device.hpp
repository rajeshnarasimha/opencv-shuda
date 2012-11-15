/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef PCL_GPU_KINFU_DEVICE_HPP_
#define PCL_GPU_KINFU_DEVICE_HPP_

#include "limits.hpp"
#include "vector_math.hpp"

#include "internal.h"
#include "block.hpp"

using namespace pcl::device;

namespace btl{ namespace device{

	template<typename T>
	__device__ __forceinline__ bool isnan(T t){	return t!=t;}
	template<typename T>
	__device__ __forceinline__ void outProductSelf(const T V_, T* pMRm_){
		pMRm_[0].x = V_.x*V_.x;
		pMRm_[0].y = pMRm_[1].x = V_.x*V_.y;
		pMRm_[0].z = pMRm_[2].x = V_.x*V_.z;
		pMRm_[1].y = V_.y*V_.y;
		pMRm_[1].z = pMRm_[2].y = V_.y*V_.z;
		pMRm_[2].z = V_.z*V_.z;
	}
	__device__ __forceinline__ void setIdentity(float fScalar_, float3* pMRm_){
		pMRm_[0].x = pMRm_[1].y = pMRm_[2].z = fScalar_;
		pMRm_[0].y = pMRm_[0].z = pMRm_[1].x = pMRm_[1].z = pMRm_[2].x = pMRm_[2].y = 0;
	}

}//device
}//btl

namespace pcl
{
namespace device
{
//////////////////////////////////////////////////////////////////////////////////////
/// for old format
//Tsdf fixed point divisor (if old format is enabled)
const int DIVISOR =  32767;     // SHRT_MAX; //30000; //

//should be multiple of 32
//enum { VOLUME_X = 512, VOLUME_Y = 512, VOLUME_Z = 512 };


const float VOLUME_SIZE = 3.0f; // in meters
 
#define INV_DIV 3.051850947599719e-5f

__device__ __forceinline__ void
pack_tsdf (float tsdf, int weight, short2& value)
{
    int fixedp = max (-DIVISOR, min (DIVISOR, __float2int_rz (tsdf * DIVISOR)));
    //int fixedp = __float2int_rz(tsdf * DIVISOR);
    value = make_short2 (fixedp, weight);
}

__device__ __forceinline__ void
unpack_tsdf (short2 value, float& tsdf, int& weight)
{
    weight = value.y;
    tsdf = __int2float_rn (value.x) / DIVISOR;   //*/ * INV_DIV;
}

__device__ __forceinline__ float
unpack_tsdf (short2 value)
{
    return static_cast<float>(value.x) / DIVISOR;    //*/ * INV_DIV;
}

//////////////////////////////////////////////////////////////////////////////////////
/// for half float
__device__ __forceinline__ void
pack_tsdf (float tsdf, int weight, ushort2& value)
{
    value = make_ushort2 (__float2half_rn (tsdf), weight);
}

__device__ __forceinline__ void
unpack_tsdf (ushort2 value, float& tsdf, int& weight)
{
    tsdf = __half2float (value.x);
    weight = value.y;
}

__device__ __forceinline__ float
unpack_tsdf (ushort2 value)
{
    return __half2float (value.x);
}

__device__ __forceinline__ float3
operator* (const Mat33& m, const float3& vec)
{
    return make_float3 (dot (m.data[0], vec), dot (m.data[1], vec), dot (m.data[2], vec));
}

struct Warp
{
	enum
	{
		LOG_WARP_SIZE = 5,
		WARP_SIZE     = 1 << LOG_WARP_SIZE,
		STRIDE        = WARP_SIZE
	};

	/** \brief Returns the warp lane ID of the calling thread. */
	static __device__ __forceinline__ unsigned int 
		laneId()
	{
		unsigned int ret;
		asm("mov.u32 %0, %laneid;" : "=r"(ret) );
		return ret;
	}

	static __device__ __forceinline__ unsigned int id()
	{
		int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
		return tid >> LOG_WARP_SIZE;
	}

	static __device__ __forceinline__ 
		int laneMaskLt()
	{
#if (__CUDA_ARCH__ >= 200)
		unsigned int ret;
		asm("mov.u32 %0, %lanemask_lt;" : "=r"(ret) );
		return ret;
#else
		return 0xFFFFFFFF >> (32 - laneId());
#endif
	}

	static __device__ __forceinline__ int binaryExclScan(int ballot_mask)
	{
		return __popc(Warp::laneMaskLt() & ballot_mask);
	}   
};


struct Emulation
{        
	static __device__ __forceinline__ int
		warp_reduce ( volatile int *ptr , const unsigned int tid)
	{
		const unsigned int lane = tid & 31; // index of thread in warp (0..31)        

		if (lane < 16)
		{				
			int partial = ptr[tid];

			ptr[tid] = partial = partial + ptr[tid + 16];
			ptr[tid] = partial = partial + ptr[tid + 8];
			ptr[tid] = partial = partial + ptr[tid + 4];
			ptr[tid] = partial = partial + ptr[tid + 2];
			ptr[tid] = partial = partial + ptr[tid + 1];            
		}
		return ptr[tid - lane];
	}

	static __forceinline__ __device__ int 
		Ballot(int predicate, volatile int* cta_buffer)
	{
#if __CUDA_ARCH__ >= 200
		(void)cta_buffer;
		return __ballot(predicate);
#else
		int tid = pcl::device::Block::flattenedThreadId();				
		cta_buffer[tid] = predicate ? (1 << (tid & 31)) : 0;
		return warp_reduce(cta_buffer, tid);
#endif
	}

	static __forceinline__ __device__ bool
		All(int predicate, volatile int* cta_buffer)
	{
#if __CUDA_ARCH__ >= 200
		(void)cta_buffer;
		return __all(predicate);
#else
		int tid = Block::flattenedThreadId();				
		cta_buffer[tid] = predicate ? 1 : 0;
		return warp_reduce(cta_buffer, tid) == 32;
#endif
	}
};


////////////////////////////////////////////////////////////////////////////////////////
///// Prefix Scan utility

enum ScanKind { exclusive, inclusive };

template<ScanKind Kind, class T>
__device__ __forceinline__ T
scan_warp ( volatile T *ptr, const unsigned int idx = threadIdx.x )
{
    const unsigned int lane = idx & 31;       // index of thread in warp (0..31)

    if (lane >= 1)
    ptr[idx] = ptr[idx - 1] + ptr[idx];
    if (lane >= 2)
    ptr[idx] = ptr[idx - 2] + ptr[idx];
    if (lane >= 4)
    ptr[idx] = ptr[idx - 4] + ptr[idx];
    if (lane >= 8)
    ptr[idx] = ptr[idx - 8] + ptr[idx];
    if (lane >= 16)
    ptr[idx] = ptr[idx - 16] + ptr[idx];

    if (Kind == inclusive)
    return ptr[idx];
    else
    return (lane > 0) ? ptr[idx - 1] : 0;
}
}
}

#endif /* PCL_GPU_KINFU_DEVICE_HPP_ */
