
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/devmem2d.hpp>

#include "pcl/device.hpp"

using namespace pcl::device;

namespace pcl
{
namespace device
{

struct SVolumnInit{

	cv::gpu::DevMem2D_<short2> _cvgmVolume;
	ushort VOLUME_X;

	__device__ __forceinline__ void operator () (){
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
      
		if (x < VOLUME_X && y < VOLUME_X)
		{
			short2 *pos = _cvgmVolume.ptr(y) + x;
			int z_step = VOLUME_X * _cvgmVolume.step / sizeof(*pos);

	#pragma unroll
			for(int z = 0; z < VOLUME_X; ++z, pos+=z_step)
				pack_tsdf (0.f, 0, *pos);
		}//if(x < VOLUME_X && y < VOLUME_X)
	}//operator   
};// struct SVolumn

__global__ void kernelInitVolume( SVolumnInit sVI_ ){
	sVI_();
}

void initVolume (cv::gpu::GpuMat* pcvgmVolume_)
{
  struct SVolumnInit sVI;
  sVI._cvgmVolume = *pcvgmVolume_;
  sVI.VOLUME_X = pcvgmVolume_->cols;

  dim3 block (32, 16);
  dim3 grid (1, 1, 1);
  grid.x = cv::gpu::divUp (sVI.VOLUME_X, block.x);      
  grid.y = cv::gpu::divUp (sVI.VOLUME_X, block.y);
  kernelInitVolume<<<grid, block>>>(sVI);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}//initVolume()

struct Tsdf
{
	enum{
		MAX_WEIGHT = 1 << 7
	};
	ushort VOLUME_X;
};




__global__ void
tsdf23 (const cv::gpu::DevMem2D_<float> depthScaled, cv::gpu::DevMem2D_<short2> volume,
        const float tranc_dist, const Mat33 Rcurr_inv, const float3 tcurr, const Intr intr, const float3 cell_size, const ushort VOLUME_X)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= VOLUME_X || y >= VOLUME_X)
    return;

    float v_g_x = (x + 0.5f) * cell_size.x - tcurr.x;
    float v_g_y = (y + 0.5f) * cell_size.y - tcurr.y;
    float v_g_z = (0 + 0.5f) * cell_size.z - tcurr.z;

    float v_g_part_norm = v_g_x * v_g_x + v_g_y * v_g_y;

    float v_x = (Rcurr_inv.data[0].x * v_g_x + Rcurr_inv.data[0].y * v_g_y + Rcurr_inv.data[0].z * v_g_z) * intr.fx;
    float v_y = (Rcurr_inv.data[1].x * v_g_x + Rcurr_inv.data[1].y * v_g_y + Rcurr_inv.data[1].z * v_g_z) * intr.fy;
    float v_z = (Rcurr_inv.data[2].x * v_g_x + Rcurr_inv.data[2].y * v_g_y + Rcurr_inv.data[2].z * v_g_z);

    float z_scaled = 0;

    float Rcurr_inv_0_z_scaled = Rcurr_inv.data[0].z * cell_size.z * intr.fx;
    float Rcurr_inv_1_z_scaled = Rcurr_inv.data[1].z * cell_size.z * intr.fy;

    float tranc_dist_inv = 1.0f / tranc_dist;

    short2* pos = volume.ptr (y) + x;
    int elem_step = volume.step * VOLUME_X / sizeof(short2);

//#pragma unroll
    for (int z = 0; z < VOLUME_X;
        ++z,
        v_g_z += cell_size.z,
        z_scaled += cell_size.z,
        v_x += Rcurr_inv_0_z_scaled,
        v_y += Rcurr_inv_1_z_scaled,
        pos += elem_step)
    {
    float inv_z = 1.0f / (v_z + Rcurr_inv.data[2].z * z_scaled);
    if (inv_z < 0)
        continue;

    // project to current cam
    int2 coo =
    {
        __float2int_rn (v_x * inv_z + intr.cx),
        __float2int_rn (v_y * inv_z + intr.cy)
    };

    if (coo.x >= 0 && coo.y >= 0 && coo.x < depthScaled.cols && coo.y < depthScaled.rows)         //6
    {
        float Dp_scaled = depthScaled.ptr (coo.y)[coo.x]; //meters

        float sdf = Dp_scaled - sqrtf (v_g_z * v_g_z + v_g_part_norm);

        if (Dp_scaled != 0 && sdf >= -tranc_dist) //meters
        {
        float tsdf = fmin (1.0f, sdf * tranc_dist_inv);

        //read and unpack
        float tsdf_prev;
        int weight_prev;
        unpack_tsdf (*pos, tsdf_prev, weight_prev);

        const int Wrk = 1;

        float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
        int weight_new = min (weight_prev + Wrk, Tsdf::MAX_WEIGHT);

        pack_tsdf (tsdf_new, weight_new, *pos);
        }
    }
    }       // for(int z = 0; z < VOLUME_Z; ++z)
}      // __global__




void integrateTsdfVolume(cv::gpu::GpuMat& cvgmDepthScaled_, const unsigned short usPyrLevel_, 
		const float fVoxelSize_, const float fTruncDistanceM_, 
		const pcl::device::Mat33& Rw_, const float3& Cw_, 
		const float fFx_, const float fFy_, const float u_, const float v_, 
		cv::gpu::GpuMat* pcvgmVolume_)
{
	Tsdf tsdf;
	tsdf.VOLUME_X = pcvgmVolume_->cols;
	float3 cell_size;
	cell_size.x = cell_size.y = cell_size.z = fVoxelSize_;

	dim3 block (16, 16);
	dim3 grid (cv::gpu::divUp (tsdf.VOLUME_X, block.x), cv::gpu::divUp (tsdf.VOLUME_X, block.y));

	tsdf23<<<grid, block>>>(cvgmDepthScaled_, *pcvgmVolume_, fTruncDistanceM_, Rw_, Cw_, pcl::device::Intr(fFx_,fFy_,u_,v_)(usPyrLevel_), cell_size,tsdf.VOLUME_X);    

	cudaSafeCall ( cudaGetLastError () );
	cudaSafeCall (cudaDeviceSynchronize ());
}
/*

void integrateTsdfVolume(cv::gpu::GpuMat& cvgmDepthScaled_, const unsigned short usPyrLevel_, 
		const float fVoxelSize_, const float fTruncDistanceM_, 
		const pcl::device::Mat33& Rw_, const float3& Cw_, 
		const float fFx_, const float fFy_, const float u_, const float v_, 
		cv::gpu::GpuMat* pcvgmVolume_)
{
	Tsdf tsdf;
	float3 cell_size;
	cell_size.x = cell_size.y = cell_size.z = fVoxelSize_;
	tsdf.cell_size = cell_size;
	tsdf.depthScaled = cvgmDepthScaled_;
	tsdf.volume = *pcvgmVolume_;
	tsdf.intr = pcl::device::Intr(fFx_,fFy_,u_,v_)(usPyrLevel_);
	tsdf.Rcurr_inv = Rw_;
	tsdf.tcurr = Cw_;
	tsdf.tranc_dist = fTruncDistanceM_;// both the depthScaled and TruncDistance are measured in Meter rather than in mm;
	tsdf.VOLUME_X = pcvgmVolume_->cols; // 512 or 256 or 128;

	dim3 block (16, 16);
	dim3 grid (cv::gpu::divUp (tsdf.VOLUME_X, block.x), cv::gpu::divUp (tsdf.VOLUME_X, block.y));

	integrateTsdfKernel<<<grid, block>>>( tsdf );    

	cudaSafeCall ( cudaGetLastError () );
	cudaSafeCall (cudaDeviceSynchronize ());
}
*/



}//device
}//pcl



