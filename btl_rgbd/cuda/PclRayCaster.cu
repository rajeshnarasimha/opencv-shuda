
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/devmem2d.hpp>
#include "cv/common.hpp"
#include "pcl/limits.hpp"
#include "pcl/device.hpp"
#include "pcl/vector_math.hpp"

namespace pcl{  namespace device  {

__device__ __forceinline__ float
getMinTime (const float3& volume_max, const float3& origin, const float3& dir)
{
    float txmin = ( (dir.x > 0 ? 0.f : volume_max.x) - origin.x) / dir.x;
    float tymin = ( (dir.y > 0 ? 0.f : volume_max.y) - origin.y) / dir.y;
    float tzmin = ( (dir.z > 0 ? 0.f : volume_max.z) - origin.z) / dir.z;

    return fmax ( fmax (txmin, tymin), tzmin);
}

__device__ __forceinline__ float
getMaxTime (const float3& volume_max, const float3& origin, const float3& dir)
{
    float txmax = ( (dir.x > 0 ? volume_max.x : 0.f) - origin.x) / dir.x;
    float tymax = ( (dir.y > 0 ? volume_max.y : 0.f) - origin.y) / dir.y;
    float tzmax = ( (dir.z > 0 ? volume_max.z : 0.f) - origin.z) / dir.z;

    return fmin (fmin (txmax, tymax), tzmax);
}

struct RayCaster
{
	enum { CTA_SIZE_X = 32, CTA_SIZE_Y = 8 };

	ushort VOLUME_X;

	Mat33 Rcurr;  //Rw'
	float3 tcurr; //Cw camera center

	float time_step;
	float3 volume_size;

	float3 cell_size;
	int cols, rows;

	cv::gpu::DevMem2D_<short2> _cvgmVolume;

	Intr intr;

	mutable cv::gpu::DevMem2D_<float3> nmap;
	mutable cv::gpu::DevMem2D_<float3> vmap;

	__device__ __forceinline__ float3
	get_ray_next (int x, int y) const
	{
		float3 ray_next;
		ray_next.x = (x - intr.cx) / intr.fx;
		ray_next.y = (y - intr.cy) / intr.fy;
		ray_next.z = 1;
		return ray_next;
	}

	__device__ __forceinline__ bool
	checkInds (const int3& g) const
	{
		return (g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < VOLUME_X && g.y < VOLUME_X && g.z < VOLUME_X);
	}

    __device__ __forceinline__ float
    readTsdf (int x, int y, int z) const
    {
		return unpack_tsdf (_cvgmVolume.ptr (VOLUME_X * z + y)[x]);
    }

    __device__ __forceinline__ int3
    getVoxel (float3 point) const
    {
		int vx = __float2int_rd (point.x / cell_size.x);        // round to negative infinity
		int vy = __float2int_rd (point.y / cell_size.y);
		int vz = __float2int_rd (point.z / cell_size.z);

		return make_int3 (vx, vy, vz);
    }

    __device__ __forceinline__ float
    interpolateTrilineary (const float3& origin, const float3& dir, float time) const
    {
		return interpolateTrilineary (origin + dir * time);
    }

    __device__ __forceinline__ float
    interpolateTrilineary (const float3& point) const
    {
		int3 g = getVoxel (point);

		if (g.x <= 0 || g.x >= VOLUME_X - 1)
			return numeric_limits<float>::quiet_NaN ();

		if (g.y <= 0 || g.y >= VOLUME_X - 1)
			return numeric_limits<float>::quiet_NaN ();

		if (g.z <= 0 || g.z >= VOLUME_X - 1)
			return numeric_limits<float>::quiet_NaN ();

		float vx = (g.x + 0.5f) * cell_size.x;
		float vy = (g.y + 0.5f) * cell_size.y;
		float vz = (g.z + 0.5f) * cell_size.z;

		g.x = (point.x < vx) ? (g.x - 1) : g.x;
		g.y = (point.y < vy) ? (g.y - 1) : g.y;
		g.z = (point.z < vz) ? (g.z - 1) : g.z;

		float a = (point.x - (g.x + 0.5f) * cell_size.x) / cell_size.x;
		float b = (point.y - (g.y + 0.5f) * cell_size.y) / cell_size.y;
		float c = (point.z - (g.z + 0.5f) * cell_size.z) / cell_size.z;

		float res = readTsdf (g.x + 0, g.y + 0, g.z + 0) * (1 - a) * (1 - b) * (1 - c) +
					readTsdf (g.x + 0, g.y + 0, g.z + 1) * (1 - a) * (1 - b) * c +
					readTsdf (g.x + 0, g.y + 1, g.z + 0) * (1 - a) * b * (1 - c) +
					readTsdf (g.x + 0, g.y + 1, g.z + 1) * (1 - a) * b * c +
					readTsdf (g.x + 1, g.y + 0, g.z + 0) * a * (1 - b) * (1 - c) +
					readTsdf (g.x + 1, g.y + 0, g.z + 1) * a * (1 - b) * c +
					readTsdf (g.x + 1, g.y + 1, g.z + 0) * a * b * (1 - c) +
					readTsdf (g.x + 1, g.y + 1, g.z + 1) * a * b * c;
		return res;
    }

    __device__ __forceinline__ void
    operator () () const
    {
		int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
		int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

		if (x >= cols || y >= rows)
			return;

		float3& V = vmap.ptr (y)[x];
		V.x = V.y = V.z = pcl::device::numeric_limits<float>::quiet_NaN ();
		float3& N = nmap.ptr (y)[x];
		N.x = N.y = N.z = pcl::device::numeric_limits<float>::quiet_NaN ();

		float3 ray_start = tcurr;
		float3 ray_next = Rcurr * get_ray_next (x, y) + tcurr; // transform from camera to world

		float3 ray_dir = normalized (ray_next - ray_start);

		//ensure that it isn't a degenerate case
		ray_dir.x = (ray_dir.x == 0.f) ? 1e-15 : ray_dir.x;
		ray_dir.y = (ray_dir.y == 0.f) ? 1e-15 : ray_dir.y;
		ray_dir.z = (ray_dir.z == 0.f) ? 1e-15 : ray_dir.z;

        // computer time when entry and exit volume
        float time_start_volume = getMinTime (volume_size, ray_start, ray_dir);
        float time_exit_volume = getMaxTime (volume_size, ray_start, ray_dir);

        const float min_dist = 0.f;         //in meters
        time_start_volume = fmax (time_start_volume, min_dist);
        if (time_start_volume >= time_exit_volume)      return;

        float time_curr = time_start_volume;
        int3 g = getVoxel (ray_start + ray_dir * time_curr);
        g.x = max (0, min (g.x, VOLUME_X - 1));
        g.y = max (0, min (g.y, VOLUME_X - 1));
        g.z = max (0, min (g.z, VOLUME_X - 1));

        float tsdf = readTsdf (g.x, g.y, g.z);

        //infinite loop guard
        const float max_time = time_exit_volume; //3 * (volume_size.x + volume_size.y + volume_size.z);

        for (; time_curr < max_time; time_curr += time_step)
        {
			float tsdf_prev = tsdf;

			int3 g = getVoxel (  ray_start + ray_dir * (time_curr + time_step)  );
			if (!checkInds (g))  break;

			tsdf = readTsdf (g.x, g.y, g.z);

			if (tsdf_prev < 0.f && tsdf > 0.f)  break;

			if (tsdf_prev > 0.f && tsdf < 0.f)           //zero crossing
			{
				float Ftdt = interpolateTrilineary (ray_start, ray_dir, time_curr + time_step);
				if (isnan (Ftdt)) break;

				float Ft = interpolateTrilineary (ray_start, ray_dir, time_curr);
				if (isnan (Ft)) break;

				float Ts = time_curr - time_step * Ft / (Ftdt - Ft);

				float3 vetex_found = ray_start + ray_dir * Ts;

				vmap.ptr (y)[x]  = vetex_found;

				int3 g = getVoxel ( ray_start + ray_dir * time_curr );
				if (g.x > 1 && g.y > 1 && g.z > 1 && g.x < VOLUME_X - 2 && g.y < VOLUME_X - 2 && g.z < VOLUME_X - 2)
				{
					float3 t;
					float3 n;

					t = vetex_found;
					t.x += cell_size.x;
					float Fx1 = interpolateTrilineary (t);

					t = vetex_found;
					t.x -= cell_size.x;
					float Fx2 = interpolateTrilineary (t);

					n.x = (Fx1 - Fx2);

					t = vetex_found;
					t.y += cell_size.y;
					float Fy1 = interpolateTrilineary (t);

					t = vetex_found;
					t.y -= cell_size.y;
					float Fy2 = interpolateTrilineary (t);

					n.y = (Fy1 - Fy2);

					t = vetex_found;
					t.z += cell_size.z;
					float Fz1 = interpolateTrilineary (t);

					t = vetex_found;
					t.z -= cell_size.z;
					float Fz2 = interpolateTrilineary (t);

					n.z = (Fz1 - Fz2);

					n = normalized (n);
					nmap.ptr (y)[x] = n;
				}
				break;
          }//if (tsdf_prev > 0.f && tsdf < 0.f) 
      }//for (; time_curr < max_time; time_curr += time_step)
   }//operator ()
};

__global__ void
rayCastKernel (const RayCaster rc) {
    rc ();
}


//get VMap and NMap in world
void raycast (const pcl::device::Intr& intr, const pcl::device::Mat33& RwInv_, const float3& Cw_, 
                      const float fVolumeSizeM_, const float fTruncDistanceM_, const float& fVoxelSize_,
                      const cv::gpu::GpuMat& cvgmVolume_, cv::gpu::GpuMat* pVMap_, cv::gpu::GpuMat* pNMap_)
{
	RayCaster rc;

	rc.Rcurr = RwInv_;
	rc.tcurr = Cw_;

	rc.time_step = fTruncDistanceM_ * 0.8f; 
	rc.volume_size.x = rc.volume_size.y = rc.volume_size.z = fVolumeSizeM_;

	rc.VOLUME_X = cvgmVolume_.cols;
	rc.cell_size.x = rc.cell_size.y = rc.cell_size.z = fVoxelSize_;

	rc.cols = pVMap_->cols;
	rc.rows = pVMap_->rows;

	rc.intr = intr;

	rc._cvgmVolume = cvgmVolume_;
	rc.vmap = *pVMap_;
	rc.nmap = *pNMap_;

	dim3 block (RayCaster::CTA_SIZE_X, RayCaster::CTA_SIZE_Y);
	dim3 grid (cv::gpu::divUp (rc.cols, block.x), cv::gpu::divUp (rc.rows, block.y));

	rayCastKernel<<<grid, block>>>(rc);
	cudaSafeCall (cudaGetLastError ());
	//cudaSafeCall(cudaDeviceSynchronize());
}

}// device
}// pcl


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

