
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/devmem2d.hpp>
#include "cv/common.hpp"
#include "pcl/limits.hpp"
#include "pcl/device.hpp"
#include "pcl/vector_math.hpp"

namespace btl{  namespace device
{
using namespace pcl::device;
__device__ __forceinline__ float getMinTime (const float3& volume_max, const float3& origin, const float3& dir) {
    float txmin = ( (dir.x > 0 ? 0.f : volume_max.x) - origin.x) / dir.x;
    float tymin = ( (dir.y > 0 ? 0.f : volume_max.y) - origin.y) / dir.y; 
    float tzmin = ( (dir.z > 0 ? 0.f : volume_max.z) - origin.z) / dir.z;
    return fmax ( fmax (txmin, tymin), tzmin);
	//return (- origin.z)/dir.z;
}

__device__ __forceinline__ float getMaxTime (const float3& volume_max, const float3& origin, const float3& dir) {
    float txmax = ( (dir.x > 0 ? volume_max.x : 0.f) - origin.x) / dir.x;
    float tymax = ( (dir.y > 0 ? volume_max.y : 0.f) - origin.y) / dir.y;
    float tzmax = ( (dir.z > 0 ? volume_max.z : 0.f) - origin.z) / dir.z;

    return fmin ( fmin (txmax, tymax), tzmax);
}

struct RayCaster
{
    enum { CTA_SIZE_X = 32, CTA_SIZE_Y = 16 };
	enum { VOLUME_X = 512 };//512

    Mat33 Rcurr;
    float3 tcurr;

    float time_step;
	float time_step_fine;
    float3 volume_size;

    float3 cell_size;
    int cols, rows;

    //PtrStep<short2> volume;
	cv::gpu::DevMem2D_<short2> _cvgmYZxXVolume;

    Intr intr;

    mutable cv::gpu::DevMem2D_<float3> _cvgmNMapWorld;
    mutable cv::gpu::DevMem2D_<float3> _cvgmVMapWorld;
	mutable cv::gpu::DevMem2D_<float> _cvgmDepth;
	//mutable cv::gpu::DevMem2D_<uchar> _cvgmDebug;
	
	//get the pixel 3D coordinate in the local
    __device__ __forceinline__ float3 get_ray_next (int x, int y) const {
		float3 ray_next;
		ray_next.x = (x - intr.cx) / intr.fx;
		ray_next.y = (y - intr.cy) / intr.fy;
		ray_next.z = 1;
		return ray_next;
    }

    __device__ __forceinline__ bool isInside (const int3& g) const {
		return (g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < VOLUME_X && g.y < VOLUME_X && g.z <VOLUME_X);
    }

    __device__ __forceinline__ float readTsdf (int x, int y, int z) const {
		return unpack_tsdf (_cvgmYZxXVolume.ptr (x)[ VOLUME_X * y + z ]);
    }
	__device__ __forceinline__ short readTsdf (int3 g ) const {
		return _cvgmYZxXVolume.ptr(g.x)[ VOLUME_X * g.y + g.z ].x;
    }

    __device__ __forceinline__ int3 getVoxel (float3 point) const  {
		int vx = __float2int_ru (point.x / cell_size.x );        // round to negative infinity
		int vy = __float2int_ru (point.y / cell_size.y );
		int vz = __float2int_ru (point.z / cell_size.z );
		return make_int3 (vx, vy, vz);
    }

    __device__ __forceinline__ float interpolateTrilineary (const float3& origin, const float3& dir, float time) const  {
		return interpolateTrilineary (origin + dir * time);
    }

    __device__ __forceinline__ float interpolateTrilineary (const float3& point) const  {
		int3 g = getVoxel (point);

		if (g.x <= 0 || g.x >= VOLUME_X - 1)		return numeric_limits<float>::quiet_NaN ();
		if (g.y <= 0 || g.y >= VOLUME_X - 1)		return numeric_limits<float>::quiet_NaN ();
		if (g.z <= 0 || g.z >= VOLUME_X - 1)		return numeric_limits<float>::quiet_NaN ();

		float vx = (g.x /*+ 0.5f*/) * cell_size.x;
		float vy = (g.y /*+ 0.5f*/) * cell_size.y;
		float vz = (g.z /*+ 0.5f*/) * cell_size.z;

		g.x = (point.x < vx) ? (g.x - 1) : g.x;
		g.y = (point.y < vy) ? (g.y - 1) : g.y;
		g.z = (point.z < vz) ? (g.z - 1) : g.z;

		float a = (point.x - (g.x + 0.5f) * cell_size.x) / cell_size.x;
		float b = (point.y - (g.y + 0.5f) * cell_size.y) / cell_size.y;
		float c = (point.z - (g.z + 0.5f) * cell_size.z) / cell_size.z;

		float tsdf0 = readTsdf (g.x + 0, g.y + 0, g.z + 0); if ( abs(tsdf0)>=1.f ) return numeric_limits<float>::quiet_NaN (); //tsdf0 = 0;//
		float tsdf1 = readTsdf (g.x + 0, g.y + 0, g.z + 1); if ( abs(tsdf1)>=1.f ) return numeric_limits<float>::quiet_NaN (); //tsdf1 = 0;//
		float tsdf2 = readTsdf (g.x + 0, g.y + 1, g.z + 0); if ( abs(tsdf2)>=1.f ) return numeric_limits<float>::quiet_NaN (); //tsdf2 = 0;//
		float tsdf3 = readTsdf (g.x + 0, g.y + 1, g.z + 1); if ( abs(tsdf3)>=1.f ) return numeric_limits<float>::quiet_NaN (); //tsdf3 = 0;//
		float tsdf4 = readTsdf (g.x + 1, g.y + 0, g.z + 0); if ( abs(tsdf4)>=1.f ) return numeric_limits<float>::quiet_NaN (); //tsdf4 = 0;//
		float tsdf5 = readTsdf (g.x + 1, g.y + 0, g.z + 1); if ( abs(tsdf5)>=1.f ) return numeric_limits<float>::quiet_NaN (); //tsdf5 = 0;//
		float tsdf6 = readTsdf (g.x + 1, g.y + 1, g.z + 0); if ( abs(tsdf6)>=1.f ) return numeric_limits<float>::quiet_NaN (); //tsdf6 = 0;//
		float tsdf7 = readTsdf (g.x + 1, g.y + 1, g.z + 1); if ( abs(tsdf7)>=1.f ) return numeric_limits<float>::quiet_NaN (); //tsdf7 = 0;//
		float res = tsdf0 * (1 - a) * (1 - b) * (1 - c) +
					tsdf1 * (1 - a) * (1 - b) * c +
					tsdf2 * (1 - a) * b * (1 - c) +
					tsdf3 * (1 - a) * b * c +
					tsdf4 * a * (1 - b) * (1 - c) +
					tsdf5 * a * (1 - b) * c +
					tsdf6 * a * b * (1 - c) +
					tsdf7 * a * b * c;
		return res;
    }//interpolateTrilineary()

	__device__ __forceinline__ void operator () () const {
		int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
        int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

        if (x >= cols || y >= rows)        return;

        float3& f3V = _cvgmVMapWorld.ptr (y)[x];	f3V.x = f3V.y = f3V.z = numeric_limits<float>::quiet_NaN ();
        float3& f3N = _cvgmNMapWorld.ptr (y)[x];	f3N.x = f3N.y = f3N.z = numeric_limits<float>::quiet_NaN ();
		//uchar& ucDebug = _cvgmDebug.ptr (y)[x]; ucDebug = 0;
		_cvgmDepth.ptr (y)[x] = numeric_limits<float>::quiet_NaN ();

		float3 ray_start = tcurr; //is the camera center in world
		float3 ray_camera = get_ray_next (x, y);
		float3 ray_next = Rcurr * ray_camera + tcurr; //transform the point from local to the world
		float3 ray_dir = normalized (ray_next - ray_start); //get ray direction in the world

        //ensure that it isn't a degenerate case
        ray_dir.x = (ray_dir.x == 0.f) ? 1e-15 : ray_dir.x;
        ray_dir.y = (ray_dir.y == 0.f) ? 1e-15 : ray_dir.y;
        ray_dir.z = (ray_dir.z == 0.f) ? 1e-15 : ray_dir.z;

        // computer time when entry and exit volume
        float time_start_volume = getMinTime (volume_size, ray_start, ray_dir);
        float time_exit_volume  = getMaxTime (volume_size, ray_start, ray_dir);

        const float min_dist = 0.f;         //in meters
        time_start_volume = fmax (time_start_volume, min_dist);
		if (time_start_volume >= time_exit_volume) {return;}

        float time_curr = time_start_volume;
        int3 g = getVoxel (ray_start + ray_dir * time_curr);
        g.x = max (0, min (g.x, VOLUME_X - 1));//ensure g.x is between 1 and VOLUME_X-1.
        g.y = max (0, min (g.y, VOLUME_X - 1));
        g.z = max (0, min (g.z, VOLUME_X - 1));

        float tsdf = readTsdf (g.x, g.y, g.z);

        //infinite loop guard
        const float max_time = time_exit_volume; //3 * (volume_size.x + volume_size.y + volume_size.z);

        for (; time_curr < max_time; time_curr += time_step)
        {
			int3 g = getVoxel (  ray_start + ray_dir * (time_curr + time_step)  );        if (!isInside (g))   { return;}
          float tsdf_prev = tsdf;
		  tsdf = readTsdf (g.x, g.y, g.z);    if (tsdf_prev < 0.f && tsdf > 0.f) {continue;}

          if (tsdf_prev > 0.f && tsdf < 0.f)           //zero crossing
          {
			  //ucDebug = 255; 
			  /*			  //do a finer search{
			  float tsdf_finer = tsdf_prev;
			  float time_finer_prev = 0;
			  for(float time_finer = time_step_fine; time_finer < time_step; time_finer += time_step_fine){
				float tsdf_finer_prev = tsdf_finer;
				int3 g_curr = getVoxel (  ray_start + ray_dir * (time_curr + time_finer)  );      
				tsdf_finer = readTsdf (g_curr.x, g_curr.y, g_curr.z);    if (tsdf_finer_prev < 0.f && tsdf_finer > 0.f) break;
				if (tsdf_finer_prev > 0.f && tsdf_finer < 0.f) {
					float Ftdt = interpolateTrilineary (ray_start, ray_dir, time_curr + time_finer); if (isnan (Ftdt))  return;
					float Ft   = interpolateTrilineary (ray_start, ray_dir, time_curr + time_finer_prev);      if (isnan (Ft))  return;
					float lambda = Ft / (Ft - Ftdt); if( lambda > 1.f || lambda < 0.f ) lambda = 0.5; //lambda = max( 0.f, min( lambda, 1.f) ); //control lambda into (0,1)
					float Ts_finer = time_curr + time_finer_prev + time_finer * lambda ; //tsdf_prev/(tsdf_prev - tsdf);//
					float3 vetex_found = ray_start + ray_dir * Ts_finer;
					float fCos = 1.f/pcl::device::norm( ray_camera );
					_cvgmDepth.ptr (y)[x] = Ts_finer*fCos;
					return;
				}//zero cross detectde, do interpolation
			    time_finer_prev = time_finer;
			  }//for each finer step*/
			
            float Ftdt = interpolateTrilineary (ray_start, ray_dir, time_curr + time_step); if (isnan (Ftdt))  continue;
            float Ft = interpolateTrilineary (ray_start, ray_dir, time_curr);               if (isnan (Ft))  continue;
			float lambda = Ft / (Ft - Ftdt); if( lambda > 1.f || lambda < 0.f ) lambda = 0.5; //lambda = max( 0.f, min( lambda, 1.f) ); //control lambda into (0,1)
            float Ts = time_curr + time_step * lambda ; //tsdf_prev/(tsdf_prev - tsdf);//
            float3 vetex_found = ray_start + ray_dir * Ts;
            
            int3 g = getVoxel ( ray_start + ray_dir * time_curr );
            if (g.x > 1 && g.y > 1 && g.z > 1 && g.x < VOLUME_X - 2 && g.y < VOLUME_X - 2 && g.z < VOLUME_X - 2)
            {
              float3 t;
              float3 n;

              t = vetex_found;
              t.x += cell_size.x;
              float Fx1 = interpolateTrilineary (t); if (isnan(Fx1)) continue;

              t = vetex_found;
              t.x -= cell_size.x;
              float Fx2 = interpolateTrilineary (t); if (isnan(Fx2)) continue;

              n.x = (Fx1 - Fx2);

              t = vetex_found;
              t.y += cell_size.y;
              float Fy1 = interpolateTrilineary (t); if (isnan(Fy1)) continue;

              t = vetex_found;
              t.y -= cell_size.y;
              float Fy2 = interpolateTrilineary (t); if (isnan(Fy2)) continue;

              n.y = (Fy1 - Fy2);

              t = vetex_found;
              t.z += cell_size.z;
              float Fz1 = interpolateTrilineary (t); if (isnan(Fz1)) continue;

              t = vetex_found;
              t.z -= cell_size.z;
              float Fz2 = interpolateTrilineary (t); if (isnan(Fz2)) continue;

              n.z = (Fz1 - Fz2);

              n = normalized (n);
			  //if( pcl::device::dot( n, ray_dir ) < -0.5f ){ //cos (pi/3)
				  //float fCos = 1.f/pcl::device::norm( ray_camera );
			      _cvgmNMapWorld.ptr (y)[x] = n;
				  _cvgmVMapWorld.ptr (y)[x] = vetex_found;
				  //_cvgmDepth.ptr (y)[x] = Ts*fCos;
			  //}
            }//if grid inside volume
            return;
          }//if zero cross has found
        }// for each time step
		return;
      }//operator()

};//SRayCaster

__global__ void
rayCastKernel (const RayCaster sRC_) {
    sRC_ ();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
void raycast (const pcl::device::Intr& sCamIntr_, const pcl::device::Mat33& RwCurrTrans_, const float3& CwCurr_, 
		float fTrancDist_, const float& fVolumeSize_,
		const cv::gpu::GpuMat& cvgmYZxXVolume_,  cv::gpu::GpuMat* pcvgmDepth_/ *cv::gpu::GpuMat* pcvgmVMapWorld_, cv::gpu::GpuMat* pcvgmNMapWorld_* /)
{
  btl::device::RayCaster sRC;

  sRC.Rcurr = RwCurrTrans_; //Rw'
  sRC.tcurr = CwCurr_; //-Rw'*Tw

  sRC.volume_size.x = fVolumeSize_;
  sRC.volume_size.y = fVolumeSize_;
  sRC.volume_size.z = fVolumeSize_;

  sRC.cell_size.x = fVolumeSize_ / cvgmYZxXVolume_.rows;
  sRC.cell_size.y = fVolumeSize_ / cvgmYZxXVolume_.rows;
  sRC.cell_size.z = fVolumeSize_ / cvgmYZxXVolume_.rows;
  
  sRC.time_step = fTrancDist_*0.125f;
  sRC.time_step_fine = sRC.cell_size.x * 1.f;


  sRC.cols = pcvgmDepth_->cols;
  sRC.rows = pcvgmDepth_->rows;

  sRC.intr = sCamIntr_;

  sRC._cvgmYZxXVolume = cvgmYZxXVolume_;

  pcvgmDepth_->setTo(std::numeric_limits<float>::quiet_NaN ());
  sRC._cvgmDepth = *pcvgmDepth_;

  dim3 block (RayCaster::CTA_SIZE_X, RayCaster::CTA_SIZE_Y);
  dim3 grid (cv::gpu::divUp (sRC.cols, block.x), cv::gpu::divUp (sRC.rows, block.y));

  rayCastKernel<<<grid, block>>>(sRC);
  cudaSafeCall (cudaGetLastError ());
  //cudaSafeCall(cudaDeviceSynchronize());
}//raycast()*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void raycast (const pcl::device::Intr& sCamIntr_, const pcl::device::Mat33& RwCurrTrans_, const float3& CwCurr_, 
		float fTrancDist_, const float& fVolumeSize_,
		const cv::gpu::GpuMat& cvgmYZxXVolume_,  cv::gpu::GpuMat* pcvgmVMapWorld_, cv::gpu::GpuMat* pcvgmNMapWorld_,cv::gpu::GpuMat* pcvgmDepth_, cv::Mat* pcvmDebug_ = NULL )
{
  btl::device::RayCaster sRC;

  sRC.Rcurr = RwCurrTrans_; //Rw'
  sRC.tcurr = CwCurr_; //-Rw'*Tw

  sRC.volume_size.x = fVolumeSize_;
  sRC.volume_size.y = fVolumeSize_;
  sRC.volume_size.z = fVolumeSize_;

  sRC.cell_size.x = fVolumeSize_ / cvgmYZxXVolume_.rows;
  sRC.cell_size.y = fVolumeSize_ / cvgmYZxXVolume_.rows;
  sRC.cell_size.z = fVolumeSize_ / cvgmYZxXVolume_.rows;
  
  sRC.time_step = fTrancDist_*0.8f;
  //sRC.time_step_fine = fTrancDist_*0.02f;//sRC.cell_size.x * 1.f;

  sRC.cols = pcvgmVMapWorld_->cols;
  sRC.rows = pcvgmVMapWorld_->rows;

  sRC.intr = sCamIntr_;

  sRC._cvgmYZxXVolume = cvgmYZxXVolume_;
  
  sRC._cvgmVMapWorld = *pcvgmVMapWorld_;
  sRC._cvgmNMapWorld = *pcvgmNMapWorld_;
  sRC._cvgmDepth     = *pcvgmDepth_;

  //cv::gpu::GpuMat cvgmDebug(sRC.rows,sRC.cols,CV_8UC1);
  //sRC._cvgmDebug = cvgmDebug;

  dim3 block (RayCaster::CTA_SIZE_X, RayCaster::CTA_SIZE_Y);
  dim3 grid (cv::gpu::divUp (sRC.cols, block.x), cv::gpu::divUp (sRC.rows, block.y));

  cv::Mat cvmDebug;

  rayCastKernel<<<grid, block>>>(sRC);
  cudaSafeCall (cudaGetLastError ());
  //if(pcvmDebug_)
  //	  cvgmDebug.download(*pcvmDebug_);
  //cudaSafeCall(cudaDeviceSynchronize());
}//raycast()
}//device
}//btl