#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

//#include "TestCudaOrb.h"

__device__ short2 operator + (const short2 s2O1_, const short2 s2O2_);
__device__ short2 operator - (const short2 s2O1_, const short2 s2O2_);
__device__ float2 operator * (const float fO1_, const short2 s2O2_);
__device__ short2 operator * (const short sO1_, const short2 s2O2_);
__device__ __host__ float2 operator + (const float2 f2O1_, const float2 f2O2_);
__device__ __host__ float2 operator - (const float2 f2O1_, const float2 f2O2_);
__device__  short2 convert2s2(const float2 f2O1_);

namespace test{


	void loadGlobalConstants( int nImgRows_, int nImgCols_, int nFREAK_OCTAVE_, float fSizeCst_, int nFREAK_SMALLEST_KP_SIZE_, int nFREAK_NB_POINTS_, int nFREAK_NB_SCALES_,
		int nFREAK_NB_ORIENPAIRS_,  int nFREAK_NB_ORIENTATION_,int nFREAK_NB_PAIRS_, double dFREAK_LOG2_);
	void loadOrientationAndDescriptorPair( int4 an4OrientationPair[ 45 ], uchar2 auc2DescriptorPair[ 512 ] );
	void cudaComputeFreakDescriptor(const cv::gpu::GpuMat& cvgmImg_, 
		const cv::gpu::GpuMat& cvgmImgInt_, 
		      cv::gpu::GpuMat& cvgmKeyPoint_, 
		const cv::gpu::GpuMat& cvgmKpScaleIdx_,
		const cv::gpu::GpuMat& cvgmPatternLookup_, //1x FREAK_NB_SCALES*FREAK_NB_ORIENTATION*FREAK_NB_POINTS, x,y,sigma
		const cv::gpu::GpuMat& cvgmPatternSize_, //1x64 
		cv::gpu::GpuMat* pcvgmFreakDescriptor_/*,
		cv::gpu::GpuMat* pcvgmFreakPointPerKp_,
		cv::gpu::GpuMat* pcvgmFreakPointPerKp2_,
		cv::gpu::GpuMat* pcvgmSigma_,
		cv::gpu::GpuMat* pcvgmInt_,
		cv::gpu::GpuMat* pcvgmTheta_*/);
}