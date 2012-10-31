#ifndef BTL_CUDA_VOLUME_HEADER
#define BTL_CUDA_VOLUME_HEADER

namespace btl { namespace device
{

void integrateFrame2VolumeCVCV(cv::gpu::GpuMat& cvgmDepthScaled_, const unsigned short usPyrLevel_, 
	const float fVoxelSize_, const float fTruncDistanceM_, 
	const double* pR_, const double* pT_,  const double* pC_, 
	const float fFx_, const float fFy_, const float u_, const float v_, cv::gpu::GpuMat* pcvgmYZxXVolume_);
//get the threshold voxel centers
void thresholdVolumeCVGL(const cv::gpu::GpuMat& cvgmYZxZVolume_, const float fThreshold_, const float fVoxelSize_, const cv::gpu::GpuMat* pcvgmYZxZVolCenter_);
void exportVolume2CrossSectionX(const cv::gpu::GpuMat& cvgmYZxXVolContentCV_, ushort usV_, ushort usType_, cv::gpu::GpuMat* pcvgmCross_);
}//device
}//btl

#endif