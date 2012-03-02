#ifndef BTL_CUDA_HEADER
#define BTL_CUDA_HEADER
#include "../OtherUtil.hpp"

namespace btl { namespace cuda_util
{

void cudaTestFloat3( const cv::gpu::GpuMat& cvgmIn_, cv::gpu::GpuMat* pcvgmOut_ );
void cudaDepth2Disparity( const cv::gpu::GpuMat& cvgmDepth_, cv::gpu::GpuMat* pcvgmDisparity_ );
void cudaDisparity2Depth( const cv::gpu::GpuMat& cvgmDisparity_, cv::gpu::GpuMat* pcvgmDepth_ );
void cudaUnprojectIRCVCV(const cv::gpu::GpuMat& cvgmDepth_ , 
	const float& dFxIR_, const float& dFyIR_, const float& uIR_, const float& vIR_, 
	cv::gpu::GpuMat* pcvgmIRWorld_ );
//template void cudaTransformIR2RGB<float>(const cv::gpu::GpuMat& cvgmIRWorld_, const T* aR_, const T* aRT_, cv::gpu::GpuMat* pcvgmRGBWorld_);
void cudaTransformIR2RGBCVCV(const cv::gpu::GpuMat& cvgmIRWorld_, const float* aR_, const float* aRT_, cv::gpu::GpuMat* pcvgmRGBWorld_);
void cudaProjectRGBCVCV(const cv::gpu::GpuMat& cvgmRGBWorld_, 
	const float& dFxRGB_, const float& dFyRGB_, const float& uRGB_, const float& vRGB_, 
	cv::gpu::GpuMat* pcvgmAligned_ );
void cudaBilateralFiltering(const cv::gpu::GpuMat& cvgmSrc_, const float& fSigmaSpace_, const float& fSigmaColor_, cv::gpu::GpuMat* pcvgmDst_ );
void cudaPyrDown (const cv::gpu::GpuMat& cvgmSrc_, const float& fSigmaColor_, cv::gpu::GpuMat* pcvgmDst_);
void cudaUnprojectRGBCVBOTH ( const cv::gpu::GpuMat& cvgmDepths_, 
	const float& fFxRGB_,const float& fFyRGB_,const float& uRGB_, const float& vRGB_, unsigned int uLevel_, 
	cv::gpu::GpuMat* pcvgmPts_, 
	btl::utility::tp_coordinate_convention eConvention_ = btl::utility::BTL_GL );
void cudaFastNormalEstimation(const cv::gpu::GpuMat& cvgmPts_, cv::gpu::GpuMat* pcvgmNls_ );
void cudaNormalHistogramCV(const cv::gpu::GpuMat& cvgmNls_, const unsigned short usSamplesAzimuth_, const unsigned short usSamplesElevationZ_, 
	const unsigned short usWidth_,const unsigned short usLevel_,  const float fNormalBinSize_, cv::gpu::GpuMat* pcvgmBinIdx_);
//set the rotation angle and axis for rendering disk GL convention; the input are normals in cv-convention
void cudaNormalSetRotationAxisCVGL(const cv::gpu::GpuMat& cvgmNlCVs_, cv::gpu::GpuMat* pcvgmAAs_ );
//get the threshold voxel centers
void thresholdVolumeCVGL(const cv::gpu::GpuMat& cvgmYZxZVolume_, const float fThreshold_, const float fVoxelSize_, const cv::gpu::GpuMat* pcvgmYZxZVolCenter_);
//get scale depth
void scaleDepthCVmCVm(unsigned short usPyrLevel_, const float fFx_, const float fFy_, const float u_, const float v_, cv::gpu::GpuMat* pcvgmDepth_);
void integrateFrame2VolumeCVCV(const cv::gpu::GpuMat& cvgmDepthScaled_, const float fVoxelSize_, const unsigned short usPyrLevel_, const double* pR_, const double* pT_, 
	const float fFx_, const float fFy_, const float u_, const float v_, cv::gpu::GpuMat* pcvgmYZxXVolume_);
}//cuda_util
}//btl
#endif