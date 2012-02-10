#ifndef BTL_CUDA_HEADER
#define BTL_CUDA_HEADER

namespace btl
{
namespace cuda_util
{
void cudaTestFloat3( const cv::gpu::GpuMat& cvgmIn_, cv::gpu::GpuMat* pcvgmOut_ );

void cudaDepth2Disparity( const cv::gpu::GpuMat& cvgmDepth_, cv::gpu::GpuMat* pcvgmDisparity_ );
void cudaDisparity2Depth( const cv::gpu::GpuMat& cvgmDisparity_, cv::gpu::GpuMat* pcvgmDepth_ );

void cudaUnProjectIR(const cv::gpu::GpuMat& cvgmUndistortDepth_ , 
	const double& dFxIR_, const double& dFyIR_, const double& uIR_, const double& vIR_, 
	cv::gpu::GpuMat* pcvgmIRWorld_ );
void cudaTransformIR2RGB(const cv::gpu::GpuMat& cvgmIRWorld_, double* aR_, double* aRT_, cv::gpu::GpuMat* pcvgmRGBWorld_);
void cudaProjectRGB(const cv::gpu::GpuMat& cvgmRGBWorld_, 
	const double& dFxRGB_, const double& dFyRGB_, const double& uRGB_, const double& vRGB_, 
	const cv::gpu::GpuMat* pcvgmAligned_ );
void cudaBilateralFiltering(const cv::gpu::GpuMat& cvgmSrc_, const float& fSigmaSpace_, const float& fSigmaColor_, cv::gpu::GpuMat* pcvgmDst_ );
}//cuda_util
}//btl
#endif