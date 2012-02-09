#ifndef BTL_CUDA_HEADER
#define BTL_CUDA_HEADER

namespace btl
{
namespace cuda_util
{

void cudaDepth2Disparity( const cv::gpu::GpuMat& cvgmDepth_, cv::gpu::GpuMat* pcvgmDisparity_ );
void cudaDisparity2Depth( const cv::gpu::GpuMat& cvgmDisparity_, cv::gpu::GpuMat* pcvgmDepth_ );

void cudaUnProjectIR(const cv::gpu::GpuMat& cvgmUndistortDepth_ , 
	const double& dFxIR_, const double& dFyIR_, const double& uIR_, const double& vIR_, 
	cv::gpu::GpuMat* pcvgmIRWorld_ );

}//cuda_util
}//btl
#endif