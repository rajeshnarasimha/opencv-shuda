#ifndef BTL_CUDA_HEADER
#define BTL_CUDA_HEADER
//#include "../OtherUtil.hpp"

namespace btl { namespace device
{
void cudaTestFloat3( const cv::gpu::GpuMat& cvgmIn_, cv::gpu::GpuMat* pcvgmOut_ );
void cudaDepth2Disparity( const cv::gpu::GpuMat& cvgmDepth_, cv::gpu::GpuMat* pcvgmDisparity_ );
void cudaDepth2Disparity2( const cv::gpu::GpuMat& cvgmDepth_, float fCutOffDistance_, cv::gpu::GpuMat* pcvgmDisparity_ );
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
void unprojectRGBCVm ( const cv::gpu::GpuMat& cvgmDepths_, 
	const float& fFxRGB_,const float& fFyRGB_,const float& uRGB_, const float& vRGB_, unsigned int uLevel_, 
	cv::gpu::GpuMat* pcvgmPts_ );
void cudaFastNormalEstimation(const cv::gpu::GpuMat& cvgmPts_, cv::gpu::GpuMat* pcvgmNls_ );
void cudaNormalHistogramCV(const cv::gpu::GpuMat& cvgmNls_, const unsigned short usSamplesAzimuth_, const unsigned short usSamplesElevationZ_, 
	const unsigned short usWidth_,const unsigned short usLevel_,  const float fNormalBinSize_, cv::gpu::GpuMat* pcvgmBinIdx_);
//set the rotation angle and axis for rendering disk GL convention; the input are normals in cv-convention
void cudaNormalSetRotationAxisCVGL(const cv::gpu::GpuMat& cvgmNlCVs_, cv::gpu::GpuMat* pcvgmAAs_ );
//get scale depth
void scaleDepthCVmCVm(unsigned short usPyrLevel_, const float fFx_, const float fFy_, const float u_, const float v_, cv::gpu::GpuMat* pcvgmDepth_);
void transformLocalToWorldCVCV(const float* pRw_/*col major*/, const float* pTw_, cv::gpu::GpuMat* pcvgmPts_, cv::gpu::GpuMat* pcvgmNls_);
void checkNVMap(const cv::gpu::GpuMat& cvgmPts_, const cv::gpu::GpuMat& cvgmNls_, cv::gpu::GpuMat* pcvgmResults_);
//resize the normal or vertex map to half of its size
void resizeMap (bool bNormalize_, const cv::gpu::GpuMat& cvgmSrc_, cv::gpu::GpuMat* pcvgmDst_);
void rgb2RGBA(const cv::gpu::GpuMat& cvgmRGB_, const uchar uA_, cv::gpu::GpuMat* pcvgmRGBA_);
void boundaryDetector(const float fThreshold, const cv::gpu::GpuMat& cvgmPt_, const cv::gpu::GpuMat& cvgmNl_, cv::gpu::GpuMat* cvgmRGB_);
void cudaConvertZValue2Depth(const cv::gpu::GpuMat& cvgmZValue_, float fNear_, float fFar_, cv::gpu::GpuMat* pcvgmDepth_);
void cudaConvertGL2CV(const cv::gpu::GpuMat cvgmRGB_, cv::gpu::GpuMat* pcvgmUndistRGB_);
}//device
}//btl
#endif