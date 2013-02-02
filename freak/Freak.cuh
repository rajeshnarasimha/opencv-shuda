
namespace btl { namespace device
{
namespace freak
{
void loadGlobalConstants( int nFREAK_OCTAVE_, float fSizeCst_, int nFREAK_SMALLEST_KP_SIZE_, int nFREAK_NB_POINTS_, int nFREAK_NB_SCALES_,
						  int nFREAK_NB_ORIENPAIRS_,  int nFREAK_NB_ORIENTATION_,int nFREAK_NB_PAIRS_, double dFREAK_LOG2_);
void loadGlobalConstantsImgResolution( int nImgRows_, int nImgCols_);
void cudaBuildFreakPattern( const double& dPatternScale_, const double& dScaleStep_, int nFREAK_NB_ORIENTATION_,  const int n[8],const double radius[8],const double sigma[8],
						    cv::gpu::GpuMat* pcvgmPatternLookup_, cv::gpu::GpuMat* pcvgmPatternSize_);
void cudaComputeScaleIndex(const cv::gpu::GpuMat& _cvgmPatternSize,cv::gpu::GpuMat* _cvgmKeypoint,cv::gpu::GpuMat* _cvgmKpScale);
void loadOrientationAndDescriptorPair( int4 an4OrientationPair[ 45 ], uchar2 auc2DescriptorPair[ 512 ] );
unsigned int cudaComputeFreakDescriptor(const cv::gpu::GpuMat& cvgmImg_, 
								const cv::gpu::GpuMat& cvgmImgInt_, 
									  cv::gpu::GpuMat& cvgmKeyPoint_, // the FREAK angle will be returned here
								const cv::gpu::GpuMat& cvgmKpScaleIdx_,
								const cv::gpu::GpuMat& cvgmPatternLookup_, //1x FREAK_NB_SCALES*FREAK_NB_ORIENTATION*FREAK_NB_POINTS, x,y,sigma
								const cv::gpu::GpuMat& cvgmPatternSize_, //1x64 
								cv::gpu::GpuMat* pcvgmFreakDescriptor_/*,
								cv::gpu::GpuMat* pcvgmFreakPointPerKp_,
								cv::gpu::GpuMat* pcvgmFreakPointPerKp2_,
								cv::gpu::GpuMat* pcvgmSigma_,
								cv::gpu::GpuMat* pcvgmInt_,
								cv::gpu::GpuMat* pcvgmTheta_*/);
}//freak
}//device
}//btl