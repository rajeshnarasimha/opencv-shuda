#ifndef BTL_CUDA_REGISTRATION_HEADER
#define BTL_CUDA_REGISTRATION_HEADER


namespace btl { namespace device {
	void registrationICP( const pcl::device::Intr& sCamIntr_, float fDistThres_, float fSinAngleThres_,
							const pcl::device::Mat33& RwCurTrans_, const float3& TwCur_, 
							const pcl::device::Mat33& RwRef_,      const float3& TwRef_, 
							const cv::gpu::GpuMat& cvgmVMapWorldPrev_, const cv::gpu::GpuMat& cvgmNMapWorldPrev_, 
							const cv::gpu::GpuMat& cvgmVMapLocalCur_,  const cv::gpu::GpuMat& cvgmNMapLocalCur_,
							cv::gpu::GpuMat* pcvgmSumBuf_);

	void registrationImageICP(  unsigned int uMatchedFeatures_, 
								const cv::gpu::GpuMat& cvgmMinMatchDistance_, const cv::gpu::GpuMat& cvgmMatchedLocationPrev_,
								const pcl::device::Intr& sCamIntr_, float fDistThres_, float fSinAngleThres_,
								const pcl::device::Mat33& RwCurTrans_, const float3& TwCur_, 
								const pcl::device::Mat33& RwRef_,      const float3& TwRef_, 
								const cv::gpu::GpuMat& cvgmVMapWorldPrev_, const cv::gpu::GpuMat& cvgmNMapWorldPrev_, 
								const cv::gpu::GpuMat& cvgmVMapLocalCur_,  const cv::gpu::GpuMat& cvgmNMapLocalCur_,
								cv::gpu::GpuMat* pcvgmSumBuf_);

}//device
}//btl

#endif