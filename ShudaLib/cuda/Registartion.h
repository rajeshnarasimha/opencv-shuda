#ifndef BTL_CUDA_REGISTRATION_HEADER
#define BTL_CUDA_REGISTRATION_HEADER


namespace btl { namespace device {
	void registrationICP(
		const pcl::device::Intr& sCamIntr_, float fDistThres_, float fSinAngleThres_,
		const pcl::device::Mat33& RwCurTrans_, const float3& TwCur_, 
		const pcl::device::Mat33& RwRef_,      const float3& TwRef_, 
		cv::gpu::GpuMat& cvgmVMapWorldRef_, cv::gpu::GpuMat& cvgmNMapWorldRef_, 
		cv::gpu::GpuMat* pVMapLocalCur_,  cv::gpu::GpuMat* pNMapLocalCur_,
		cv::gpu::GpuMat* pcvgmSumBuf_);

}//device
}//btl

#endif