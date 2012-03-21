#ifndef BTL_CUDA_RAYCASTER_HEADER
#define BTL_CUDA_RAYCASTER_HEADER


namespace btl { namespace device {
	void raycast (const pcl::device::Intr& sCamIntr_, const pcl::device::Mat33& RwCurrTrans_, const float3& CwCurr_, 
		float fTrancDist_, const float& fVolumeSize_,
		const cv::gpu::GpuMat& cvgmYZxXVolume_, cv::gpu::GpuMat* pcvgmVMapWorld_, cv::gpu::GpuMat* pcvgmNMapWorld_);

}//device
}//btl

#endif