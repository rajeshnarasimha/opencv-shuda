#ifndef BTL_CUDA_RAYCASTER_HEADER
#define BTL_CUDA_RAYCASTER_HEADER


namespace btl { namespace device {
	void raycast (const pcl::device::Intr& sCamIntr_, const pcl::device::Mat33& RwCurrTrans_, const float3& CwCurr_, 
		float fTrancDist_, const float& fVolumeSize_,
		const cv::gpu::GpuMat& cvgmYZxXVolume_, cv::gpu::GpuMat* pcvgmDepth_/*cv::gpu::GpuMat* pcvgmVMapWorld_, cv::gpu::GpuMat* pcvgmNMapWorld_*/);
	/*void raycast (const pcl::device::Intr& sCamIntr_, const pcl::device::Mat33& RwCurrTrans_, const float3& CwCurr_, 
		float fTrancDist_, const float& fVolumeSize_,
		const cv::gpu::GpuMat& cvgmYZxXVolume_,  cv::gpu::GpuMat* pcvgmVMapWorld_, cv::gpu::GpuMat* pcvgmNMapWorld_,cv::gpu::GpuMat* pcvgmDepth_ );*/
	void raycast (const pcl::device::Intr& sCamIntr_, const pcl::device::Mat33& RwCurrTrans_, const float3& CwCurr_, 
		float fTrancDist_, const float& fVolumeSize_,
		const cv::gpu::GpuMat& cvgmYZxXVolume_,  cv::gpu::GpuMat* pcvgmVMapWorld_, cv::gpu::GpuMat* pcvgmNMapWorld_,cv::gpu::GpuMat* pcvgmDepth_, cv::Mat* pcvmDebug_ = NULL );
}//device
}//btl


namespace pcl { namespace device{
	void raycast (const pcl::device::Intr& intr, const pcl::device::Mat33& RwInv_, const float3& Cw_, 
		const float fVolumeSizeM_, const float fTruncDistanceM_, const float& fVoxelSize_,
		const cv::gpu::GpuMat& cvgmVolume_, cv::gpu::GpuMat* pVMap_, cv::gpu::GpuMat* pNMap_);
}
}

#endif