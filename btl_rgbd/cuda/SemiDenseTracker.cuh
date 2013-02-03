#ifndef SEMIDENSE_CUDA_HEADER
#define SEMIDENSE_CUDA_HEADER


namespace btl{ namespace device{ namespace semidense{
	//for debug
	void cudaCalcMaxContrast(const cv::gpu::GpuMat& cvgmImage_, const unsigned char ucContrastThreshold_, cv::gpu::GpuMat* pcvgmContrast_);
	//for debug
	void cudaCalcMinDiameterContrast(const cv::gpu::GpuMat& cvgmImage_, cv::gpu::GpuMat* pcvgmContrast_);
	unsigned int cudaCalcSaliency(const cv::gpu::GpuMat& cvgmImage_, const unsigned short usHalfSizeRound_,
		const unsigned char ucContrastThreshold_, const float& fSaliencyThreshold_, 
		cv::gpu::GpuMat* pcvgmSaliency_, cv::gpu::GpuMat* pcvgmKeyPointLocations_);
	unsigned int cudaNonMaxSupression(const cv::gpu::GpuMat& cvgmKeyPointLocation_, const unsigned int uMaxSalientPoints_, 
		const cv::gpu::GpuMat& cvgmSaliency_, short2* ps2devLocations_, float* pfdevResponse_);
	//sort
	void thrustSort(short2* pnLoc_, float* pfResponse_, const unsigned int nCorners_);
	unsigned int cudaPredictAndMatch(const unsigned int uFinalSalientPoints_, const cv::gpu::GpuMat& cvgmImage_,const cv::gpu::GpuMat& cvgmSaliency_, cv::gpu::GpuMat& cvgmFinalKeyPointsLocations_,cv::gpu::GpuMat& cvgmFinalKeyPointsResponse_,cv::gpu::GpuMat& cvgmParticlesAge_,cv::gpu::GpuMat& cvgmParticlesVelocity_, cv::gpu::GpuMat& cvgmParticlesDescriptors_);
	void cudaExtractAllDescriptorFast(const cv::gpu::GpuMat& cvgmImage_, 
									  const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, 
									  const unsigned int uTotalParticles_,  const unsigned int usHalfPatchSize_, 
									  cv::gpu::GpuMat* pcvgmParticleResponses_, cv::gpu::GpuMat* pcvgmParticleDescriptor_ );

	unsigned int cudaTrackFast(float fMatchThreshold_, const unsigned short usHalfSize_, const short sSearchRange_, 
								const cv::gpu::GpuMat& cvgmParticleDescriptorPrev_, const cv::gpu::GpuMat& cvgmParticleResponsesPrev_, 
								const cv::gpu::GpuMat& cvgmParticleDescriptorCurrTmp_, const cv::gpu::GpuMat& cvgmSaliencyCurr_, 
								cv::gpu::GpuMat* pcvgmMinMatchDistance_,
								cv::gpu::GpuMat* pcvgmMatchedLocationPrev_);
	void cudaCollectKeyPointsFast(unsigned int uTotalParticles_, unsigned int uMaxNewKeyPoints_, const float fRho_,
									const cv::gpu::GpuMat& cvgmSaliency_, 
									const cv::gpu::GpuMat& cvgmParticleDescriptorCurrTmp_,
									const cv::gpu::GpuMat& cvgmParticleVelocityPrev_,
									const cv::gpu::GpuMat& cvgmParticleAgePrev_,
									const cv::gpu::GpuMat& cvgmMinMatchDistance_,
									const cv::gpu::GpuMat& cvgmMatchedLocationPrev_, 
									cv::gpu::GpuMat* pcvgmNewlyAddedKeyPointLocation_, cv::gpu::GpuMat* pcvgmNewlyAddedKeyPointResponse_,
									cv::gpu::GpuMat* pcvgmMatchedKeyPointLocation_, cv::gpu::GpuMat* pcvgmMatchedKeyPointResponse_,
									cv::gpu::GpuMat* pcvgmParticleResponseCurr_, cv::gpu::GpuMat* pcvgmParticleDescriptorCurr_,
									cv::gpu::GpuMat* pcvgmParticleVelocityCurr_, cv::gpu::GpuMat* pcvgmParticleAgeCurr_);
}//semidense
}//device
}//btl


#endif