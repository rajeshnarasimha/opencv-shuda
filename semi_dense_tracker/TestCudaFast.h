


unsigned int testCudaTrackFast(float fMatchThreshold_, const unsigned short usHalfSize_, const short sSearchRange_, 
							   const cv::gpu::GpuMat& cvgmParticleDescriptorPrev_, const cv::gpu::GpuMat& cvgmParticleResponsesPrev_, 
							   const cv::gpu::GpuMat& cvgmParticleDescriptorCurrTmp_, const cv::gpu::GpuMat& cvgmSaliencyCurr_, 
							   cv::gpu::GpuMat* pcvgmMinMatchDistance_,
							   cv::gpu::GpuMat* pcvgmMatchedLocationPrev_);
void testCudaCollectKeyPointsFast(unsigned int uTotalParticles_, unsigned int uMaxNewKeyPoints_, const float fRho_,
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
bool testCountMinDistAndMatchedLocationFast(const cv::gpu::GpuMat cvgmMinMatchDistance_, const cv::gpu::GpuMat& cvgmMatchedLocationPrev_, int* pnCounter_);
bool testCountResponseAndDescriptorFast(const cv::gpu::GpuMat cvgmParticleResponse_, const cv::gpu::GpuMat& cvgmParticleDescriptor_, int* pnCounter_);
float testMatDiff(const cv::gpu::GpuMat& cvgm1_,const cv::gpu::GpuMat& cvgm2_ );
