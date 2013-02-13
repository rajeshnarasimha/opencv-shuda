


float testMatDiff(const cv::gpu::GpuMat& cvgm1_,const cv::gpu::GpuMat& cvgm2_ );
void testCudaCollectParticlesAndOrbDescriptors(const cv::gpu::GpuMat& cvgmFinalKeyPointsLocationsAfterNonMax_, const cv::gpu::GpuMat& cvmFinalKeyPointsResponseAfterNonMax_, const cv::gpu::GpuMat& cvgmImage_,
													const unsigned int uTotalParticles_, const unsigned short _usHalfPatchSize,
													const cv::gpu::GpuMat& cvgmPattern_, const short sDescriptorByte_,
													cv::gpu::GpuMat* pcvgmParticleResponses_, cv::gpu::GpuMat* pcvgmParticleAngle_, cv::gpu::GpuMat* pcvgmParticleDescriptor_);

unsigned int testCudaTrackOrb(  const short n_, const unsigned short usMatchThreshold_[4], const unsigned short usHalfSize_, const short sSearchRange_, const short sDescriptorByte_,
						  const cv::gpu::GpuMat cvgmParticleOrbDescriptorPrev_[4], const cv::gpu::GpuMat cvgmParticleResponsePrev_[4], 
						  const cv::gpu::GpuMat cvgmParticleDescriptorCurrTmp_[4],  const cv::gpu::GpuMat cvgmSaliencyCurr_[4],
						  cv::gpu::GpuMat pcvgmMinMatchDistance_[4], cv::gpu::GpuMat pcvgmMatchedLocationPrev_[4], cv::gpu::GpuMat pcvgmVelocityBuf_[4]);
void testCudaCollectNewlyAddedKeyPoints(unsigned int uTotalParticles_, unsigned int uMaxNewKeyPoints_, const float fRho_, const short sDescritporByte_,
										const cv::gpu::GpuMat& cvgmSaliency_,/*const cv::gpu::GpuMat& cvgmParticleResponseCurrTmp_,*/
										const cv::gpu::GpuMat& cvgmParticleDescriptorCurrTmp_,
										const cv::gpu::GpuMat& cvgmParticleVelocityPrev_,
										const cv::gpu::GpuMat& cvgmParticleAgePrev_,
										const cv::gpu::GpuMat& cvgmMinMatchDistance_,
										const cv::gpu::GpuMat& cvgmMatchedLocationPrev_,
										cv::gpu::GpuMat* pcvgmNewlyAddedKeyPointLocation_, cv::gpu::GpuMat* pcvgmNewlyAddedKeyPointResponse_,
										cv::gpu::GpuMat* pcvgmMatchedKeyPointLocation_, cv::gpu::GpuMat* pcvgmMatchedKeyPointResponse_,
										cv::gpu::GpuMat* pcvgmParticleResponseCurr_, cv::gpu::GpuMat* pcvgmParticleDescriptorCurr_,
										cv::gpu::GpuMat* pcvgmParticleVelocityCurr_, cv::gpu::GpuMat* pcvgmParticleAgeCurr_);
bool testCountMinDistAndMatchedLocation(const cv::gpu::GpuMat cvgmMinMatchDistance_, const cv::gpu::GpuMat& cvgmMatchedLocationPrev_, int* pnCounter_);
bool testCountResponseAndDescriptor(const cv::gpu::GpuMat cvgmParticleResponse_, const cv::gpu::GpuMat& cvgmParticleDescriptor_, int* pnCounter_,const short sDescritporByte_);
