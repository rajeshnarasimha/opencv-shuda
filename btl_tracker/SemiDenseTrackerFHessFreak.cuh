
#ifndef SEMIDENSE_FHESSFREA_CUDA_HEADER
#define SEMIDENSE_FHESSFREAK_CUDA_HEADER
/*


namespace btl{ namespace device{ namespace semidense{
	unsigned int cudaCalcSaliency(const cv::gpu::GpuMat& cvgmImage_, const unsigned short usHalfSizeRound_,
									const unsigned char ucContrastThreshold_, const float& fSaliencyThreshold_, 
									cv::gpu::GpuMat* pcvgmSaliency_, cv::gpu::GpuMat* pcvgmKeyPointLocations_);
	unsigned int cudaNonMaxSupression(const cv::gpu::GpuMat& cvgmKeyPointLocation_, const unsigned int uMaxSalientPoints_, 
										const cv::gpu::GpuMat& cvgmSaliency_, short2* ps2devLocations_, float* pfdevResponse_);
	void thrustSort(short2* pnLoc_, float* pfResponse_, const unsigned int nCorners_);
	void cudaCalcAngles(const cv::gpu::GpuMat& cvgmImage_, const short2* pdevFinalKeyPointsLocations_, const unsigned int uPoints_,  const unsigned short usHalf_, 
						cv::gpu::GpuMat* pcvgmParticleAngle_);
	void loadUMax(const int* pUMax_, int nCount_);
	void cudaExtractAllDescriptorOrb(	const cv::gpu::GpuMat& cvgmImage_,
										const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, 
										const unsigned int uTotalParticles_, const unsigned short usHalfPatchSize_,
										const short* psPatternX_, const short* psPatternY_,
										cv::gpu::GpuMat* pcvgmParticleResponses_, cv::gpu::GpuMat* pcvgmParticleAngle_, cv::gpu::GpuMat* pcvgmParticleDescriptor_);
	//after tracking, the matched particles are filled into the pcvgmParticleResponsesCurr_, pcvgmParticlesAgeCurr_, pcvgmParticlesVelocityCurr_, 
	//and pcvgmParticleOrbDescriptorsCurr_, moreover, the cvgmSaliencyCurr_
	unsigned int cudaTrackOrb(  const short n_, const unsigned short usMatchThreshold_[4], const unsigned short usHalfSize_, const short sSearchRange_,
								const cv::gpu::GpuMat cvgmParticleOrbDescriptorPrev_[4], const cv::gpu::GpuMat cvgmParticleResponsePrev_[4], 
								const cv::gpu::GpuMat cvgmParticleDescriptorCurrTmp_[4],  const cv::gpu::GpuMat cvgmSaliencyCurr_[4],
								cv::gpu::GpuMat pcvgmMinMatchDistance_[4], cv::gpu::GpuMat pcvgmMatchedLocationPrev_[4], cv::gpu::GpuMat pcvgmVelocityBuf_[4]);
	//separate salient point into matched and newly added.
	//for matched keypoints the velocity and age will be updated
	void cudaCollectKeyPointOrb(unsigned int uTotalParticles_, unsigned int uMaxNewKeyPoints_, const float fRho_,
										const cv::gpu::GpuMat& cvgmSaliency_,/ *const cv::gpu::GpuMat& cvgmParticleResponseCurrTmp_,* /
										const cv::gpu::GpuMat& cvgmParticleDescriptorCurrTmp_,
										const cv::gpu::GpuMat& cvgmParticleVelocityPrev_,
										const cv::gpu::GpuMat& cvgmParticleAgePrev_,
										const cv::gpu::GpuMat& cvgmMinMatchDistance_,
										const cv::gpu::GpuMat& cvgmMatchedLocationPrev_,
										cv::gpu::GpuMat* pcvgmNewlyAddedKeyPointLocation_, cv::gpu::GpuMat* pcvgmNewlyAddedKeyPointResponse_,
										cv::gpu::GpuMat* pcvgmMatchedKeyPointLocation_, cv::gpu::GpuMat* pcvgmMatchedKeyPointResponse_,
										cv::gpu::GpuMat* pcvgmParticleResponseCurr_, cv::gpu::GpuMat* pcvgmParticleDescriptorCurr_,
										cv::gpu::GpuMat* pcvgmParticleVelocityCurr_, cv::gpu::GpuMat* pcvgmParticleAgeCurr_);
	void cudaCollectParticles(const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, const unsigned int uTotalParticles_, 
								cv::gpu::GpuMat* pcvgmParticleResponses_, cv::gpu::GpuMat* pcvgmParticleDescriptor_, const cv::gpu::GpuMat& cvgmImage_=cv::gpu::GpuMat() );
	unsigned int cudaMatchedAndNewlyAddedKeyPointsCollection(cv::gpu::GpuMat& cvgmKeyPointLocation_, 
																unsigned int* puMaxSalientPoints_, cv::gpu::GpuMat* pcvgmParticleResponsesCurr_, 
																short2* ps2devMatchedKeyPointLocations_, float* pfdevMatchedKeyPointResponse_, 
																short2* ps2devNewlyAddedKeyPointLocations_, float* pfdevNewlyAddedKeyPointResponse_);
}//semidense
}//device
}//btl
*/

#endif