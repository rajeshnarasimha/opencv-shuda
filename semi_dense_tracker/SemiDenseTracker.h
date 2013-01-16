#ifndef SEMIDENSE_BTL
#define SEMIDENSE_BTL


namespace btl{	namespace image	{
	namespace semidense {

class CSemiDenseTracker{
public:
	CSemiDenseTracker();


	//Gaussian filter
	float _fSigma; // page3: r=3/6 and sigma = 1.f/2.f respectively
	unsigned int _uRadius; // radius of the fast corner
	unsigned int _uGaussianKernelSize;
	unsigned int _uFinalSalientPoints[4];
	//contrast threshold
	unsigned char _ucContrastThresold; // 255 * 0.02 = 5.1
	cv::Ptr<cv::gpu::FilterEngine_GPU> _pBlurFilter; 

	//saliency threshold
	float _fSaliencyThreshold;

	unsigned int _uTotalParticles[4];
	short _sSearchRange;
	unsigned short _usHalfPatchSize; //	the size of a circular region where the patch angle and the orb descriptor are defined

	unsigned int _uMatchedPoints[4]; // the # of successful matches of frome previous frame to current frame

	float _fMatchThreshold;
	//# of Max key points
	unsigned int _uMaxKeyPointsBeforeNonMax[4];
	unsigned int _uMaxKeyPointsAfterNonMax[4];
	//key point locations
	cv::gpu::GpuMat _cvgmBlurredPrev[4];
	cv::gpu::GpuMat _cvgmBlurredCurr[4];

	cv::gpu::GpuMat _cvgmSaliency[4];
	cv::gpu::GpuMat _cvgmInitKeyPointLocation[4];
	//opencv key points
	cv::gpu::GpuMat _cvgmFinalKeyPointsLocationsAfterNonMax[4];
	cv::gpu::GpuMat _cvgmFinalKeyPointsResponseAfterNonMax[4];
	cv::gpu::GpuMat _cvgmParticleVelocityCurr[4];
	cv::gpu::GpuMat _cvgmParticleVelocityPrev[4];
	cv::gpu::GpuMat _cvgmParticleAgeCurr[4];
	cv::gpu::GpuMat _cvgmParticleAgePrev[4];
	cv::gpu::GpuMat _cvgmParticleResponseCurr[4];
	cv::gpu::GpuMat _cvgmParticleResponsePrev[4];
	cv::gpu::GpuMat _cvgmParticleAngleCurr[4];
	cv::gpu::GpuMat _cvgmParticleDescriptorPrev[4];
	cv::gpu::GpuMat _cvgmParticleDescriptorCurr[4];

	cv::gpu::GpuMat _cvgmMatchedKeyPointLocation[4];
	cv::gpu::GpuMat _cvgmMatchedKeyPointResponse[4];
	cv::gpu::GpuMat _cvgmNewlyAddedKeyPointLocation[4];
	cv::gpu::GpuMat _cvgmNewlyAddedKeyPointResponse[4];

	cv::gpu::GpuMat _cvgmParticleDescriptorCurrTmp[4];
	cv::gpu::GpuMat _cvgmMinMatchDistance[4];
	cv::gpu::GpuMat _cvgmMatchedLocationPrev[4];
	
	cv::Mat _cvmKeyPointLocation[4];
	cv::Mat _cvmKeyPointAge[4];
	cv::Mat _cvmKeyPointVelocity[30][4];

	int _nFrameIdx;
	bool init( boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrBW[4] );
	void trackAll(boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrBW[4] );

	virtual bool initialize( boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrBW[4] );
	virtual void track(boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrBW[4] );

	void displayCandidates( cv::Mat& cvmColorFrame_ );
	virtual void display(cv::Mat& cvmColorFrame_) const;

};//class CSemiDenseTracker

}//semidense
}//image
}//btl

#endif