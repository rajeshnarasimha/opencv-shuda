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
	unsigned int _uFinalSalientPoints;
	//contrast threshold
	unsigned char _ucContrastThresold; // 255 * 0.02 = 5.1
	cv::Ptr<cv::gpu::FilterEngine_GPU> _pBlurFilter; 

	//saliency threshold
	float _fSaliencyThreshold;

	unsigned int _uTotalParticles;
	short _sSearchRange;
	unsigned short _usHalfPatchSize; //	the size of a circular region where the patch angle and the orb descriptor are defined

	//# of Max key points
	unsigned int _uMaxKeyPointsBeforeNonMax;
	unsigned int _uMaxKeyPointsAfterNonMax;
	//key point locations
	cv::Mat _cvmGrayFrame;
	cv::Mat _cvmSaliency;
	cv::gpu::GpuMat _cvgmPattern;
	cv::gpu::GpuMat _cvgmColorFrame;
	cv::gpu::GpuMat _cvgmGrayFrame;
	cv::gpu::GpuMat _cvgmBlurredPrev;
	cv::gpu::GpuMat _cvgmBlurredCurr;

	cv::gpu::GpuMat _cvgmSaliency;
	cv::gpu::GpuMat _cvgmInitKeyPointLocation;
	//opencv key points
	cv::gpu::GpuMat _cvgmFinalKeyPointsLocationsAfterNonMax;
	cv::gpu::GpuMat _cvgmFinalKeyPointsResponseAfterNonMax;
	cv::gpu::GpuMat _cvgmParticleVelocityCurr;
	cv::gpu::GpuMat _cvgmParticleVelocityPrev;
	cv::gpu::GpuMat _cvgmParticleAgeCurr;
	cv::gpu::GpuMat _cvgmParticleAgePrev;
	cv::gpu::GpuMat _cvgmParticleResponseCurr;
	cv::gpu::GpuMat _cvgmParticleResponsePrev;
	cv::gpu::GpuMat _cvgmParticleAnglePrev;
	cv::gpu::GpuMat _cvgmParticleAngleCurr;
	cv::gpu::GpuMat _cvgmParticleDescriptorPrev;
	cv::gpu::GpuMat _cvgmParticleDescriptorCurr;

	cv::gpu::GpuMat _cvgmMatchedKeyPointLocation;
	cv::gpu::GpuMat _cvgmMatchedKeyPointResponse;
	cv::gpu::GpuMat _cvgmNewlyAddedKeyPointLocation;
	cv::gpu::GpuMat _cvgmNewlyAddedKeyPointResponse;

	cv::gpu::GpuMat _cvgmParticleDescriptorCurrTmp;
	cv::gpu::GpuMat _cvgmMinMatchDistance;
	cv::gpu::GpuMat _cvgmMatchedLocationPrev;
	
	cv::Mat _cvmKeyPointLocation;
	cv::Mat _cvmKeyPointAge;
	cv::Mat _cvmKeyPointVelocity[30];

	int _nFrameIdx;

	virtual bool initialize( cv::Mat& cvmColorFrame_ );
	virtual void track( cv::Mat& cvmColorFrame_ );

	void displayCandidates( cv::Mat& cvmColorFrame_ );

};//class CSemiDenseTracker

}//semidense
}//image
}//btl

#endif