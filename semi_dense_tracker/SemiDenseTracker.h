#ifndef SEMIDENSE_BTL
#define SEMIDENSE_BTL


namespace btl{	namespace image	{
	namespace semidense {

class CSemiDenseTracker{
	enum
	{
		LOCATION_ROW = 0,
		RESPONSE_ROW,
		VELOCITY_ROW,
		AGE_ROW,
		DESCRIPTOR_ROW1,
		DESCRIPTOR_ROW2,
		DESCRIPTOR_ROW3,
		DESCRIPTOR_ROW4
	};
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
	cv::gpu::GpuMat _cvgmParticlesVelocityCurr;
	cv::gpu::GpuMat _cvgmParticlesVelocityPrev;
	cv::gpu::GpuMat _cvgmParticlesAgeCurr;
	cv::gpu::GpuMat _cvgmParticlesAgePrev;
	cv::gpu::GpuMat _cvgmParticleResponsesCurr;
	cv::gpu::GpuMat _cvgmParticleResponsesPrev;
	cv::gpu::GpuMat _cvgmParticleAnglePrev;
	cv::gpu::GpuMat _cvgmParticleAngleCurr;
	cv::gpu::GpuMat _cvgmParticleDescriptorsPrev;
	cv::gpu::GpuMat _cvgmParticleDescriptorsCurr;


	cv::gpu::GpuMat _cvgmMatchedKeyPointsLocations;
	cv::gpu::GpuMat _cvgmMatchedKeyPointsResponse;
	cv::gpu::GpuMat _cvgmNewlyAddedKeyPointsLocations;
	cv::gpu::GpuMat _cvgmNewlyAddedKeyPointsResponse;
	cv::Mat _cvmKeyPointsLocations;
	cv::Mat _cvmKeyPointsAge;
	cv::Mat _cvmKeyPointVelocitys[30];

	int _nFrameIdx;

	virtual void initialize( cv::Mat& cvmColorFrame_ );
	virtual void track( cv::Mat& cvmColorFrame_ );

	void trackTest( cv::Mat& cvmColorFrame_ );
	void displayCandidates( cv::Mat& cvmColorFrame_ );

};//class CSemiDenseTracker

}//semidense
}//image
}//btl

#endif