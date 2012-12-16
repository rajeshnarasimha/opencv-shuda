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
	float fSigma; // page3: r=3/6 and sigma = 1.f/2.f respectively
	unsigned int uRadius; // 
	unsigned int uSize;
	//contrast threshold
	unsigned char ucContrastThresold; // 255 * 0.02 = 5.1
	cv::Ptr<cv::gpu::FilterEngine_GPU> _pBlurFilter; 

	//saliency threshold
	float fSaliencyThreshold;

	unsigned int uTotalSalientPoints;
	unsigned int _uTotalParticles;

	//# of Max key points
	unsigned int _uInitMaxKeyPoints;
	unsigned int _uFinalMaxKeyPoints;
	//key point locations
	cv::Mat _cvmGrayFrame;
	cv::Mat _cvmSaliency;
	cv::gpu::GpuMat _cvgmColorFrame;
	cv::gpu::GpuMat _cvgmBlurred;
	cv::gpu::GpuMat _cvgmBlurredPrev;
	cv::gpu::GpuMat _cvgmBlurredCurr;
	cv::gpu::GpuMat _cvgmBufferC1;
	cv::gpu::GpuMat _cvgmSaliency;
	cv::gpu::GpuMat _cvgmInitKeyPointLocation;
	//opencv key points
	cv::gpu::GpuMat _cvgmFinalKeyPointsLocations;
	cv::gpu::GpuMat _cvgmFinalKeyPointsResponse;
	cv::gpu::GpuMat _cvgmParticlesVelocityCurr;
	cv::gpu::GpuMat _cvgmParticlesVelocityPrev;
	cv::gpu::GpuMat _cvgmParticlesAgeCurr;
	cv::gpu::GpuMat _cvgmParticlesAgePrev;
	cv::gpu::GpuMat _cvgmParticleResponsesCurr;
	cv::gpu::GpuMat _cvgmParticleResponsesPrev;
	cv::gpu::GpuMat _cvgmParticlesDescriptors;

	cv::gpu::GpuMat _cvgmMatchedKeyPointsLocations;
	cv::gpu::GpuMat _cvgmMatchedKeyPointsResponse;
	cv::gpu::GpuMat _cvgmNewlyAddedKeyPointsLocations;
	cv::gpu::GpuMat _cvgmNewlyAddedKeyPointsResponse;
	cv::Mat _cvmKeyPointsLocations;


	void init(cv::Mat& cvmColorFrame_);
	void initialize( cv::Mat& cvmColorFrame_ );
	void track( cv::Mat& cvmColorFrame_ );
	void tracking(cv::Mat& cvmColorFrame_);

};//class CSemiDenseTracker

}//semidense
}//image
}//btl

#endif