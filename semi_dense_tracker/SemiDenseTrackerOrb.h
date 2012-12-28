#ifndef SEMIDENSE_ORB_BTL
#define SEMIDENSE_ORB_BTL


namespace btl{	namespace image	{
	namespace semidense {

class CSemiDenseTrackerOrb: public CSemiDenseTracker{
public:

	//Gaussian filter
	unsigned short _usHalfPatchSize; //	the size of a circular region where the patch angle and the orb descriptor are defined

	cv::gpu::GpuMat _cvgmPattern;
	cv::gpu::GpuMat _cvgmParticleOrbDescriptorsPrev;
	cv::gpu::GpuMat _cvgmParticleOrbDescriptorsCurr;
	//opencv key points


	CSemiDenseTrackerOrb();
	virtual void initialize( cv::Mat& cvmColorFrame_ );
	virtual void track( cv::Mat& cvmColorFrame_ );

	void trackTest( cv::Mat& cvmColorFrame_ );
	void initUMax();
	void trackOrb( cv::Mat& cvmColorFrame_ );
	void initOrbPattern();
	void makeRandomPattern(unsigned short usHalfPatchSize_, int nPoints_, cv::Mat* pcvmPattern_);

};//class CSemiDenseTrackerOrb

}//semidense
}//image
}//btl

#endif