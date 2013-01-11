#ifndef SEMIDENSE_ORB_BTL
#define SEMIDENSE_ORB_BTL


namespace btl{	namespace image	{
	namespace semidense {

class CSemiDenseTrackerOrb: public CSemiDenseTracker{
public:

	//Gaussian filter

	cv::gpu::GpuMat _cvgmPattern;

	CSemiDenseTrackerOrb();
	virtual bool initialize( cv::Mat& cvmColorFrame_ );
	virtual void track( cv::Mat& cvmColorFrame_ );

private:
	void initUMax();
	void initOrbPattern();
	void makeRandomPattern(unsigned short usHalfPatchSize_, int nPoints_, cv::Mat* pcvmPattern_);

};//class CSemiDenseTrackerOrb

}//semidense
}//image
}//btl

#endif