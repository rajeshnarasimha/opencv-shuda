#ifndef SEMIDENSE_ORB_BTL
#define SEMIDENSE_ORB_BTL


namespace btl{	namespace image	{
	namespace semidense {

class CSemiDenseTrackerOrb: public CSemiDenseTracker{
public:
	typedef boost::shared_ptr< CSemiDenseTrackerOrb > tp_shared_ptr;
	typedef boost::scoped_ptr< CSemiDenseTrackerOrb > tp_scoped_ptr;

	unsigned short _usMatchThreshod[4];
	cv::gpu::GpuMat _cvgmPattern;

	CSemiDenseTrackerOrb();
	virtual bool initialize( boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrBW[4] );
	virtual void track( boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrBW[4] );

private:
	void initUMax();
	void initOrbPattern();
	void makeRandomPattern(unsigned short usHalfPatchSize_, int nPoints_, cv::Mat* pcvmPattern_);

};//class CSemiDenseTrackerOrb

}//semidense
}//image
}//btl

#endif