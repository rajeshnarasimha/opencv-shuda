#ifndef SEMIDENSE_FREAK_BTL
#define SEMIDENSE_FREAK_BTL


namespace btl{	namespace image	{
namespace semidense {

class CSemiDenseTrackerFHessFreak: public CSemiDenseTracker{
public:

	unsigned short _usMatchThreshod[4];
	cv::gpu::GpuMat _cvgmPattern;

	CSemiDenseTrackerFHessFreak(unsigned int uPyrHeight_);
	virtual bool initialize( boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrBW[4] );
	virtual void track( boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrBW[4] );

private:
	void initUMax();
	void initOrbPattern();
	void makeRandomPattern(unsigned short usHalfPatchSize_, int nPoints_, cv::Mat* pcvmPattern_);

	//Hessian detector
	boost::scoped_ptr<cv::gpu::SURF_GPU> _pSURF;
	cv::gpu::GpuMat _cvgmKeyPoint[4];

	short _sDescriptorByte;
};//class CSemiDenseTrackerFHessFreak

}//semidense
}//image
}//btl

#endif