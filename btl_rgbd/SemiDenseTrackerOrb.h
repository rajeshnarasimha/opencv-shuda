#ifndef SEMIDENSE_ORB_BTL
#define SEMIDENSE_ORB_BTL


namespace btl{	namespace image	{
namespace semidense {

class CSemiDenseTrackerOrb: public CSemiDenseTracker{
public:
	//type definition
	typedef boost::scoped_ptr<CSemiDenseTrackerOrb> tp_scoped_ptr;
	typedef boost::shared_ptr<CSemiDenseTrackerOrb> tp_shared_ptr;

	unsigned short _usMatchThreshod[4];

	CSemiDenseTrackerOrb(unsigned int uPyrHeight_);
	virtual bool initialize( boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrBW[4] );
	virtual void track( boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrBW[4] );
private:
	void initUMax();
	void initOrbPattern();
	void makeRandomPattern(unsigned short usHalfPatchSize_, int nPoints_, cv::Mat* pcvmPattern_);
	short _sDescriptorByte;
};//class CSemiDenseTrackerOrb

}//semidense
}//image
}//btl

#endif