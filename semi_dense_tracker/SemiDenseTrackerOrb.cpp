#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "SemiDenseTracker.h"
#include "SemiDenseTrackerOrb.h"

#include <cuda.h>
#include <cuda_runtime.h>

float testMatDiff(const cv::gpu::GpuMat& cvgm1_,const cv::gpu::GpuMat& cvgm2_ );
void testCudaCollectParticlesAndOrbDescriptors(const cv::gpu::GpuMat& cvgmFinalKeyPointsLocationsAfterNonMax_, const cv::gpu::GpuMat& cvmFinalKeyPointsResponseAfterNonMax_, const cv::gpu::GpuMat& cvgmImage_,
	const unsigned int uTotalParticles_, const unsigned short _usHalfPatchSize,
	const cv::gpu::GpuMat& cvgmPattern_,
	cv::gpu::GpuMat* pcvgmParticleResponses_, cv::gpu::GpuMat* pcvgmParticleAngle_, cv::gpu::GpuMat* pcvgmParticleDescriptor_);
unsigned int testCudaTrackOrb(const unsigned short usMatchThreshold_, const unsigned short usHalfSize_, 
	const short* psPatternX_, const short* psPatternY_,
	const cv::gpu::GpuMat& cvgmParticleOrbDescriptorsPrev_, const cv::gpu::GpuMat& cvgmParticleResponsesPrev_, 
	const cv::gpu::GpuMat& cvgmParticlesAgePrev_,const cv::gpu::GpuMat& cvgmParticlesVelocityPrev_, 
	const cv::gpu::GpuMat& cvgmImage_, const cv::gpu::GpuMat& cvgmParticleAngleCurr_,
	cv::gpu::GpuMat* pcvgmParticleResponsesCurr_,
	cv::gpu::GpuMat* pcvgmParticlesAgeCurr_,cv::gpu::GpuMat* pcvgmParticlesVelocityCurr_,cv::gpu::GpuMat* pcvgmParticleOrbDescriptorsCurr_);

namespace btl{ namespace device{ namespace semidense{
	unsigned int cudaCalcSaliency(const cv::gpu::GpuMat& cvgmImage_, const unsigned char ucContrastThreshold_, const float& fSaliencyThreshold_, cv::gpu::GpuMat* pcvgmSaliency_, cv::gpu::GpuMat* pcvgmKeyPointLocations_);
	unsigned int cudaNonMaxSupression(const cv::gpu::GpuMat& cvgmKeyPointLocation_, const unsigned int uMaxSalientPoints_, 
		const cv::gpu::GpuMat& cvgmSaliency_, short2* ps2devLocations_, float* pfdevResponse_);
	void thrustSort(short2* pnLoc_, float* pfResponse_, const unsigned int nCorners_);
	void cudaCalcAngles(const cv::gpu::GpuMat& cvgmImage_, const short2* pdevFinalKeyPointsLocations_, const unsigned int uPoints_,  const unsigned short usHalf_, 
		cv::gpu::GpuMat* pcvgmParticleAngle_);
	void loadUMax(const int* pUMax_, int nCount_);
	void cudaCollectParticlesAndOrbDescriptors(
		const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, const cv::gpu::GpuMat& cvgmImage_,
		const unsigned int uTotalParticles_, const unsigned short _usHalfPatchSize,
		const short* psPatternX_, const short* psPatternY_,
		cv::gpu::GpuMat* pcvgmParticleResponses_, cv::gpu::GpuMat* pcvgmParticleAngle_, cv::gpu::GpuMat* pcvgmParticleDescriptor_);
	unsigned int cudaTrackOrb(const unsigned short usMatchThreshold_, const unsigned short usHalfSize_, const short sSearchRange_,
		const short* psPatternX_, const short* psPatternY_,
		const cv::gpu::GpuMat& cvgmParticleOrbDescriptorsPrev_, const cv::gpu::GpuMat& cvgmParticleResponsesPrev_, 
		const cv::gpu::GpuMat& cvgmParticlesAgePrev_,const cv::gpu::GpuMat& cvgmParticlesVelocityPrev_, 
		const cv::gpu::GpuMat& cvgmBlurredCurr_, const cv::gpu::GpuMat& cvgmParticleAngleCurr_,
		cv::gpu::GpuMat* pcvgmParticleResponsesCurr_,
		cv::gpu::GpuMat* pcvgmParticlesAgeCurr_,cv::gpu::GpuMat* pcvgmParticlesVelocityCurr_,cv::gpu::GpuMat* pcvgmParticleOrbDescriptorsCurr_);
	void cudaCollectParticles(const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, const unsigned int uTotalParticles_, 
		cv::gpu::GpuMat* pcvgmParticleResponses_, cv::gpu::GpuMat* pcvgmParticleDescriptor_, const cv::gpu::GpuMat& cvgmImage_=cv::gpu::GpuMat() );
	unsigned int cudaMatchedAndNewlyAddedKeyPointsCollection(cv::gpu::GpuMat& cvgmKeyPointLocation_, 
		unsigned int* puMaxSalientPoints_, cv::gpu::GpuMat* pcvgmParticleResponsesCurr_, 
		short2* ps2devMatchedKeyPointLocations_, float* pfdevMatchedKeyPointResponse_, 
		short2* ps2devNewlyAddedKeyPointLocations_, float* pfdevNewlyAddedKeyPointResponse_);
}//semidense
}//device
}//btl

__device__ short2 operator + (const short2 s2O1_, const short2 s2O2_);
__device__ short2 operator - (const short2 s2O1_, const short2 s2O2_);
__device__ short2 operator * (const float fO1_, const short2 s2O2_);

btl::image::semidense::CSemiDenseTrackerOrb::CSemiDenseTrackerOrb()
{
	//Gaussian filter
	_fSigma = 1.f; // page3: r=3/6 and sigma = 1.f/2.f respectively
	_uRadius = 3; // 
	_uGaussianKernelSize = 2*_uRadius + 1;
	//contrast threshold
	_ucContrastThresold = 5; // 255 * 0.02 = 5.1

	//saliency threshold
	_fSaliencyThreshold = 0.25;

	//# of Max key points
	_uMaxKeyPointsBeforeNonMax = 50000;
	_uMaxKeyPointsAfterNonMax= 20000;
	_uTotalParticles = 2000;

	_usHalfPatchSize = 10; //the size of the orb feature
	_sSearchRange = 10;

	_nFrameIdx = 0;
}


void btl::image::semidense::CSemiDenseTrackerOrb::initUMax(){
	// pre-compute the end of a row in a circular patch (1/4 of the circular patch)
	int half_patch_size = _usHalfPatchSize;
	std::vector<int> u_max(half_patch_size + 2);
	for (int v = 0; v <= half_patch_size * std::sqrt(2.f) / 2 + 1; ++v)
		u_max[v] = cvRound(std::sqrt(static_cast<float>(half_patch_size * half_patch_size - v * v)));

	// Make sure we are symmetric
	for (int v = half_patch_size, v_0 = 0; v >= half_patch_size * std::sqrt(2.f) / 2; --v)
	{
		while (u_max[v_0] == u_max[v_0 + 1])
			++v_0;
		u_max[v] = v_0;
		++v_0;
	}
	btl::device::semidense::loadUMax(&u_max[0], static_cast<int>(u_max.size()));
}
void btl::image::semidense::CSemiDenseTrackerOrb::makeRandomPattern(unsigned short usHalfPatchSize_, int nPoints_, cv::Mat* pcvmPattern_)
{
	// we always start with a fixed seed,
	// to make patterns the same on each run
	cv::RNG rng(0x34985739);

	for (int i = 0; i < nPoints_; i++){
		pcvmPattern_->ptr<short>(0)[i] = rng.uniform(- usHalfPatchSize_, usHalfPatchSize_ + 1);
		pcvmPattern_->ptr<short>(1)[i] = rng.uniform(- usHalfPatchSize_, usHalfPatchSize_ + 1);
	}
}
void btl::image::semidense::CSemiDenseTrackerOrb::initOrbPattern(){
	// Calc cvmPattern_
	const int nPoints = 128; // 64 tests and each test requires 2 points 256x2 = 512
	cv::Mat cvmPattern; //2 x n : 1st row is x and 2nd row is y; test point1, test point2;
	//assign cvmPattern_ from precomputed patterns
	cvmPattern.create(2, nPoints, CV_16SC1);
	makeRandomPattern(_usHalfPatchSize, nPoints, &cvmPattern );
	_cvgmPattern.upload(cvmPattern);//2 x n : 1st row is x and 2nd row is y; test point1, test point2;
	return;
}

void increase(const int nCycle_,int* pnIdx_ ){
	++*pnIdx_;
	*pnIdx_ = *pnIdx_ < nCycle_? *pnIdx_: *pnIdx_-nCycle_;
	*pnIdx_ = *pnIdx_ < 0?       *pnIdx_+nCycle_: *pnIdx_;
}
void decrease(const int nCycle_,int* pnIdx_ ){
	--*pnIdx_;
	*pnIdx_ = *pnIdx_ < nCycle_? *pnIdx_: *pnIdx_-nCycle_;
	*pnIdx_ = *pnIdx_ < 0?       *pnIdx_+nCycle_: *pnIdx_;
}



void btl::image::semidense::CSemiDenseTrackerOrb::initialize( cv::Mat& cvmColorFrame_ )
{
	_nFrameIdx=0;
	initUMax();
	initOrbPattern();
	_cvgmColorFrame.upload(cvmColorFrame_);
	cv::gpu::cvtColor(_cvgmColorFrame,_cvgmGrayFrame,cv::COLOR_RGB2GRAY);
	_cvgmSaliency.create(cvmColorFrame_.size(),CV_32FC1);
	
	_cvgmInitKeyPointLocation.create(1, _uMaxKeyPointsBeforeNonMax, CV_16SC2);
	_cvgmFinalKeyPointsLocationsAfterNonMax.create(1, _uMaxKeyPointsAfterNonMax, CV_16SC2);//short2 location;
	_cvgmFinalKeyPointsResponseAfterNonMax.create(1, _uMaxKeyPointsAfterNonMax, CV_32FC1);//float corner strength(response);  

	_cvgmMatchedKeyPointsLocations.create(1, _uTotalParticles, CV_16SC2);
	_cvgmMatchedKeyPointsResponse.create(1, _uTotalParticles, CV_32FC1);
	_cvgmNewlyAddedKeyPointsLocations.create(1, _uMaxKeyPointsAfterNonMax, CV_16SC2);
	_cvgmNewlyAddedKeyPointsResponse.create(1, _uMaxKeyPointsAfterNonMax, CV_32FC1);

	//init particles
	_cvgmParticleResponsesPrev.create(cvmColorFrame_.size(),CV_32FC1);	   _cvgmParticleResponsesPrev.setTo(0);
	_cvgmParticleResponsesCurr.create(cvmColorFrame_.size(),CV_32FC1);	   _cvgmParticleResponsesCurr.setTo(0);
	_cvgmParticleAnglePrev.create(cvmColorFrame_.size(),CV_32FC1);		   _cvgmParticleAnglePrev.setTo(0);
	_cvgmParticleAngleCurr.create(cvmColorFrame_.size(),CV_32FC1);		   _cvgmParticleAngleCurr.setTo(0);
	_cvgmParticlesVelocityPrev.create(cvmColorFrame_.size(),CV_16SC2);	   _cvgmParticlesVelocityPrev.setTo(cv::Scalar::all(0));//float velocity; 
	_cvgmParticlesVelocityCurr.create(cvmColorFrame_.size(),CV_16SC2);	   _cvgmParticlesVelocityCurr.setTo(cv::Scalar::all(0));//float velocity; 
	_cvgmParticlesAgePrev.create(cvmColorFrame_.size(),CV_8UC1);	       _cvgmParticlesAgePrev.setTo(0);//uchar age;
	_cvgmParticlesAgeCurr.create(cvmColorFrame_.size(),CV_8UC1);		   _cvgmParticlesAgeCurr.setTo(0);//uchar age;
	_cvgmParticleOrbDescriptorsPrev.create(cvmColorFrame_.size(),CV_32SC2);_cvgmParticleOrbDescriptorsPrev.setTo(cv::Scalar::all(0));
	_cvgmParticleOrbDescriptorsCurr.create(cvmColorFrame_.size(),CV_32SC2);_cvgmParticleOrbDescriptorsCurr.setTo(cv::Scalar::all(0));

	//allocate filter
	_pBlurFilter = cv::gpu::createGaussianFilter_GPU(CV_8UC1, cv::Size(_uGaussianKernelSize, _uGaussianKernelSize), _fSigma, _fSigma, cv::BORDER_REFLECT_101);

	//processing the frame
	//0.gaussian filter 
	//a) cvgmBlurred(i) = Gaussian(cvgmImage); // gaussian filter the input image 
	_pBlurFilter->apply(_cvgmGrayFrame, _cvgmBlurredPrev, cv::Rect(0, 0, _cvgmGrayFrame.cols, _cvgmGrayFrame.rows));
	//detect key points
	//1.compute the saliency score 
	//b) cvgmResponse = ExtractSalientPixels(cvgmBlurred(i));
	unsigned int uTotalSalientPoints = btl::device::semidense::cudaCalcSaliency(_cvgmBlurredPrev, _ucContrastThresold, _fSaliencyThreshold, &_cvgmSaliency, &_cvgmInitKeyPointLocation);
	uTotalSalientPoints = std::min( uTotalSalientPoints, _uMaxKeyPointsBeforeNonMax );
	
	//2.do a non-max suppression and initialize particles ( extract feature descriptors ) 
	//c) cvgmSupressed, KeyPoints, Response = NonMaxSupression(cvgmResponse);
	unsigned int uFinalSalientPointsAfterNonMax = btl::device::semidense::cudaNonMaxSupression(_cvgmInitKeyPointLocation, uTotalSalientPoints, _cvgmSaliency, 
		_cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(), _cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>() );
	uFinalSalientPointsAfterNonMax = std::min( uFinalSalientPointsAfterNonMax, _uMaxKeyPointsAfterNonMax );
	
	//3.sort all salient points according to their strength 
	//d) cvgmParitclesResponse(i) = Sort(KeyPoints, Response, cvgmSupressed, N); //choose top N strongest salient pixels are particles
	btl::device::semidense::thrustSort(_cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(),_cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>(),uFinalSalientPointsAfterNonMax);
	_uTotalParticles = std::min( _uTotalParticles, uFinalSalientPointsAfterNonMax );
	_cvgmParticleResponsesPrev.setTo(0.f);

	btl::device::semidense::cudaCollectParticlesAndOrbDescriptors(
		_cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(),_cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>(),_cvgmBlurredPrev,
		_uTotalParticles,_usHalfPatchSize,
		_cvgmPattern.ptr<short>(0),_cvgmPattern.ptr<short>(1),
		&_cvgmParticleResponsesPrev, &_cvgmParticleAnglePrev, &_cvgmParticleOrbDescriptorsPrev);

	/*cv::gpu::GpuMat cvgmTestResponse(_cvgmParticleResponsesPrev); cvgmTestResponse.setTo(0.f);
	cv::gpu::GpuMat cvgmTestOrbDescriptor(_cvgmParticleOrbDescriptorsPrev);cvgmTestOrbDescriptor.setTo(cv::Scalar::all(0));
	testCudaCollectParticlesAndOrbDescriptors(_cvgmFinalKeyPointsLocationsAfterNonMax,_cvgmFinalKeyPointsResponseAfterNonMax,_cvgmBlurredPrev,_uTotalParticles,_usHalfPatchSize,_cvgmPattern,
		&cvgmTestResponse,&_cvgmParticleAnglePrev,&cvgmTestOrbDescriptor);
	float fD1 = testMatDiff(_cvgmParticleResponsesPrev, cvgmTestResponse);
	float fD2 = testMatDiff(_cvgmParticleOrbDescriptorsPrev, cvgmTestOrbDescriptor);*/
	_cvgmParticlesVelocityPrev.download(_cvmKeyPointVelocitys[_nFrameIdx]);
	increase(30,&_nFrameIdx);

	//render keypoints
	//store veloctiy
	_cvgmParticlesVelocityCurr.download(_cvmKeyPointVelocitys[_nFrameIdx]);
	cvmColorFrame_.setTo(cv::Scalar::all(255));
	cv::putText(cvmColorFrame_, "Proposed", cv::Point(10, 15), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(1.,0.,0.) );

	return;
}

void btl::image::semidense::CSemiDenseTrackerOrb::track( cv::Mat& cvmColorFrame_ )
{
	_cvgmColorFrame.upload(cvmColorFrame_);
	cv::gpu::cvtColor(_cvgmColorFrame,_cvgmGrayFrame,cv::COLOR_RGB2GRAY);
	//processing the frame
	//a) cvgmBlurred(i+1) = Gaussian(cvgmImage); // gaussian filter the input image 
	_pBlurFilter->apply(_cvgmGrayFrame, _cvgmBlurredCurr, cv::Rect(0, 0, _cvgmGrayFrame.cols, _cvgmGrayFrame.rows));
	//b) cvgmResponse = ExtractSalientPixels(cvgmBlurred(i+1))
	unsigned int uTotalSalientPoints = btl::device::semidense::cudaCalcSaliency(_cvgmBlurredCurr, _ucContrastThresold, _fSaliencyThreshold, &_cvgmSaliency, &_cvgmInitKeyPointLocation);
	uTotalSalientPoints = std::min( uTotalSalientPoints, _uMaxKeyPointsBeforeNonMax );
	
	//2.do a non-max suppression and initialize particles ( extract feature descriptors ) 
	//c) cvgmSupressed, KeyPoints, Response = NonMaxSupression(cvgmResponse);
	unsigned int uFinalSalientPoints = btl::device::semidense::cudaNonMaxSupression(_cvgmInitKeyPointLocation, uTotalSalientPoints, _cvgmSaliency, 
		_cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(), _cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>() );
	_uFinalSalientPoints = uFinalSalientPoints = std::min( uFinalSalientPoints, unsigned int(_uMaxKeyPointsAfterNonMax) );
	_cvgmSaliency.setTo(0.f);
	btl::device::semidense::cudaCollectParticles(_cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(),_cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>(),uFinalSalientPoints,
		&_cvgmSaliency,NULL);
	//c.a) calculate the angles of fast corners
	btl::device::semidense::cudaCalcAngles(_cvgmBlurredCurr,_cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(),uFinalSalientPoints,_usHalfPatchSize,&_cvgmParticleAngleCurr);
	
/*
	cv::gpu::GpuMat cvgmSaliencyTest; _cvgmSaliency.copyTo(cvgmSaliencyTest);
	cv::Mat cvmPattern; _cvgmPattern.download(cvmPattern);
	unsigned int uDeletedPointsTest = testCudaTrackOrb(32,_usHalfPatchSize,
		cvmPattern.ptr<short>(0),cvmPattern.ptr<short>(1),
		_cvgmParticleOrbDescriptorsPrev, _cvgmParticleResponsesPrev, 
		_cvgmParticlesAgePrev, _cvgmParticlesVelocityPrev, 
		_cvgmBlurredCurr, _cvgmParticleAngleCurr,
		&cvgmSaliencyTest,
		&_cvgmParticlesAgeCurr, &_cvgmParticlesVelocityCurr, &_cvgmParticleOrbDescriptorsCurr);
*/


	//d) for each PixelLocation in cvgmParitclesResponse(i)
	unsigned int uDeletedPoints = btl::device::semidense::cudaTrackOrb(16,_usHalfPatchSize,_sSearchRange,
		_cvgmPattern.ptr<short>(0),_cvgmPattern.ptr<short>(1),
		_cvgmParticleOrbDescriptorsPrev, _cvgmParticleResponsesPrev, 
		_cvgmParticlesAgePrev, _cvgmParticlesVelocityPrev, 
		_cvgmBlurredCurr, _cvgmParticleAngleCurr,
		&_cvgmSaliency,
		&_cvgmParticlesAgeCurr, &_cvgmParticlesVelocityCurr, &_cvgmParticleOrbDescriptorsCurr);

	
	//e) KeyPoints, Response = NonMaxSupressionAndCollection(cvgmResponse );
	unsigned int uNewlyAdded = btl::device::semidense::cudaMatchedAndNewlyAddedKeyPointsCollection(_cvgmFinalKeyPointsLocationsAfterNonMax, 
		&uFinalSalientPoints, &_cvgmSaliency, 
		_cvgmMatchedKeyPointsLocations.ptr<short2>(), _cvgmMatchedKeyPointsResponse.ptr<float>(), 
		_cvgmNewlyAddedKeyPointsLocations.ptr<short2>(), _cvgmNewlyAddedKeyPointsResponse.ptr<float>() );
	uNewlyAdded = std::min(uNewlyAdded,_uMaxKeyPointsAfterNonMax);
	unsigned int uMatched = std::min(uFinalSalientPoints,_uTotalParticles);
	//f) cvgmParticlesResponse(i+1) = Sort( KeyPoint, Response, uDeletePoints)
	btl::device::semidense::thrustSort(_cvgmNewlyAddedKeyPointsLocations.ptr<short2>(),_cvgmNewlyAddedKeyPointsResponse.ptr<float>(),uNewlyAdded);
	uNewlyAdded = std::min( uNewlyAdded, _uTotalParticles - uMatched );
	//g) collect keypoints
	_cvgmParticleResponsesCurr.setTo(0);
	btl::device::semidense::cudaCollectParticles(_cvgmMatchedKeyPointsLocations.ptr<short2>(),_cvgmMatchedKeyPointsResponse.ptr<float>(),uMatched,
		&_cvgmParticleResponsesCurr,NULL);
	btl::device::semidense::cudaCollectParticlesAndOrbDescriptors(
		_cvgmNewlyAddedKeyPointsLocations.ptr<short2>(),_cvgmNewlyAddedKeyPointsResponse.ptr<float>(),_cvgmBlurredCurr,
		uNewlyAdded,_usHalfPatchSize,
		_cvgmPattern.ptr<short>(0),_cvgmPattern.ptr<short>(1),
		&_cvgmParticleResponsesCurr, &_cvgmParticleAngleCurr, &_cvgmParticleOrbDescriptorsCurr);
	//h) assign the current frame to previous frame
	_cvgmParticleResponsesCurr.copyTo(_cvgmParticleResponsesPrev);
	_cvgmParticlesAgeCurr.copyTo(_cvgmParticlesAgePrev);
	_cvgmParticlesVelocityCurr.copyTo(_cvgmParticlesVelocityPrev);
	_cvgmParticleOrbDescriptorsCurr.copyTo(_cvgmParticleOrbDescriptorsPrev);
	_cvgmBlurredCurr.copyTo(_cvgmBlurredPrev);

	//render keypoints
	_cvgmMatchedKeyPointsLocations.download(_cvmKeyPointsLocations);
	_cvgmParticlesAgeCurr.download(_cvmKeyPointsAge);
	//store veloctiy
	_cvgmParticlesVelocityCurr.download(_cvmKeyPointVelocitys[_nFrameIdx]);
	cvmColorFrame_.setTo(cv::Scalar::all(255));
	for (unsigned int i=0;i<uMatched; i++){
		short2 ptCurr = _cvmKeyPointsLocations.ptr<short2>()[i];
		cv::circle(cvmColorFrame_,cv::Point(ptCurr.x,ptCurr.y),1,cv::Scalar(0,0,255.));
		short2 vi = _cvmKeyPointVelocitys[_nFrameIdx].ptr<short2>(ptCurr.y)[ptCurr.x];
		uchar ucAge = _cvmKeyPointsAge.ptr(ptCurr.y)[ptCurr.x];
		int nFrameCurr = _nFrameIdx;
		while (ucAge > 5 ){
			short2 ptPrev = ptCurr - vi;
			cv::line(cvmColorFrame_, cv::Point(ptCurr.x,ptCurr.y), cv::Point(ptPrev.x,ptPrev.y), cv::Scalar(0,0,0));
			ptCurr = ptPrev;
			decrease(30,&nFrameCurr);
			vi = _cvmKeyPointVelocitys[nFrameCurr].ptr<short2>(ptCurr.y)[ptCurr.x];
			--ucAge;
		}
	}
	cv::putText(cvmColorFrame_, "Proposed", cv::Point(10, 15), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(1.,0.,0.) );
	increase(30, &_nFrameIdx);
	return;	
}




