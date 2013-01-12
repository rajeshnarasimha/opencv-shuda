#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "SemiDenseTracker.h"
#include "SemiDenseTrackerOrb.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "Helper.hpp"

#include <opencv2/gpu/device/common.hpp>

__device__ short2 operator + (const short2 s2O1_, const short2 s2O2_);
__device__ short2 operator - (const short2 s2O1_, const short2 s2O2_);
__device__ short2 operator * (const float fO1_, const short2 s2O2_);

float testMatDiff(const cv::gpu::GpuMat& cvgm1_,const cv::gpu::GpuMat& cvgm2_ );
void testCudaCollectParticlesAndOrbDescriptors(const cv::gpu::GpuMat& cvgmFinalKeyPointsLocationsAfterNonMax_, const cv::gpu::GpuMat& cvmFinalKeyPointsResponseAfterNonMax_, const cv::gpu::GpuMat& cvgmImage_,
													const unsigned int uTotalParticles_, const unsigned short _usHalfPatchSize,
													const cv::gpu::GpuMat& cvgmPattern_,
													cv::gpu::GpuMat* pcvgmParticleResponses_, cv::gpu::GpuMat* pcvgmParticleAngle_, cv::gpu::GpuMat* pcvgmParticleDescriptor_);
unsigned int testCudaTrackOrb(const unsigned short usMatchThreshold_, const unsigned short usHalfSize_, const short sSearchRange_,
							const short* psPatternX_, const short* psPatternY_, /*const unsigned int uMaxMatchedKeyPoints_,*/
							const cv::gpu::GpuMat& cvgmParticleOrbDescriptorsPrev_, const cv::gpu::GpuMat& cvgmParticleResponsesPrev_, 
							/*const cv::gpu::GpuMat& cvgmParticlesAgePrev_,const cv::gpu::GpuMat& cvgmParticlesVelocityPrev_, 
							const cv::gpu::GpuMat& cvgmImage_,*/ const cv::gpu::GpuMat& cvgmParticleDescriptorCurrTmp_,
							const cv::gpu::GpuMat& cvgmSaliencyCurr_,
							/*cv::gpu::GpuMat* pcvgmMutex_,*/
							cv::gpu::GpuMat* pcvgmMinMatchDistance_,
							cv::gpu::GpuMat* pcvgmMatchedLocationPrev_
							/*cv::gpu::GpuMat* pcvgmParticleResponsesCurr_,
							cv::gpu::GpuMat* pcvgmParticlesAgeCurr_,cv::gpu::GpuMat* pcvgmParticlesVelocityCurr_,cv::gpu::GpuMat* pcvgmParticleOrbDescriptorsCurr_*/);
void testCudaCollectNewlyAddedKeyPoints(unsigned int uTotalParticles_, unsigned int uMaxNewKeyPoints_, const float fRho_,
										const cv::gpu::GpuMat& cvgmSaliency_,/*const cv::gpu::GpuMat& cvgmParticleResponseCurrTmp_,*/
										const cv::gpu::GpuMat& cvgmParticleDescriptorCurrTmp_,
										const cv::gpu::GpuMat& cvgmParticleVelocityPrev_,
										const cv::gpu::GpuMat& cvgmParticleAgePrev_,
										const cv::gpu::GpuMat& cvgmMinMatchDistance_,
										const cv::gpu::GpuMat& cvgmMatchedLocationPrev_,
										cv::gpu::GpuMat* pcvgmNewlyAddedKeyPointLocation_, cv::gpu::GpuMat* pcvgmNewlyAddedKeyPointResponse_,
										cv::gpu::GpuMat* pcvgmMatchedKeyPointLocation_, cv::gpu::GpuMat* pcvgmMatchedKeyPointResponse_,
										cv::gpu::GpuMat* pcvgmParticleResponseCurr_, cv::gpu::GpuMat* pcvgmParticleDescriptorCurr_,
										cv::gpu::GpuMat* pcvgmParticleVelocityCurr_, cv::gpu::GpuMat* pcvgmParticleAgeCurr_);
bool testCountMinDistAndMatchedLocation(const cv::gpu::GpuMat cvgmMinMatchDistance_, const cv::gpu::GpuMat& cvgmMatchedLocationPrev_, int* pnCounter_);
bool testCountResponseAndDescriptor(const cv::gpu::GpuMat cvgmParticleResponse_, const cv::gpu::GpuMat& cvgmParticleDescriptor_, int* pnCounter_);

namespace btl{ namespace device{ namespace semidense{
	unsigned int cudaCalcSaliency(const cv::gpu::GpuMat& cvgmImage_, const unsigned short usHalfSizeRound_,
									const unsigned char ucContrastThreshold_, const float& fSaliencyThreshold_, 
									cv::gpu::GpuMat* pcvgmSaliency_, cv::gpu::GpuMat* pcvgmKeyPointLocations_);
	unsigned int cudaNonMaxSupression(const cv::gpu::GpuMat& cvgmKeyPointLocation_, const unsigned int uMaxSalientPoints_, 
										const cv::gpu::GpuMat& cvgmSaliency_, short2* ps2devLocations_, float* pfdevResponse_);
	void thrustSort(short2* pnLoc_, float* pfResponse_, const unsigned int nCorners_);
	void cudaCalcAngles(const cv::gpu::GpuMat& cvgmImage_, const short2* pdevFinalKeyPointsLocations_, const unsigned int uPoints_,  const unsigned short usHalf_, 
						cv::gpu::GpuMat* pcvgmParticleAngle_);
	void loadUMax(const int* pUMax_, int nCount_);
	void cudaExtractAllDescriptorOrb(	const cv::gpu::GpuMat& cvgmImage_,
										const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, 
										const unsigned int uTotalParticles_, const unsigned short usHalfPatchSize_,
										const short* psPatternX_, const short* psPatternY_,
										cv::gpu::GpuMat* pcvgmParticleResponses_, cv::gpu::GpuMat* pcvgmParticleAngle_, cv::gpu::GpuMat* pcvgmParticleDescriptor_);
	//after tracking, the matched particles are filled into the pcvgmParticleResponsesCurr_, pcvgmParticlesAgeCurr_, pcvgmParticlesVelocityCurr_, 
	//and pcvgmParticleOrbDescriptorsCurr_, moreover, the cvgmSaliencyCurr_
	unsigned int cudaTrackOrb(const unsigned short usMatchThreshold_, const unsigned short usHalfSize_, const short sSearchRange_,
							const cv::gpu::GpuMat& cvgmParticleOrbDescriptorsPrev_, const cv::gpu::GpuMat& cvgmParticleResponsesPrev_, 
							const cv::gpu::GpuMat& cvgmParticleDescriptorCurrTmp_,
							const cv::gpu::GpuMat& cvgmSaliencyCurr_,
							cv::gpu::GpuMat* pcvgmMinMatchDistance_,
							cv::gpu::GpuMat* pcvgmMatchedLocationPrev_);
	//separate salient point into matched and newly added.
	//for matched keypoints the velocity and age will be updated
	void cudaCollectKeyPointOrb(unsigned int uTotalParticles_, unsigned int uMaxNewKeyPoints_, const float fRho_,
										const cv::gpu::GpuMat& cvgmSaliency_,/*const cv::gpu::GpuMat& cvgmParticleResponseCurrTmp_,*/
										const cv::gpu::GpuMat& cvgmParticleDescriptorCurrTmp_,
										const cv::gpu::GpuMat& cvgmParticleVelocityPrev_,
										const cv::gpu::GpuMat& cvgmParticleAgePrev_,
										const cv::gpu::GpuMat& cvgmMinMatchDistance_,
										const cv::gpu::GpuMat& cvgmMatchedLocationPrev_,
										cv::gpu::GpuMat* pcvgmNewlyAddedKeyPointLocation_, cv::gpu::GpuMat* pcvgmNewlyAddedKeyPointResponse_,
										cv::gpu::GpuMat* pcvgmMatchedKeyPointLocation_, cv::gpu::GpuMat* pcvgmMatchedKeyPointResponse_,
										cv::gpu::GpuMat* pcvgmParticleResponseCurr_, cv::gpu::GpuMat* pcvgmParticleDescriptorCurr_,
										cv::gpu::GpuMat* pcvgmParticleVelocityCurr_, cv::gpu::GpuMat* pcvgmParticleAgeCurr_);
	void cudaCollectParticles(const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, const unsigned int uTotalParticles_, 
								cv::gpu::GpuMat* pcvgmParticleResponses_, cv::gpu::GpuMat* pcvgmParticleDescriptor_, const cv::gpu::GpuMat& cvgmImage_=cv::gpu::GpuMat() );
	unsigned int cudaMatchedAndNewlyAddedKeyPointsCollection(cv::gpu::GpuMat& cvgmKeyPointLocation_, 
																unsigned int* puMaxSalientPoints_, cv::gpu::GpuMat* pcvgmParticleResponsesCurr_, 
																short2* ps2devMatchedKeyPointLocations_, float* pfdevMatchedKeyPointResponse_, 
																short2* ps2devNewlyAddedKeyPointLocations_, float* pfdevNewlyAddedKeyPointResponse_);
}//semidense
}//device
}//btl

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

	_usHalfPatchSize = 9; //the size of the orb feature
	_sSearchRange = 5;

	_nFrameIdx = 0;
}


void btl::image::semidense::CSemiDenseTrackerOrb::initUMax(){
	// pre-compute the end of a row in a circular patch (1/4 of the circular patch)
	int half_patch_size = _usHalfPatchSize;
	std::vector<int> u_max(half_patch_size + 2);
	for (int v = 0; v <= half_patch_size * std::sqrt(2.f) / 2 + 1; ++v)
		u_max[v] = cvRound(std::sqrt(static_cast<float>(half_patch_size * half_patch_size - v * v)));

	// Make sure we are symmetric
	for (int v = half_patch_size, v_0 = 0; v >= half_patch_size * std::sqrt(2.f) / 2; --v){
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

bool btl::image::semidense::CSemiDenseTrackerOrb::initialize( cv::Mat& cvmColorFrame_ )
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

	_cvgmMatchedKeyPointLocation.create(1, _uTotalParticles, CV_16SC2);
	_cvgmMatchedKeyPointResponse.create(1, _uTotalParticles, CV_32FC1);
	_cvgmNewlyAddedKeyPointLocation.create(1, _uMaxKeyPointsAfterNonMax, CV_16SC2);
	_cvgmNewlyAddedKeyPointResponse.create(1, _uMaxKeyPointsAfterNonMax, CV_32FC1);

	//init particles
	_cvgmParticleResponsePrev.create(cvmColorFrame_.size(),CV_32FC1);	   _cvgmParticleResponsePrev.setTo(0);
	_cvgmParticleResponseCurr.create(cvmColorFrame_.size(),CV_32FC1);	   _cvgmParticleResponseCurr.setTo(0);
	_cvgmParticleAnglePrev.create(cvmColorFrame_.size(),CV_32FC1);		   _cvgmParticleAnglePrev.setTo(0);
	_cvgmParticleAngleCurr.create(cvmColorFrame_.size(),CV_32FC1);		   _cvgmParticleAngleCurr.setTo(0);
	_cvgmParticleVelocityPrev.create(cvmColorFrame_.size(),CV_16SC2);	   _cvgmParticleVelocityPrev.setTo(cv::Scalar::all(0));//float velocity; 
	_cvgmParticleVelocityCurr.create(cvmColorFrame_.size(),CV_16SC2);	   _cvgmParticleVelocityCurr.setTo(cv::Scalar::all(0));//float velocity; 
	_cvgmParticleAgePrev.create(cvmColorFrame_.size(),CV_8UC1);	       _cvgmParticleAgePrev.setTo(0);//uchar age;
	_cvgmParticleAgeCurr.create(cvmColorFrame_.size(),CV_8UC1);		   _cvgmParticleAgeCurr.setTo(0);//uchar age;
	_cvgmParticleDescriptorPrev.create(cvmColorFrame_.size(),CV_32SC2);_cvgmParticleDescriptorPrev.setTo(cv::Scalar::all(0));
	_cvgmParticleDescriptorCurr.create(cvmColorFrame_.size(),CV_32SC2);_cvgmParticleDescriptorCurr.setTo(cv::Scalar::all(0));
	_cvgmParticleDescriptorCurrTmp.create(cvmColorFrame_.size(),CV_32SC2);_cvgmParticleDescriptorCurrTmp.setTo(cv::Scalar::all(0));
	_cvgmMinMatchDistance.create(cvmColorFrame_.size(),CV_8UC1);
	_cvgmMatchedLocationPrev.create(cvmColorFrame_.size(),CV_16SC2);

	//allocate filter
	if (_pBlurFilter.empty()){
		_pBlurFilter = cv::gpu::createGaussianFilter_GPU(CV_8UC1, cv::Size(_uGaussianKernelSize, _uGaussianKernelSize), _fSigma, _fSigma, cv::BORDER_REFLECT_101);
	}
	//processing the frame
	//apply gaussian filter
	_pBlurFilter->apply(_cvgmGrayFrame, _cvgmBlurredPrev, cv::Rect(0, 0, _cvgmGrayFrame.cols, _cvgmGrayFrame.rows));
	//detect key points
	//1.compute the saliency score 
	unsigned int uTotalSalientPoints = btl::device::semidense::cudaCalcSaliency(_cvgmBlurredPrev, unsigned short(_usHalfPatchSize*1.5) ,_ucContrastThresold, _fSaliencyThreshold, 
																				&_cvgmSaliency, &_cvgmInitKeyPointLocation); 
	if (uTotalSalientPoints< _uTotalParticles/2) return false;
	uTotalSalientPoints = std::min( uTotalSalientPoints, _uMaxKeyPointsBeforeNonMax );
		
	//2.do a non-max suppression and initialize particles ( extract feature descriptors ) 
	unsigned int uFinalSalientPointsAfterNonMax = btl::device::semidense::cudaNonMaxSupression(_cvgmInitKeyPointLocation, uTotalSalientPoints, _cvgmSaliency, 
																							   _cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(), _cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>() );
	uFinalSalientPointsAfterNonMax = std::min( uFinalSalientPointsAfterNonMax, _uMaxKeyPointsAfterNonMax );
	
	//3.sort all salient points according to their strength and pick the first _uTotalParticles;
	btl::device::semidense::thrustSort(_cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(),_cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>(),uFinalSalientPointsAfterNonMax);
	_uTotalParticles = std::min( _uTotalParticles, uFinalSalientPointsAfterNonMax );

	//4.collect all salient points and descriptors on them
	_cvgmParticleResponsePrev.setTo(0.f);
	btl::device::semidense::cudaExtractAllDescriptorOrb(_cvgmBlurredPrev,
														_cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(),_cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>(),
														_uTotalParticles,_usHalfPatchSize,
														_cvgmPattern.ptr<short>(0),_cvgmPattern.ptr<short>(1),
														&_cvgmParticleResponsePrev, &_cvgmParticleAnglePrev, &_cvgmParticleDescriptorPrev);
	/*
	//test
	cv::gpu::GpuMat cvgmTestResponse(_cvgmParticleResponsePrev); cvgmTestResponse.setTo(0.f);
	cv::gpu::GpuMat cvgmTestOrbDescriptor(_cvgmParticleDescriptorPrev);cvgmTestOrbDescriptor.setTo(cv::Scalar::all(0));
	testCudaCollectParticlesAndOrbDescriptors(_cvgmFinalKeyPointsLocationsAfterNonMax,_cvgmFinalKeyPointsResponseAfterNonMax,_cvgmBlurredPrev,
											 _uTotalParticles,_usHalfPatchSize,_cvgmPattern,
									  		 &cvgmTestResponse,&_cvgmParticleAnglePrev,&cvgmTestOrbDescriptor);
	float fD1 = testMatDiff(_cvgmParticleResponsePrev, cvgmTestResponse);
	float fD2 = testMatDiff(_cvgmParticleDescriptorPrev, cvgmTestOrbDescriptor);*/
	
	//store velocity
	_cvgmParticleVelocityPrev.download(_cvmKeyPointVelocity[_nFrameIdx]);
	btl::other::increase<int>(30,&_nFrameIdx);

	//render keypoints
	cvmColorFrame_.setTo(cv::Scalar::all(255));
	//cv::putText(cvmColorFrame_, "Proposed Method", cv::Point(10, 15), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(1.,0.,0.) );
	return true;
}

void btl::image::semidense::CSemiDenseTrackerOrb::track( cv::Mat& cvmColorFrame_ )
{
	_cvgmColorFrame.upload(cvmColorFrame_);
	cv::gpu::cvtColor(_cvgmColorFrame,_cvgmGrayFrame,cv::COLOR_RGB2GRAY);
	//processing the frame
	//Gaussian smoothes the input image 
	_pBlurFilter->apply(_cvgmGrayFrame, _cvgmBlurredCurr, cv::Rect(0, 0, _cvgmGrayFrame.cols, _cvgmGrayFrame.rows));
	//calc the saliency score for each pixel
	unsigned int uTotalSalientPoints = btl::device::semidense::cudaCalcSaliency(_cvgmBlurredCurr, unsigned short(_usHalfPatchSize*1.5), _ucContrastThresold, _fSaliencyThreshold, &_cvgmSaliency, &_cvgmInitKeyPointLocation);
	uTotalSalientPoints = std::min( uTotalSalientPoints, _uMaxKeyPointsBeforeNonMax );
	
	//do a non-max suppression and collect the candidate particles into a temporary vectors( extract feature descriptors ) 
	unsigned int uFinalSalientPoints = btl::device::semidense::cudaNonMaxSupression(_cvgmInitKeyPointLocation, uTotalSalientPoints, _cvgmSaliency, 
																					_cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(), _cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>() );
	_uFinalSalientPoints = uFinalSalientPoints = std::min( uFinalSalientPoints, unsigned int(_uMaxKeyPointsAfterNonMax) );
	_cvgmSaliency.setTo(0.f);//clear saliency scores
	//redeploy the saliency matrix
	btl::device::semidense::cudaExtractAllDescriptorOrb(_cvgmBlurredCurr,
														_cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(),_cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>(),
														uFinalSalientPoints,_usHalfPatchSize,
														_cvgmPattern.ptr<short>(0),_cvgmPattern.ptr<short>(1),
														&_cvgmSaliency, &_cvgmParticleAngleCurr, &_cvgmParticleDescriptorCurrTmp);

/*
	int nCounter = 0;
	bool bIsLegal = testCountResponseAndDescriptor(_cvgmSaliency,_cvgmParticleDescriptorCurrTmp,&nCounter);
*/
	//track particles in previous frame by searching the candidates of current frame. 
	//Note that _cvgmSaliency is the input as well as output, tracked particles are marked as negative scores
	_cvgmMatchedLocationPrev.setTo(cv::Scalar::all(0));
	unsigned int uMatchedPoints = btl::device::semidense::cudaTrackOrb( 20, _usHalfPatchSize, _sSearchRange,
																		_cvgmParticleDescriptorPrev,  _cvgmParticleResponsePrev, 
																		_cvgmParticleDescriptorCurrTmp, _cvgmSaliency,
																		&_cvgmMinMatchDistance,
																		&_cvgmMatchedLocationPrev);
	/*nCounter = 0;
	bIsLegal = testCountResponseAndDescriptor(_cvgmSaliency,_cvgmParticleDescriptorCurrTmp,&nCounter);

	nCounter = 0;
	bIsLegal = testCountMinDistAndMatchedLocation( _cvgmMinMatchDistance, _cvgmMatchedLocationPrev, &nCounter );*/

/*
	cv::Mat cvmPattern; _cvgmPattern.download(cvmPattern);
	cv::gpu::GpuMat cvgmMinMatchDistanceTest,cvgmMatchedLocationPrevTest;
	cvgmMinMatchDistanceTest       .create(_cvgmBlurredCurr.size(),CV_8UC1);
	cvgmMatchedLocationPrevTest    .create(_cvgmBlurredCurr.size(),CV_16SC2);	cvgmMatchedLocationPrevTest.setTo(cv::Scalar::all(0));
	unsigned int uMatchedPointsTest = testCudaTrackOrb( 12, _usHalfPatchSize, _sSearchRange,
														_cvgmPattern.ptr<short>(0), _cvgmPattern.ptr<short>(1), 
														_cvgmParticleDescriptorPrev, _cvgmParticleResponsePrev, 
														_cvgmParticleDescriptorCurrTmp,
														_cvgmSaliency, &cvgmMinMatchDistanceTest, &cvgmMatchedLocationPrevTest);
	float fD0 = testMatDiff(_cvgmMatchedLocationPrev, cvgmMatchedLocationPrevTest);
	float fD1 = testMatDiff(_cvgmMinMatchDistance,cvgmMinMatchDistanceTest);
	float fD2 = (float)uMatchedPointsTest - nCounter;
*/
	//separate tracked particles and rest of candidates. Note that saliency scores are updated 
	//Note that _cvgmSaliency is the input as well as output, after the tracked particles are separated with rest of candidates, their negative saliency
	//scores are recovered into positive scores
	_cvgmMatchedKeyPointLocation   .setTo(cv::Scalar::all(0));//clear all memory
	_cvgmMatchedKeyPointResponse   .setTo(0.f);
	_cvgmNewlyAddedKeyPointLocation.setTo(cv::Scalar::all(0));//clear all memory
	_cvgmNewlyAddedKeyPointResponse.setTo(0.f);
	btl::device::semidense::cudaCollectKeyPointOrb(_uTotalParticles, _uMaxKeyPointsAfterNonMax, 0.75f,
													_cvgmSaliency, _cvgmParticleDescriptorCurrTmp,
													_cvgmParticleVelocityPrev,_cvgmParticleAgePrev,
													_cvgmMinMatchDistance,_cvgmMatchedLocationPrev,
													&_cvgmNewlyAddedKeyPointLocation, &_cvgmNewlyAddedKeyPointResponse, 
													&_cvgmMatchedKeyPointLocation, &_cvgmMatchedKeyPointResponse,
													&_cvgmParticleResponseCurr, &_cvgmParticleDescriptorCurr,
													&_cvgmParticleVelocityCurr,&_cvgmParticleAgeCurr);
/*
	nCounter = 0;
	bIsLegal = testCountResponseAndDescriptor(_cvgmSaliency,_cvgmParticleDescriptorCurrTmp,&nCounter);

	nCounter = 0;
	bIsLegal = testCountResponseAndDescriptor(_cvgmParticleResponseCurr,_cvgmParticleDescriptorCurr,&nCounter);
	cv::gpu::GpuMat cvgmNewlyAddedKeyPointLocationTest(_cvgmNewlyAddedKeyPointLocation), cvgmNewlyAddedKeyPointResponseTest(_cvgmNewlyAddedKeyPointResponse), 
		            cvgmMatchedKeyPointLocationTest(_cvgmMatchedKeyPointLocation), cvgmMatchedKeyPointResponseTest(_cvgmMatchedKeyPointResponse),
					cvgmParticleResponseCurrTest(_cvgmParticleResponseCurr), cvgmParticleDescriptorCurrTest(_cvgmParticleDescriptorCurr),
					cvgmParticleVelocityCurrTest(_cvgmParticleVelocityCurr), cvgmParticleAgeCurrTest(_cvgmParticleAgeCurr);
	
	nCounter = 0;
	bIsLegal = testCountResponseAndDescriptor(_cvgmSaliency, _cvgmParticleDescriptorCurrTmp,&nCounter);

	testCudaCollectNewlyAddedKeyPoints(_uTotalParticles, _uMaxKeyPointsAfterNonMax, 0.75f,
									  _cvgmSaliency, _cvgmParticleDescriptorCurrTmp,
									  _cvgmParticleVelocityPrev,_cvgmParticleAgePrev,
									  _cvgmMinMatchDistance,_cvgmMatchedLocationPrev,
									  &cvgmNewlyAddedKeyPointLocationTest, &cvgmNewlyAddedKeyPointResponseTest, 
									  &cvgmMatchedKeyPointLocationTest, &cvgmMatchedKeyPointResponseTest,
									  &cvgmParticleResponseCurrTest, &cvgmParticleDescriptorCurrTest,
									  &cvgmParticleVelocityCurrTest,&cvgmParticleAgeCurrTest);
	nCounter = 0;
	bIsLegal = testCountResponseAndDescriptor(_cvgmParticleResponseCurr,_cvgmParticleDescriptorCurr,&nCounter);
	float fD3 = testMatDiff(cvgmNewlyAddedKeyPointLocationTest, _cvgmNewlyAddedKeyPointLocation);
	float fD4 = testMatDiff(cvgmNewlyAddedKeyPointResponseTest, _cvgmNewlyAddedKeyPointResponse);
	float fD5 = testMatDiff(cvgmMatchedKeyPointLocationTest, _cvgmMatchedKeyPointLocation);
	float fD6 = testMatDiff(cvgmMatchedKeyPointResponseTest, _cvgmMatchedKeyPointResponse);
	float fD7 = testMatDiff(cvgmParticleResponseCurrTest, _cvgmParticleResponseCurr);
	float fD8 = testMatDiff(cvgmParticleDescriptorCurrTest, _cvgmParticleDescriptorCurr);
	float fD9 = testMatDiff(cvgmParticleVelocityCurrTest, _cvgmParticleVelocityCurr);
	float fD10= testMatDiff(cvgmParticleAgeCurrTest, _cvgmParticleAgeCurr);

*/

	//h) assign the current frame to previous frame
	_cvgmBlurredCurr		 .copyTo(_cvgmBlurredPrev);
	_cvgmParticleResponseCurr.copyTo(_cvgmParticleResponsePrev);
	_cvgmParticleAgeCurr	 .copyTo(_cvgmParticleAgePrev);
	_cvgmParticleVelocityCurr.copyTo(_cvgmParticleVelocityPrev);
	_cvgmParticleDescriptorCurr.copyTo(_cvgmParticleDescriptorPrev);

	//store velocity
	_cvgmParticleVelocityCurr.download(_cvmKeyPointVelocity[_nFrameIdx]);
	//render keypoints
	_cvgmMatchedKeyPointLocation.download(_cvmKeyPointLocation);
	_cvgmParticleAgeCurr.download(_cvmKeyPointAge);
	cvmColorFrame_.setTo(cv::Scalar::all(255));
	float fTotalAge = 0.f;
	for (unsigned int i=0;i<uMatchedPoints; i+=2){
		short2 ptCurr = _cvmKeyPointLocation.ptr<short2>()[i];
		uchar ucAge = _cvmKeyPointAge.ptr(ptCurr.y)[ptCurr.x];
		//if(ucAge < 2 ) continue;
		cv::circle(cvmColorFrame_,cv::Point(ptCurr.x,ptCurr.y),1,cv::Scalar(0,0,255.));
		short2 vi = _cvmKeyPointVelocity[_nFrameIdx].ptr<short2>(ptCurr.y)[ptCurr.x];
		int nFrameCurr = _nFrameIdx;
		fTotalAge  += ucAge;
		int nFrame = 0;
		while (ucAge > 0 && nFrame < 5){//render trajectory 
			short2 ptPrev = ptCurr - vi;
			cv::line(cvmColorFrame_, cv::Point(ptCurr.x,ptCurr.y), cv::Point(ptPrev.x,ptPrev.y), cv::Scalar(0,0,0));
			ptCurr = ptPrev;
			btl::other::decrease<int>(30,&nFrameCurr);
			vi = _cvmKeyPointVelocity[nFrameCurr].ptr<short2>(ptCurr.y)[ptCurr.x];
			--ucAge; ++nFrame;
		}
	}
	float fAvgAge = fTotalAge/uMatchedPoints;
	btl::other::increase<int>(30,&_nFrameIdx);
	return;	
}




