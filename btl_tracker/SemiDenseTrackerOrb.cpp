#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>

#include "SemiDenseTracker.h"
#include "SemiDenseTrackerOrb.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "Helper.hpp"
//#include "OtherUtil.hpp"

#include <opencv2/gpu/device/common.hpp>
#include "TestCudaOrb.h"
#include "SemiDenseTrackerOrb.cuh"

__device__ short2 operator + (const short2 s2O1_, const short2 s2O2_);
__device__ short2 operator - (const short2 s2O1_, const short2 s2O2_);
__device__ short2 operator * (const float fO1_, const short2 s2O2_);


btl::image::semidense::CSemiDenseTrackerOrb::CSemiDenseTrackerOrb(unsigned int uPyrHeight_)
	:CSemiDenseTracker(uPyrHeight_)
{
	//Gaussian filter
	_fSigma = 1.f; // page3: r=3/6 and sigma = 1.f/2.f respectively
	_uRadius = 3; // 
	_uGaussianKernelSize = 2*_uRadius + 1;
	//contrast threshold
	_ucContrastThresold = 5; // 255 * 0.02 = 5.1

	//saliency threshold
	_fSaliencyThreshold = 0.2f;
	//match threshold
	_usMatchThreshod[0] = 19;
	_usMatchThreshod[1] = 18;
	_usMatchThreshod[2] = 18;
	_usMatchThreshod[3] = 18; 

	//# of Max key points
	_uMaxKeyPointsBeforeNonMax[0] = 80000;
	_uMaxKeyPointsBeforeNonMax[1] = 10000;
	_uMaxKeyPointsBeforeNonMax[2] =  2500;
	_uMaxKeyPointsBeforeNonMax[3] =   650;

	_uMaxKeyPointsAfterNonMax[0] = 20000;
	_uMaxKeyPointsAfterNonMax[1] =  2500;
	_uMaxKeyPointsAfterNonMax[2] =   600;
	_uMaxKeyPointsAfterNonMax[3] =   150;

	_uTotalParticles[0] = 8000;
	_uTotalParticles[1] = 2000;
	_uTotalParticles[2] =  500;
	_uTotalParticles[3] =  100;

	_usHalfPatchSize = 14; //the size of the orb feature
	_sSearchRange = 7;
	_sDescriptorByte =24;

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
	const int nPoints = 8*_sDescriptorByte*2; //128; // 64 tests and each test requires 2 points 256x2 = 512
	cv::Mat cvmPattern; //2 x n : 1st row is x and 2nd row is y; test point1, test point2;
	//assign cvmPattern_ from precomputed patterns
	cvmPattern.create(2, nPoints, CV_16SC1);
	makeRandomPattern(_usHalfPatchSize, nPoints, &cvmPattern );
	btl::device::semidense::loadOrbPattern(cvmPattern.ptr<short>(0),cvmPattern.ptr<short>(1),_sDescriptorByte,_uPyrHeight);
	return;
}

bool btl::image::semidense::CSemiDenseTrackerOrb::initialize( boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrBW[4] )
{
	_nFrameIdx = 0;
	initUMax();
	initOrbPattern();
	for (int n = _uPyrHeight-1; n>-1; --n ){
		_cvgmSaliency[n].create(_acvgmShrPtrPyrBW[n]->size(),CV_32FC1);
		_cvgmInitKeyPointLocation[n].create(1, _uMaxKeyPointsBeforeNonMax[n], CV_16SC2);
		_cvgmFinalKeyPointsLocationsAfterNonMax[n].create(1, _uMaxKeyPointsAfterNonMax[n], CV_16SC2);//short2 location;
		_cvgmFinalKeyPointsResponseAfterNonMax[n].create(1, _uMaxKeyPointsAfterNonMax[n], CV_32FC1);//float corner strength(response);  

		_cvgmMatchedKeyPointLocation[n].create(1, _uTotalParticles[n], CV_16SC2);
		_cvgmMatchedKeyPointResponse[n].create(1, _uTotalParticles[n], CV_32FC1);
		_cvgmNewlyAddedKeyPointLocation[n].create(1, _uMaxKeyPointsAfterNonMax[n], CV_16SC2);
		_cvgmNewlyAddedKeyPointResponse[n].create(1, _uMaxKeyPointsAfterNonMax[n], CV_32FC1);

		//init particles
		_cvgmParticleResponsePrev[n].create(_acvgmShrPtrPyrBW[n]->size(),CV_32FC1);	   _cvgmParticleResponsePrev[n].setTo(0);
		_cvgmParticleVelocityPrev[n].create(_acvgmShrPtrPyrBW[n]->size(),CV_16SC2);	   _cvgmParticleVelocityPrev[n].setTo(cv::Scalar::all(0));//float velocity; 
		_cvgmParticleAgePrev[n].create(_acvgmShrPtrPyrBW[n]->size(),CV_8UC1);			   _cvgmParticleAgePrev[n].setTo(0);//uchar age;
		_cvgmParticleDescriptorPrev[n].create(_acvgmShrPtrPyrBW[n]->rows,_acvgmShrPtrPyrBW[n]->cols*_sDescriptorByte, CV_8UC1);    _cvgmParticleDescriptorPrev[n].setTo(cv::Scalar::all(0));
		//in current frame
		_cvgmParticleResponseCurr[n].create(_acvgmShrPtrPyrBW[n]->size(),CV_32FC1);	   _cvgmParticleResponseCurr[n].setTo(0);
		_cvgmParticleAngleCurr[n].create(_acvgmShrPtrPyrBW[n]->size(),CV_32FC1);		   _cvgmParticleAngleCurr[n].setTo(0);
		_cvgmParticleVelocityCurr[n].create(_acvgmShrPtrPyrBW[n]->size(),CV_16SC2);	   _cvgmParticleVelocityCurr[n].setTo(cv::Scalar::all(0));//float velocity; 
		_cvgmParticleAgeCurr[n].create(_acvgmShrPtrPyrBW[n]->size(),CV_8UC1);		       _cvgmParticleAgeCurr[n].setTo(0);//uchar age;
		_cvgmParticleDescriptorCurr[n].create(_acvgmShrPtrPyrBW[n]->rows,_acvgmShrPtrPyrBW[n]->cols*_sDescriptorByte, CV_8UC1);	   _cvgmParticleDescriptorCurr[n].setTo(cv::Scalar::all(0));
		_cvgmParticleDescriptorCurrTmp[n].create(_acvgmShrPtrPyrBW[n]->rows,_acvgmShrPtrPyrBW[n]->cols*_sDescriptorByte, CV_8UC1); _cvgmParticleDescriptorCurrTmp[n].setTo(cv::Scalar::all(0));
		//other
		_cvgmMinMatchDistance[n].create(_acvgmShrPtrPyrBW[n]->size(),CV_8UC1);
		_cvgmMatchedLocationPrev[n].create(_acvgmShrPtrPyrBW[n]->size(),CV_16SC2);
		_cvgmVelocityPrev2Curr[n].create(_acvgmShrPtrPyrBW[n]->size(),CV_16SC2);

		//allocate filter
		if (_pBlurFilter.empty()){
			_pBlurFilter = cv::gpu::createGaussianFilter_GPU(CV_8UC1, cv::Size(_uGaussianKernelSize, _uGaussianKernelSize), _fSigma, _fSigma, cv::BORDER_REFLECT_101);
		}
		//processing the frame
		//apply gaussian filter
		_pBlurFilter->apply(*_acvgmShrPtrPyrBW[n], _cvgmBlurredPrev[n], cv::Rect(0, 0, _acvgmShrPtrPyrBW[n]->cols, _acvgmShrPtrPyrBW[n]->rows));
		//detect key points
		//1.compute the saliency score 
		unsigned int uTotalSalientPoints = btl::device::semidense::cudaCalcSaliency(_cvgmBlurredPrev[n], unsigned short(_usHalfPatchSize*1.5) ,_ucContrastThresold, _fSaliencyThreshold, 
																					&_cvgmSaliency[n], &_cvgmInitKeyPointLocation[n]); 
		if (uTotalSalientPoints< 10 ) return false;
		uTotalSalientPoints = std::min( uTotalSalientPoints, _uMaxKeyPointsBeforeNonMax[n] );
		
		//2.do a non-max suppression and initialize particles ( extract feature descriptors ) 
		unsigned int uFinalSalientPointsAfterNonMax = btl::device::semidense::cudaNonMaxSupression(_cvgmInitKeyPointLocation[n], uTotalSalientPoints, _cvgmSaliency[n], 
																								   _cvgmFinalKeyPointsLocationsAfterNonMax[n].ptr<short2>(), _cvgmFinalKeyPointsResponseAfterNonMax[n].ptr<float>() );
		uFinalSalientPointsAfterNonMax = std::min( uFinalSalientPointsAfterNonMax, _uMaxKeyPointsAfterNonMax[n] );
	
		//3.sort all salient points according to their strength and pick the first _uTotalParticles;
		btl::device::semidense::thrustSort(_cvgmFinalKeyPointsLocationsAfterNonMax[n].ptr<short2>(),_cvgmFinalKeyPointsResponseAfterNonMax[n].ptr<float>(),uFinalSalientPointsAfterNonMax);
		_uTotalParticles[n] = std::min( _uTotalParticles[n], uFinalSalientPointsAfterNonMax );

		//4.collect all salient points and descriptors on them
		_cvgmParticleResponsePrev[n].setTo(0.f);
		btl::device::semidense::cudaExtractAllDescriptorOrb(_cvgmBlurredPrev[n],
															_cvgmFinalKeyPointsLocationsAfterNonMax[n].ptr<short2>(),_cvgmFinalKeyPointsResponseAfterNonMax[n].ptr<float>(),
															_uTotalParticles[n],_usHalfPatchSize,
															_sDescriptorByte,
															&_cvgmParticleResponsePrev[n], &_cvgmParticleAngleCurr[n], &_cvgmParticleDescriptorPrev[n]);
		//for testing
		/*int nCounter = 0;
		bool bIsLegal = testCountResponseAndDescriptor(_cvgmParticleResponsePrev[n],_cvgmParticleDescriptorPrev[n],&nCounter,_sDescriptorByte);
		//test
		cv::gpu::GpuMat cvgmTestResponse(_cvgmParticleResponsePrev[n]); cvgmTestResponse.setTo(0.f);
		cv::gpu::GpuMat cvgmTestOrbDescriptor(_cvgmParticleDescriptorPrev[n]);cvgmTestOrbDescriptor.setTo(cv::Scalar::all(0));
		testCudaCollectParticlesAndOrbDescriptors(_cvgmFinalKeyPointsLocationsAfterNonMax[n],_cvgmFinalKeyPointsResponseAfterNonMax[n],_cvgmBlurredPrev[n],
												 _uTotalParticles[n],_usHalfPatchSize,_cvgmPattern, _sDescriptorByte,
									  			 &cvgmTestResponse,&_cvgmParticleAngleCurr[n],&cvgmTestOrbDescriptor);
		float fD1 = testMatDiff(_cvgmParticleResponsePrev[n], cvgmTestResponse);
		float fD2 = testMatDiff(_cvgmParticleDescriptorPrev[n], cvgmTestOrbDescriptor);*/

		//store velocity
		_cvgmParticleVelocityCurr[n].download(_cvmKeyPointVelocity[_nFrameIdx][n]);
	}
	
	return true;
}

void btl::image::semidense::CSemiDenseTrackerOrb::track( boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrBW[4] )
{
	for (short n = _uPyrHeight-1; n>-1; --n ) {
		//processing the frame
		//Gaussian smoothes the input image 
		_pBlurFilter->apply(*_acvgmShrPtrPyrBW[n] , _cvgmBlurredCurr[n], cv::Rect(0, 0, _acvgmShrPtrPyrBW[n]->cols, _acvgmShrPtrPyrBW[n]->rows));
		//calc the saliency score for each pixel
		unsigned int uTotalSalientPoints = btl::device::semidense::cudaCalcSaliency( _cvgmBlurredCurr[n], 
																					 unsigned short(_usHalfPatchSize*1.5), _ucContrastThresold, _fSaliencyThreshold, 
																					 &_cvgmSaliency[n], &_cvgmInitKeyPointLocation[n]);
		uTotalSalientPoints = std::min( uTotalSalientPoints, _uMaxKeyPointsBeforeNonMax[n] );
	
		//do a non-max suppression and collect the candidate particles into a temporary vectors( extract feature descriptors ) 
		unsigned int uFinalSalientPoints = btl::device::semidense::cudaNonMaxSupression(_cvgmInitKeyPointLocation[n], uTotalSalientPoints, _cvgmSaliency[n], 
																						_cvgmFinalKeyPointsLocationsAfterNonMax[n].ptr<short2>(), _cvgmFinalKeyPointsResponseAfterNonMax[n].ptr<float>() );
		_uFinalSalientPoints[n] = uFinalSalientPoints = std::min( uFinalSalientPoints, unsigned int(_uMaxKeyPointsAfterNonMax[n]) );
		_cvgmSaliency[n].setTo(0.f);//clear saliency scores
		//redeploy the saliency matrix
		btl::device::semidense::cudaExtractAllDescriptorOrb(_cvgmBlurredCurr[n],
															_cvgmFinalKeyPointsLocationsAfterNonMax[n].ptr<short2>(),_cvgmFinalKeyPointsResponseAfterNonMax[n].ptr<float>(),
															uFinalSalientPoints,_usHalfPatchSize,
															_sDescriptorByte,
															&_cvgmSaliency[n], &_cvgmParticleAngleCurr[n], &_cvgmParticleDescriptorCurrTmp[n]);
		/*int nCounter = 0;
		bool bIsLegal = testCountResponseAndDescriptor(_cvgmSaliency[n],_cvgmParticleDescriptorCurrTmp[n],&nCounter,_sDescriptorByte);*/

		//track particles in previous frame by searching the candidates of current frame. 
		//Note that _cvgmSaliency is the input as well as output, tracked particles are marked as negative scores
		_cvgmMatchedLocationPrev[n].setTo(cv::Scalar::all(0));
		_uMatchedPoints[n] = btl::device::semidense::cudaTrackOrb(  n, _usMatchThreshod, _usHalfPatchSize, _sSearchRange, _sDescriptorByte, _uPyrHeight,
																	_cvgmParticleDescriptorPrev,  _cvgmParticleResponsePrev, 
																	_cvgmParticleDescriptorCurrTmp, _cvgmSaliency,
																	_cvgmMinMatchDistance, _cvgmMatchedLocationPrev, _cvgmVelocityPrev2Curr );
		
		/*int nCounter = 0;
		bool bIsLegal = testCountResponseAndDescriptor(_cvgmSaliency[n],_cvgmParticleDescriptorCurrTmp[n],&nCounter,_sDescriptorByte);
		nCounter = 0;
		bIsLegal = testCountMinDistAndMatchedLocation( _cvgmMinMatchDistance[n], _cvgmMatchedLocationPrev[n], &nCounter );

		cv::Mat cvmPattern; _cvgmPattern.download(cvmPattern);
		cv::gpu::GpuMat cvgmMinMatchDistanceTest[4],cvgmMatchedLocationPrevTest[4];
		cvgmMinMatchDistanceTest[n]       .create(_cvgmBlurredCurr[n].size(),CV_8UC1);
		cvgmMatchedLocationPrevTest[n]    .create(_cvgmBlurredCurr[n].size(),CV_16SC2);	cvgmMatchedLocationPrevTest[n].setTo(cv::Scalar::all(0));

		unsigned int uMatchedPointsTest = testCudaTrackOrb( n, _usMatchThreshod, _usHalfPatchSize, _sSearchRange, _sDescriptorByte,
															_cvgmParticleDescriptorPrev, _cvgmParticleResponsePrev, 
															_cvgmParticleDescriptorCurrTmp, _cvgmSaliency, 
															cvgmMinMatchDistanceTest, cvgmMatchedLocationPrevTest, _cvgmVelocityPrev2Curr);
		float fD0 = testMatDiff(_cvgmMatchedLocationPrev[n], cvgmMatchedLocationPrevTest[n]);
		float fD1 = testMatDiff(_cvgmMinMatchDistance[n],cvgmMinMatchDistanceTest[n]);
		float fD2 = (float)uMatchedPointsTest - nCounter;*/
		

		//separate tracked particles and rest of candidates. Note that saliency scores are updated 
		//Note that _cvgmSaliency is the input as well as output, after the tracked particles are separated with rest of candidates, their negative saliency
		//scores are recovered into positive scores
		_cvgmMatchedKeyPointLocation   [n].setTo(cv::Scalar::all(0));//clear all memory
		_cvgmMatchedKeyPointResponse   [n].setTo(0.f);
		_cvgmNewlyAddedKeyPointLocation[n].setTo(cv::Scalar::all(0));//clear all memory
		_cvgmNewlyAddedKeyPointResponse[n].setTo(0.f);
		btl::device::semidense::cudaCollectKeyPointOrb( _uTotalParticles[n], _uMaxKeyPointsAfterNonMax[n], 0.75f, _sDescriptorByte,
														_cvgmSaliency[n], _cvgmParticleDescriptorCurrTmp[n],
														_cvgmParticleVelocityPrev[n],_cvgmParticleAgePrev[n],
														_cvgmMinMatchDistance[n],_cvgmMatchedLocationPrev[n],
														&_cvgmNewlyAddedKeyPointLocation[n], &_cvgmNewlyAddedKeyPointResponse[n],
														&_cvgmMatchedKeyPointLocation[n], &_cvgmMatchedKeyPointResponse[n],
														&_cvgmParticleResponseCurr[n], &_cvgmParticleDescriptorCurr[n],
														&_cvgmParticleVelocityCurr[n],&_cvgmParticleAgeCurr[n]);
		/*int nCounter = 0;
		bool bIsLegal = testCountResponseAndDescriptor(_cvgmSaliency[n],_cvgmParticleDescriptorCurrTmp[n],&nCounter,_sDescriptorByte);

		nCounter = 0;
		bIsLegal = testCountResponseAndDescriptor(_cvgmParticleResponseCurr[n],_cvgmParticleDescriptorCurr[n],&nCounter,_sDescriptorByte);*/
		/*cv::gpu::GpuMat cvgmNewlyAddedKeyPointLocationTest(_cvgmNewlyAddedKeyPointLocation[n]), cvgmNewlyAddedKeyPointResponseTest(_cvgmNewlyAddedKeyPointResponse[n]), 
						cvgmMatchedKeyPointLocationTest(_cvgmMatchedKeyPointLocation[n]), cvgmMatchedKeyPointResponseTest(_cvgmMatchedKeyPointResponse[n]),
						cvgmParticleResponseCurrTest(_cvgmParticleResponseCurr[n]), cvgmParticleDescriptorCurrTest(_cvgmParticleDescriptorCurr[n]),
						cvgmParticleVelocityCurrTest(_cvgmParticleVelocityCurr[n]), cvgmParticleAgeCurrTest(_cvgmParticleAgeCurr[n]);
	
		nCounter = 0;
		bIsLegal = testCountResponseAndDescriptor(_cvgmSaliency[n], _cvgmParticleDescriptorCurrTmp[n],&nCounter,_sDescriptorByte);

		testCudaCollectNewlyAddedKeyPoints(_uTotalParticles[n], _uMaxKeyPointsAfterNonMax[n], 0.75f, _sDescriptorByte,
										  _cvgmSaliency[n], _cvgmParticleDescriptorCurrTmp[n],
										  _cvgmParticleVelocityPrev[n],_cvgmParticleAgePrev[n],
										  _cvgmMinMatchDistance[n],_cvgmMatchedLocationPrev[n],
										  &cvgmNewlyAddedKeyPointLocationTest, &cvgmNewlyAddedKeyPointResponseTest, 
										  &cvgmMatchedKeyPointLocationTest, &cvgmMatchedKeyPointResponseTest,
										  &cvgmParticleResponseCurrTest, &cvgmParticleDescriptorCurrTest,
										  &cvgmParticleVelocityCurrTest,&cvgmParticleAgeCurrTest);
		nCounter = 0;
		bIsLegal = testCountResponseAndDescriptor(_cvgmParticleResponseCurr[n],_cvgmParticleDescriptorCurr[n],&nCounter,_sDescriptorByte);
		float fD3 = testMatDiff(cvgmNewlyAddedKeyPointLocationTest, _cvgmNewlyAddedKeyPointLocation[n]);
		float fD4 = testMatDiff(cvgmNewlyAddedKeyPointResponseTest, _cvgmNewlyAddedKeyPointResponse[n]);
		float fD5 = testMatDiff(cvgmMatchedKeyPointLocationTest, _cvgmMatchedKeyPointLocation[n]);
		float fD6 = testMatDiff(cvgmMatchedKeyPointResponseTest, _cvgmMatchedKeyPointResponse[n]);
		float fD7 = testMatDiff(cvgmParticleResponseCurrTest, _cvgmParticleResponseCurr[n]);
		float fD8 = testMatDiff(cvgmParticleDescriptorCurrTest, _cvgmParticleDescriptorCurr[n]);
		float fD9 = testMatDiff(cvgmParticleVelocityCurrTest, _cvgmParticleVelocityCurr[n]);
		float fD10= testMatDiff(cvgmParticleAgeCurrTest, _cvgmParticleAgeCurr[n]);*/

		//h) assign the current frame to previous frame
		_cvgmBlurredCurr		 [n].copyTo(_cvgmBlurredPrev[n]);
		_cvgmParticleResponseCurr[n].copyTo(_cvgmParticleResponsePrev[n]);
		_cvgmParticleAgeCurr	 [n].copyTo(_cvgmParticleAgePrev[n]);
		_cvgmParticleVelocityCurr[n].copyTo(_cvgmParticleVelocityPrev[n]);
		_cvgmParticleDescriptorCurr[n].copyTo(_cvgmParticleDescriptorPrev[n]);
	}//for
	
	return;
}






