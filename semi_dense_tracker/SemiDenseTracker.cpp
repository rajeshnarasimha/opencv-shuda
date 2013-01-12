#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "SemiDenseTracker.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "Helper.hpp"

__device__ short2 operator + (const short2 s2O1_, const short2 s2O2_);
__device__ short2 operator - (const short2 s2O1_, const short2 s2O2_);
__device__ short2 operator * (const float fO1_, const short2 s2O2_);

unsigned int testCudaTrackFast(float fMatchThreshold_, const unsigned short usHalfSize_, const short sSearchRange_, 
							   const cv::gpu::GpuMat& cvgmParticleDescriptorPrev_, const cv::gpu::GpuMat& cvgmParticleResponsesPrev_, 
							   const cv::gpu::GpuMat& cvgmParticleDescriptorCurrTmp_, const cv::gpu::GpuMat& cvgmSaliencyCurr_, 
							   cv::gpu::GpuMat* pcvgmMinMatchDistance_,
							   cv::gpu::GpuMat* pcvgmMatchedLocationPrev_);
void testCudaCollectKeyPointsFast(unsigned int uTotalParticles_, unsigned int uMaxNewKeyPoints_, const float fRho_,
								  const cv::gpu::GpuMat& cvgmSaliency_, 
								  const cv::gpu::GpuMat& cvgmParticleDescriptorCurrTmp_,
								  const cv::gpu::GpuMat& cvgmParticleVelocityPrev_,
								  const cv::gpu::GpuMat& cvgmParticleAgePrev_,
								  const cv::gpu::GpuMat& cvgmMinMatchDistance_,
								  const cv::gpu::GpuMat& cvgmMatchedLocationPrev_, 
								  cv::gpu::GpuMat* pcvgmNewlyAddedKeyPointLocation_, cv::gpu::GpuMat* pcvgmNewlyAddedKeyPointResponse_,
								  cv::gpu::GpuMat* pcvgmMatchedKeyPointLocation_, cv::gpu::GpuMat* pcvgmMatchedKeyPointResponse_,
								  cv::gpu::GpuMat* pcvgmParticleResponseCurr_, cv::gpu::GpuMat* pcvgmParticleDescriptorCurr_,
								  cv::gpu::GpuMat* pcvgmParticleVelocityCurr_, cv::gpu::GpuMat* pcvgmParticleAgeCurr_);
bool testCountMinDistAndMatchedLocationFast(const cv::gpu::GpuMat cvgmMinMatchDistance_, const cv::gpu::GpuMat& cvgmMatchedLocationPrev_, int* pnCounter_);
bool testCountResponseAndDescriptorFast(const cv::gpu::GpuMat cvgmParticleResponse_, const cv::gpu::GpuMat& cvgmParticleDescriptor_, int* pnCounter_);
float testMatDiff(const cv::gpu::GpuMat& cvgm1_,const cv::gpu::GpuMat& cvgm2_ );

namespace btl{ namespace device{ namespace semidense{
	//for debug
	void cudaCalcMaxContrast(const cv::gpu::GpuMat& cvgmImage_, const unsigned char ucContrastThreshold_, cv::gpu::GpuMat* pcvgmContrast_);
	//for debug
	void cudaCalcMinDiameterContrast(const cv::gpu::GpuMat& cvgmImage_, cv::gpu::GpuMat* pcvgmContrast_);
	unsigned int cudaCalcSaliency(const cv::gpu::GpuMat& cvgmImage_, const unsigned short usHalfSizeRound_,
		const unsigned char ucContrastThreshold_, const float& fSaliencyThreshold_, 
		cv::gpu::GpuMat* pcvgmSaliency_, cv::gpu::GpuMat* pcvgmKeyPointLocations_);
	unsigned int cudaNonMaxSupression(const cv::gpu::GpuMat& cvgmKeyPointLocation_, const unsigned int uMaxSalientPoints_, 
		const cv::gpu::GpuMat& cvgmSaliency_, short2* ps2devLocations_, float* pfdevResponse_);
	//sort
	void thrustSort(short2* pnLoc_, float* pfResponse_, const unsigned int nCorners_);
	void cudaFastDescriptors(const cv::gpu::GpuMat& cvgmImage_, unsigned int uFinalSalientPoints_, cv::gpu::GpuMat* pcvgmKeyPointsLocations_, cv::gpu::GpuMat* pcvgmParticlesDescriptors_);
	unsigned int cudaPredictAndMatch(const unsigned int uFinalSalientPoints_, const cv::gpu::GpuMat& cvgmImage_,const cv::gpu::GpuMat& cvgmSaliency_, cv::gpu::GpuMat& cvgmFinalKeyPointsLocations_,cv::gpu::GpuMat& cvgmFinalKeyPointsResponse_,cv::gpu::GpuMat& cvgmParticlesAge_,cv::gpu::GpuMat& cvgmParticlesVelocity_, cv::gpu::GpuMat& cvgmParticlesDescriptors_);
	void cudaExtractAllDescriptorFast(const cv::gpu::GpuMat& cvgmImage_, 
									  const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, 
									  const unsigned int uTotalParticles_,  const unsigned int usHalfPatchSize_, 
									  cv::gpu::GpuMat* pcvgmParticleResponses_, cv::gpu::GpuMat* pcvgmParticleDescriptor_ );

	unsigned int cudaTrackFast(float fMatchThreshold_, const unsigned short usHalfSize_, const short sSearchRange_, 
								const cv::gpu::GpuMat& cvgmParticleDescriptorPrev_, const cv::gpu::GpuMat& cvgmParticleResponsesPrev_, 
								const cv::gpu::GpuMat& cvgmParticleDescriptorCurrTmp_, const cv::gpu::GpuMat& cvgmSaliencyCurr_, 
								cv::gpu::GpuMat* pcvgmMinMatchDistance_,
								cv::gpu::GpuMat* pcvgmMatchedLocationPrev_);
	void cudaCollectKeyPointsFast(unsigned int uTotalParticles_, unsigned int uMaxNewKeyPoints_, const float fRho_,
									const cv::gpu::GpuMat& cvgmSaliency_, 
									const cv::gpu::GpuMat& cvgmParticleDescriptorCurrTmp_,
									const cv::gpu::GpuMat& cvgmParticleVelocityPrev_,
									const cv::gpu::GpuMat& cvgmParticleAgePrev_,
									const cv::gpu::GpuMat& cvgmMinMatchDistance_,
									const cv::gpu::GpuMat& cvgmMatchedLocationPrev_, 
									cv::gpu::GpuMat* pcvgmNewlyAddedKeyPointLocation_, cv::gpu::GpuMat* pcvgmNewlyAddedKeyPointResponse_,
									cv::gpu::GpuMat* pcvgmMatchedKeyPointLocation_, cv::gpu::GpuMat* pcvgmMatchedKeyPointResponse_,
									cv::gpu::GpuMat* pcvgmParticleResponseCurr_, cv::gpu::GpuMat* pcvgmParticleDescriptorCurr_,
									cv::gpu::GpuMat* pcvgmParticleVelocityCurr_, cv::gpu::GpuMat* pcvgmParticleAgeCurr_);
}//semidense
}//device
}//btl

unsigned int testCudaTrack(const float fMatchThreshold_, const short sSearchRange_, 
	const cv::gpu::GpuMat& cvgmParticleDescriptorsPrev_, const cv::gpu::GpuMat& cvgmParticleResponsesPrev_,
	const cv::gpu::GpuMat& cvgmParticlesAgePrev_,const cv::gpu::GpuMat& cvgmParticlesVelocityPrev_, 
	const cv::gpu::GpuMat& cvgmBlurredCurr_,
	cv::gpu::GpuMat* pcvgmSaliency_,
	cv::gpu::GpuMat* pcvgmParticlesAgeCurr_,cv::gpu::GpuMat* pcvgmParticlesVelocityCurr_,cv::gpu::GpuMat* pcvgmParticleDescriptorsCurr_);


btl::image::semidense::CSemiDenseTracker::CSemiDenseTracker()
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

	_usHalfPatchSize = 6;
	_sSearchRange = 5;

}

bool btl::image::semidense::CSemiDenseTracker::initialize( cv::Mat& cvmColorFrame_ )
{
	_nFrameIdx = 0;
	_cvgmColorFrame.upload(cvmColorFrame_);
	cv::gpu::cvtColor(_cvgmColorFrame,_cvgmGrayFrame,cv::COLOR_RGB2GRAY);
	_cvgmSaliency.create(cvmColorFrame_.size(),CV_32FC1);
	_cvgmInitKeyPointLocation.create(1, _uMaxKeyPointsBeforeNonMax, CV_16SC2);
	_cvgmFinalKeyPointsLocationsAfterNonMax.create(1, _uMaxKeyPointsAfterNonMax, CV_16SC2);//short2 location;
	_cvgmFinalKeyPointsResponseAfterNonMax.create(1, _uMaxKeyPointsAfterNonMax, CV_32FC1);// float corner strength(response);  

	_cvgmMatchedKeyPointLocation.create(1, _uTotalParticles, CV_16SC2);
	_cvgmMatchedKeyPointResponse.create(1, _uTotalParticles, CV_32FC1);
	_cvgmNewlyAddedKeyPointLocation.create(1, _uMaxKeyPointsAfterNonMax, CV_16SC2);
	_cvgmNewlyAddedKeyPointResponse.create(1, _uMaxKeyPointsAfterNonMax, CV_32FC1);

	//init particles
	_cvgmParticleResponsePrev.create(cvmColorFrame_.size(),CV_32FC1);_cvgmParticleResponsePrev.setTo(0);
	_cvgmParticleVelocityPrev.create(cvmColorFrame_.size(),CV_16SC2);_cvgmParticleVelocityPrev.setTo(cv::Scalar::all(0));//float velocity; 
	_cvgmParticleAgePrev.create(cvmColorFrame_.size(),CV_8UC1);	  _cvgmParticleAgePrev.setTo(0);//uchar age;
	_cvgmParticleDescriptorPrev.create(cvmColorFrame_.size(),CV_32SC4);_cvgmParticleDescriptorPrev.setTo(cv::Scalar::all(0));

	_cvgmParticleResponseCurr.create(cvmColorFrame_.size(),CV_32FC1);_cvgmParticleResponseCurr.setTo(0);
	_cvgmParticleVelocityCurr.create(cvmColorFrame_.size(),CV_16SC2);_cvgmParticleVelocityCurr.setTo(cv::Scalar::all(0));//float velocity; 
	_cvgmParticleAgeCurr.create(cvmColorFrame_.size(),CV_8UC1);	  _cvgmParticleAgeCurr.setTo(0);//uchar age;
	_cvgmParticleDescriptorCurr.create(cvmColorFrame_.size(),CV_32SC4);_cvgmParticleDescriptorCurr.setTo(cv::Scalar::all(0));
	_cvgmParticleDescriptorCurrTmp.create(cvmColorFrame_.size(),CV_32SC4);_cvgmParticleDescriptorCurr.setTo(cv::Scalar::all(0));

	_cvgmMinMatchDistance.create(cvmColorFrame_.size(),CV_32FC1);
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
	unsigned int uTotalSalientPoints = btl::device::semidense::cudaCalcSaliency(_cvgmBlurredPrev, unsigned short(6*1.5), _ucContrastThresold, _fSaliencyThreshold, 
																				&_cvgmSaliency,&_cvgmInitKeyPointLocation); 
	if (uTotalSalientPoints< _uTotalParticles/2) return false;
	uTotalSalientPoints = std::min( uTotalSalientPoints, _uMaxKeyPointsBeforeNonMax );
	
	//2.do a non-max suppression and initialize particles ( extract feature descriptors ) 
	unsigned int uFinalSalientPointsAfterNonMax = btl::device::semidense::cudaNonMaxSupression(_cvgmInitKeyPointLocation, uTotalSalientPoints, _cvgmSaliency, 
																							   _cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(), _cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>() ); 
	uFinalSalientPointsAfterNonMax = std::min( uFinalSalientPointsAfterNonMax, _uMaxKeyPointsAfterNonMax );
	
	//3.sort all salient points according to their strength 
	btl::device::semidense::thrustSort(_cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(),_cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>(),uFinalSalientPointsAfterNonMax);
	_uTotalParticles = std::min( _uTotalParticles, uFinalSalientPointsAfterNonMax );
	
	//4.collect all salient points and descriptors on them
	_cvgmParticleResponsePrev.setTo(0.f);
	btl::device::semidense::cudaExtractAllDescriptorFast(_cvgmBlurredPrev, 
														 _cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(),_cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>(),
														 _uTotalParticles, _usHalfPatchSize,
												         &_cvgmParticleResponsePrev,&_cvgmParticleDescriptorPrev);

	//test
	/*int nCounter = 0;
	bool bIsLegal = testCountResponseAndDescriptorFast(_cvgmParticleResponsePrev,_cvgmParticleDescriptorPrev,&nCounter);*/

	//store velocity
	_cvgmParticleVelocityPrev.download(_cvmKeyPointVelocity[_nFrameIdx]);
	btl::other::increase<int>(30, &_nFrameIdx);

	cvmColorFrame_.setTo(cv::Scalar::all(255));
	//cv::putText(cvmColorFrame_, "Matthieu 2012", cv::Point(10, 15), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(1.,0.,0.) );
	return true;
}

void btl::image::semidense::CSemiDenseTracker::displayCandidates( cv::Mat& cvmColorFrame_ ){
	cv::Mat cvmKeyPoint;
	_cvgmFinalKeyPointsLocationsAfterNonMax.download(cvmKeyPoint);
	for (unsigned int i=0;i<_uFinalSalientPoints; i++){
		short2 ptCurr = cvmKeyPoint.ptr<short2>()[i];
		cv::circle(cvmColorFrame_,cv::Point(ptCurr.x,ptCurr.y),2,cv::Scalar(0,0,255.));
	}
}

void btl::image::semidense::CSemiDenseTracker::track( cv::Mat& cvmColorFrame_ )
{
	_cvgmColorFrame.upload(cvmColorFrame_);
	//convert the image to gray
	cv::gpu::cvtColor(_cvgmColorFrame,_cvgmGrayFrame,cv::COLOR_RGB2GRAY);
	//processing the frame
	//Gaussian smoothes the input image 
	_pBlurFilter->apply(_cvgmGrayFrame, _cvgmBlurredCurr, cv::Rect(0, 0, _cvgmGrayFrame.cols, _cvgmGrayFrame.rows));
	//calc the saliency score for each pixel
	unsigned int uTotalSalientPoints = btl::device::semidense::cudaCalcSaliency(_cvgmBlurredCurr, unsigned short( _usHalfPatchSize*1.5) /*the fast corner radius*/, _ucContrastThresold, _fSaliencyThreshold, &_cvgmSaliency, &_cvgmInitKeyPointLocation);
	uTotalSalientPoints = std::min( uTotalSalientPoints, _uMaxKeyPointsBeforeNonMax );
	
	//do a non-max suppression and collect the candidate particles into a temporary vectors( extract feature descriptors ) 
	unsigned int uFinalSalientPoints = btl::device::semidense::cudaNonMaxSupression(_cvgmInitKeyPointLocation, uTotalSalientPoints, _cvgmSaliency, 
		_cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(), _cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>() );
	_uFinalSalientPoints = uFinalSalientPoints = std::min( uFinalSalientPoints, unsigned int(_uMaxKeyPointsAfterNonMax) );
	_cvgmSaliency.setTo(0.f);//clear saliency scores
	//redeploy the saliency matrix
	btl::device::semidense::cudaExtractAllDescriptorFast(_cvgmBlurredCurr,
														_cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(),_cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>(),
														uFinalSalientPoints, _usHalfPatchSize,
														&_cvgmSaliency, &_cvgmParticleDescriptorCurrTmp);
	/*int nCounter = 0;
	bool bIsLegal = testCountResponseAndDescriptorFast(_cvgmSaliency,_cvgmParticleDescriptorCurrTmp,&nCounter);*/

	//track particles in previous frame by searching the candidates of current frame. 
	//Note that _cvgmSaliency is the input as well as output, tracked particles are marked as negative scores
	_cvgmParticleDescriptorCurr.setTo(cv::Scalar::all(0));_cvgmParticleAgeCurr.setTo(0);_cvgmParticleVelocityCurr.setTo(cv::Scalar::all(0));//clear all memory
	unsigned int uMatchedPoints = btl::device::semidense::cudaTrackFast(5.f,_usHalfPatchSize, _sSearchRange,
																		_cvgmParticleDescriptorPrev,  _cvgmParticleResponsePrev, 
																		_cvgmParticleDescriptorCurrTmp, _cvgmSaliency,
																		&_cvgmMinMatchDistance,
																		&_cvgmMatchedLocationPrev);
	/*
	nCounter = 0;
	bIsLegal = testCountMinDistAndMatchedLocationFast( _cvgmMinMatchDistance, _cvgmMatchedLocationPrev, &nCounter );
	cv::gpu::GpuMat _cvgmMinMatchDistanceTest(_cvgmMinMatchDistance),_cvgmMatchedLocationPrevTest(_cvgmMatchedLocationPrev);
	unsigned int uMatchedPointsTest = testCudaTrackFast(10.f,_usHalfPatchSize, _sSearchRange,
														_cvgmParticleDescriptorPrev,  _cvgmParticleResponsePrev, 
														_cvgmParticleDescriptorCurrTmp, _cvgmSaliency,
														&_cvgmMinMatchDistanceTest,
														&_cvgmMatchedLocationPrevTest);
	nCounter = 0;
	bIsLegal = testCountMinDistAndMatchedLocationFast( _cvgmMinMatchDistanceTest, _cvgmMatchedLocationPrevTest, &nCounter );
	float fD0 = testMatDiff(_cvgmMatchedLocationPrev, _cvgmMatchedLocationPrevTest);
	float fD1 = testMatDiff(_cvgmMinMatchDistance,_cvgmMinMatchDistanceTest);
	float fD2 = (float)uMatchedPointsTest - nCounter;
	*/
	//separate tracked particles and rest of candidates. Note that saliency scores are updated 
	//Note that _cvgmSaliency is the input as well as output, after the tracked particles are separated with rest of candidates, their negative saliency
	//scores are recovered into positive scores
	_cvgmMatchedKeyPointLocation   .setTo(cv::Scalar::all(0));//clear all memory
	_cvgmMatchedKeyPointResponse   .setTo(0.f);
	_cvgmNewlyAddedKeyPointLocation.setTo(cv::Scalar::all(0));//clear all memory
	_cvgmNewlyAddedKeyPointResponse.setTo(0.f);
	btl::device::semidense::cudaCollectKeyPointsFast(_uTotalParticles, _uMaxKeyPointsAfterNonMax, 0.75f,
													_cvgmSaliency, _cvgmParticleDescriptorCurrTmp,
													_cvgmParticleVelocityPrev,_cvgmParticleAgePrev,
													_cvgmMinMatchDistance,_cvgmMatchedLocationPrev,
													&_cvgmNewlyAddedKeyPointLocation, &_cvgmNewlyAddedKeyPointResponse, 
													&_cvgmMatchedKeyPointLocation, &_cvgmMatchedKeyPointResponse,
													&_cvgmParticleResponseCurr, &_cvgmParticleDescriptorCurr,
													&_cvgmParticleVelocityCurr,&_cvgmParticleAgeCurr);
	/*
	nCounter = 0;
	bIsLegal = testCountResponseAndDescriptorFast(_cvgmParticleResponseCurr,_cvgmParticleDescriptorCurr,&nCounter);
	
    cv::gpu::GpuMat _cvgmMatchedKeyPointLocationTest(_cvgmMatchedKeyPointLocation);      _cvgmMatchedKeyPointLocationTest.setTo(cv::Scalar::all(0));//clear all memory
	cv::gpu::GpuMat _cvgmMatchedKeyPointResponseTest(_cvgmMatchedKeyPointResponse);      _cvgmMatchedKeyPointResponseTest.setTo(0.f);
	cv::gpu::GpuMat _cvgmNewlyAddedKeyPointLocationTest(_cvgmNewlyAddedKeyPointLocation);_cvgmNewlyAddedKeyPointLocationTest.setTo(cv::Scalar::all(0));//clear all memory
	cv::gpu::GpuMat _cvgmNewlyAddedKeyPointResponseTest(_cvgmNewlyAddedKeyPointResponse);_cvgmNewlyAddedKeyPointResponseTest.setTo(0.f);
	cv::gpu::GpuMat _cvgmParticleResponseCurrTest(_cvgmParticleResponseCurr);
	cv::gpu::GpuMat _cvgmParticleDescriptorCurrTest(_cvgmParticleDescriptorCurr);
	cv::gpu::GpuMat _cvgmParticleVelocityCurrTest(_cvgmParticleVelocityCurr);
	cv::gpu::GpuMat _cvgmParticleAgeCurrTest(_cvgmParticleAgeCurr);

	testCudaCollectKeyPointsFast(_uTotalParticles, _uMaxKeyPointsAfterNonMax, 0.75f,
								_cvgmSaliency, _cvgmParticleDescriptorCurrTmp,
								_cvgmParticleVelocityPrev,_cvgmParticleAgePrev,
								_cvgmMinMatchDistance,_cvgmMatchedLocationPrev,
								&_cvgmNewlyAddedKeyPointLocationTest, &_cvgmNewlyAddedKeyPointResponseTest, 
								&_cvgmMatchedKeyPointLocationTest, &_cvgmMatchedKeyPointResponseTest,
								&_cvgmParticleResponseCurrTest, &_cvgmParticleDescriptorCurrTest,
								&_cvgmParticleVelocityCurrTest,&_cvgmParticleAgeCurrTest);


	nCounter = 0;
	bIsLegal = testCountResponseAndDescriptorFast(_cvgmParticleResponseCurrTest,_cvgmParticleDescriptorCurrTest,&nCounter);

	float fD3 = testMatDiff(_cvgmNewlyAddedKeyPointLocationTest, _cvgmNewlyAddedKeyPointLocation);
	float fD4 = testMatDiff(_cvgmNewlyAddedKeyPointResponseTest, _cvgmNewlyAddedKeyPointResponse);
	float fD5 = testMatDiff(_cvgmMatchedKeyPointLocationTest, _cvgmMatchedKeyPointLocation);
	float fD6 = testMatDiff(_cvgmMatchedKeyPointResponseTest, _cvgmMatchedKeyPointResponse);
	float fD7 = testMatDiff(_cvgmParticleResponseCurrTest, _cvgmParticleResponseCurr);
	float fD8 = testMatDiff(_cvgmParticleDescriptorCurrTest, _cvgmParticleDescriptorCurr);
	float fD9 = testMatDiff(_cvgmParticleVelocityCurrTest, _cvgmParticleVelocityCurr);
	float fD10= testMatDiff(_cvgmParticleAgeCurrTest, _cvgmParticleAgeCurr);
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
	float fAvgAge = 0.f; 
	for (unsigned int i=0;i<uMatchedPoints; i+=2){
		short2 ptCurr = _cvmKeyPointLocation.ptr<short2>()[i];
		uchar ucAge = _cvmKeyPointAge.ptr(ptCurr.y)[ptCurr.x];
		//if(ucAge < 2 ) continue;
		cv::circle(cvmColorFrame_,cv::Point(ptCurr.x,ptCurr.y),1,cv::Scalar(0,0,255.));
		short2 vi = _cvmKeyPointVelocity[_nFrameIdx].ptr<short2>(ptCurr.y)[ptCurr.x];
		int nFrameCurr = _nFrameIdx;
		fAvgAge += ucAge;
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
	fAvgAge /= uMatchedPoints;
	btl::other::increase<int>(30,&_nFrameIdx);
	return;	
}













