#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "SemiDenseTracker.h"


#include <cuda.h>
#include <cuda_runtime.h>

namespace btl{ namespace device{ namespace semidense{
	//for debug
	void cudaCalcMaxContrast(const cv::gpu::GpuMat& cvgmImage_, const unsigned char ucContrastThreshold_, cv::gpu::GpuMat* pcvgmContrast_);
	//for debug
	void cudaCalcMinDiameterContrast(const cv::gpu::GpuMat& cvgmImage_, cv::gpu::GpuMat* pcvgmContrast_);
	unsigned int cudaCalcSaliency(const cv::gpu::GpuMat& cvgmImage_, const unsigned char ucContrastThreshold_, const float& fSaliencyThreshold_, cv::gpu::GpuMat* pcvgmSaliency_, cv::gpu::GpuMat* pcvgmKeyPointLocations_);
	unsigned int cudaNonMaxSupression(const cv::gpu::GpuMat& cvgmImage_, const cv::gpu::GpuMat& cvgmKeyPointLocation_, const unsigned int uMaxSalientPoints_, cv::gpu::GpuMat* pcvgmSaliency_, short2* ps2devLocations_, float* pfdevResponse_);
	//sort
	void thrustSort(short2* pnLoc_, float* pfResponse_, const unsigned int nCorners_);
	void cudaFastDescriptors(const cv::gpu::GpuMat& cvgmImage_, unsigned int uFinalSalientPoints_, cv::gpu::GpuMat* pcvgmKeyPointsLocations_, cv::gpu::GpuMat* pcvgmParticlesDescriptors_);
	unsigned int cudaPredictAndMatch(const unsigned int uFinalSalientPoints_, const cv::gpu::GpuMat& cvgmImage_,const cv::gpu::GpuMat& cvgmSaliency_, cv::gpu::GpuMat& cvgmFinalKeyPointsLocations_,cv::gpu::GpuMat& cvgmFinalKeyPointsResponse_,cv::gpu::GpuMat& cvgmParticlesAge_,cv::gpu::GpuMat& cvgmParticlesVelocity_, cv::gpu::GpuMat& cvgmParticlesDescriptors_);
	void cudaCollectParticles(const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, const unsigned int uTotalParticles_, cv::gpu::GpuMat* pcvgmParticleResponses_);
	unsigned int cudaTrack(const cv::gpu::GpuMat& cvgmBlurredPrev_, const cv::gpu::GpuMat& cvgmParticleResponsesPrev_,const cv::gpu::GpuMat& cvgmParticlesAgePrev_,const cv::gpu::GpuMat& cvgmParticlesVelocityPrev_, const cv::gpu::GpuMat& cvgmBlurredCurr_,cv::gpu::GpuMat* pcvgmParticleResponsesCurr_,cv::gpu::GpuMat* pcvgmParticlesAgeCurr_,cv::gpu::GpuMat* pcvgmParticlesVelocityCurr_);
	unsigned int cudaMatchCollectionAndNonMaxSupression(const cv::gpu::GpuMat& cvgmKeyPointLocation_, unsigned int* puMaxSalientPoints_, cv::gpu::GpuMat* pcvgmParticleResponsesCurr_, short2* ps2devMatchedKeyPointLocations_, float* pfdevMatchedKeyPointResponse_, short2* ps2devNewlyAddedKeyPointLocations_, float* pfdevNewlyAddedKeyPointResponse_);

}//semidense
}//device
}//btl

btl::image::semidense::CSemiDenseTracker::CSemiDenseTracker()
{
	//Gaussian filter
	fSigma = 1.f; // page3: r=3/6 and sigma = 1.f/2.f respectively
	uRadius = 3; // 
	uSize = 2*uRadius + 1;
	//contrast threshold
	ucContrastThresold = 5; // 255 * 0.02 = 5.1

	//saliency threshold
	fSaliencyThreshold = 0.25;

	//# of Max key points
	_uInitMaxKeyPoints = 50000;
	_uFinalMaxKeyPoints= 20000;
	_uTotalParticles = 5000;
}

void btl::image::semidense::CSemiDenseTracker::init( cv::Mat& cvmColorFrame_ )
{
	_cvgmColorFrame.upload(cvmColorFrame_);
	_cvgmBufferC1.create(cvmColorFrame_.size(),CV_8UC1);
	_cvgmSaliency.create(cvmColorFrame_.size(),CV_32FC1);
	_cvgmInitKeyPointLocation.create(1, _uInitMaxKeyPoints, CV_16SC2);
	_cvgmFinalKeyPointsLocations.create(1, _uFinalMaxKeyPoints, CV_16SC2);//short2 location;
	_cvgmFinalKeyPointsResponse.create(1, _uFinalMaxKeyPoints, CV_32FC1);// float corner strength(response);  
	//init particles
	_cvgmParticlesVelocityPrev.create(1, _uTotalParticles, CV_16SC2); _cvgmParticlesVelocityPrev.setTo(cv::Scalar::all(0));//float velocity; 
	_cvgmParticlesAgePrev.create(1,_uTotalParticles,CV_8UC1);		  _cvgmParticlesAgePrev.setTo(0);//uchar age;
	_cvgmParticlesDescriptors.create(1,_uTotalParticles,CV_32SC4);_cvgmParticlesDescriptors.setTo(cv::Scalar::all(0));//16 byte feature descriptor
	
	//apply filter
	_pBlurFilter = cv::gpu::createGaussianFilter_GPU(CV_8UC3, cv::Size(uSize, uSize), fSigma, fSigma, cv::BORDER_REFLECT_101);

	//processing the frame
	//gaussian filter
	_pBlurFilter->apply(_cvgmColorFrame, _cvgmBlurred, cv::Rect(0, 0, _cvgmColorFrame.cols, _cvgmColorFrame.rows));
	//detect key points
	//1.compute the saliency score
	unsigned int uTotalSalientPoints = btl::device::semidense::cudaCalcSaliency(_cvgmBlurred, ucContrastThresold, fSaliencyThreshold, &_cvgmSaliency, &_cvgmInitKeyPointLocation);
	uTotalSalientPoints = std::min( uTotalSalientPoints, _uInitMaxKeyPoints );
	//2.do a non-max suppression and initialize particles ( extract feature descriptors )
	unsigned int uFinalSalientPoints = btl::device::semidense::cudaNonMaxSupression(_cvgmBlurred,_cvgmInitKeyPointLocation, uTotalSalientPoints, &_cvgmSaliency, _cvgmFinalKeyPointsLocations.ptr<short2>(), _cvgmFinalKeyPointsResponse.ptr<float>() );
	uFinalSalientPoints = std::min( uFinalSalientPoints, _uFinalMaxKeyPoints );
	//3.sort all salient points according to their strength
	btl::device::semidense::thrustSort(_cvgmFinalKeyPointsLocations.ptr<short2>(),_cvgmFinalKeyPointsResponse.ptr<float>(),uFinalSalientPoints);
	_uTotalParticles = std::min( _uTotalParticles, uFinalSalientPoints );
	//4.extract fast descriptor
	btl::device::semidense::cudaFastDescriptors(_cvgmBlurred,_uTotalParticles,&_cvgmFinalKeyPointsLocations,&_cvgmParticlesDescriptors);

	//convert saliency score to 3-channel RGB image
	_cvgmSaliency.convertTo(_cvgmBufferC1,CV_8UC1,255);
	cv::gpu::cvtColor(_cvgmBufferC1,_cvgmColorFrame,CV_GRAY2RGB);
	_cvgmColorFrame.download(cvmColorFrame_);
	//render keypoints
	_cvgmFinalKeyPointsLocations.download(_cvmKeyPointsLocations);
	for (unsigned int i=0;i<_uTotalParticles; i++){
		short2 s2Loc = _cvmKeyPointsLocations.ptr<short2>()[i];
		cv::circle(cvmColorFrame_,cv::Point(s2Loc.x,s2Loc.y),0,cv::Scalar(0,0,255.));
	}
	return;
}

void btl::image::semidense::CSemiDenseTracker::initialize( cv::Mat& cvmColorFrame_ )
{
	_cvgmColorFrame.upload(cvmColorFrame_);
	_cvgmBufferC1.create(cvmColorFrame_.size(),CV_8UC1);
	_cvgmSaliency.create(cvmColorFrame_.size(),CV_32FC1);
	_cvgmInitKeyPointLocation.create(1, _uInitMaxKeyPoints, CV_16SC2);
	_cvgmFinalKeyPointsLocations.create(1, _uFinalMaxKeyPoints, CV_16SC2);//short2 location;
	_cvgmFinalKeyPointsResponse.create(1, _uFinalMaxKeyPoints, CV_32FC1);// float corner strength(response);  

	_cvgmMatchedKeyPointsLocations.create(1, _uFinalMaxKeyPoints, CV_16SC2);
	_cvgmMatchedKeyPointsResponse.create(1, _uFinalMaxKeyPoints, CV_32FC1);
	_cvgmNewlyAddedKeyPointsLocations.create(1, _uFinalMaxKeyPoints/10, CV_16SC2);
	_cvgmNewlyAddedKeyPointsResponse.create(1, _uFinalMaxKeyPoints/10, CV_32FC1);

	//init particles
	_cvgmParticleResponsesPrev.create(cvmColorFrame_.size(),CV_32FC1);_cvgmParticleResponsesPrev.setTo(0);
	_cvgmParticlesVelocityPrev.create(cvmColorFrame_.size(),CV_16SC2);_cvgmParticlesVelocityPrev.setTo(cv::Scalar::all(0));//float velocity; 
	_cvgmParticlesAgePrev.create(cvmColorFrame_.size(),CV_8UC1);	  _cvgmParticlesAgePrev.setTo(0);//uchar age;

	_cvgmParticleResponsesCurr.create(cvmColorFrame_.size(),CV_32FC1);_cvgmParticleResponsesCurr.setTo(0);
	_cvgmParticlesVelocityCurr.create(cvmColorFrame_.size(),CV_16SC2);_cvgmParticlesVelocityCurr.setTo(cv::Scalar::all(0));//float velocity; 
	_cvgmParticlesAgeCurr.create(cvmColorFrame_.size(),CV_8UC1);	  _cvgmParticlesAgeCurr.setTo(0);//uchar age;



	//allocate filter
	_pBlurFilter = cv::gpu::createGaussianFilter_GPU(CV_8UC3, cv::Size(uSize, uSize), fSigma, fSigma, cv::BORDER_REFLECT_101);

	//processing the frame
	//0.gaussian filter 
	//a) cvgmBlurred(i) = Gaussian(cvgmImage); // gaussian filter the input image 
	_pBlurFilter->apply(_cvgmColorFrame, _cvgmBlurredPrev, cv::Rect(0, 0, _cvgmColorFrame.cols, _cvgmColorFrame.rows));
	//detect key points

	//1.compute the saliency score 
	//b) cvgmResponse = ExtractSalientPixels(cvgmBlurred(i));
	unsigned int uTotalSalientPoints = btl::device::semidense::cudaCalcSaliency(_cvgmBlurredPrev, ucContrastThresold, fSaliencyThreshold, &_cvgmSaliency, &_cvgmInitKeyPointLocation);
	uTotalSalientPoints = std::min( uTotalSalientPoints, _uInitMaxKeyPoints );
	
	//2.do a non-max suppression and initialize particles ( extract feature descriptors ) 
	//c) cvgmSupressed, KeyPoints, Response = NonMaxSupression(cvgmResponse);
	unsigned int uFinalSalientPoints = btl::device::semidense::cudaNonMaxSupression(_cvgmBlurredPrev,_cvgmInitKeyPointLocation, uTotalSalientPoints, &_cvgmSaliency, _cvgmFinalKeyPointsLocations.ptr<short2>(), _cvgmFinalKeyPointsResponse.ptr<float>() );
	uFinalSalientPoints = std::min( uFinalSalientPoints, _uFinalMaxKeyPoints );
	
	//3.sort all salient points according to their strength 
	//d) cvgmParitclesResponse(i) = Sort(KeyPoints, Response, cvgmSupressed, N); //choose top N strongest salient pixels are particles
	btl::device::semidense::thrustSort(_cvgmFinalKeyPointsLocations.ptr<short2>(),_cvgmFinalKeyPointsResponse.ptr<float>(),uFinalSalientPoints);
	_uTotalParticles = std::min( _uTotalParticles, uFinalSalientPoints );
	btl::device::semidense::cudaCollectParticles(_cvgmFinalKeyPointsLocations.ptr<short2>(),_cvgmFinalKeyPointsResponse.ptr<float>(),_uTotalParticles,&_cvgmParticleResponsesPrev);

	//convert saliency score to 3-channel RGB image
	_cvgmParticleResponsesPrev.convertTo(_cvgmBufferC1,CV_8UC1,255);
	cv::gpu::cvtColor(_cvgmBufferC1,_cvgmColorFrame,CV_GRAY2RGB);
	_cvgmColorFrame.download(cvmColorFrame_);
	//render keypoints
	_cvgmFinalKeyPointsLocations.download(_cvmKeyPointsLocations);
	for (unsigned int i=0;i<_uTotalParticles; i++){
		short2 s2Loc = _cvmKeyPointsLocations.ptr<short2>()[i];
		cv::circle(cvmColorFrame_,cv::Point(s2Loc.x,s2Loc.y),0,cv::Scalar(0,0,255.));
	}

	cv::imwrite("temp.png",cvmColorFrame_);
	return;
}
void btl::image::semidense::CSemiDenseTracker::track( cv::Mat& cvmColorFrame_ )
{
	_cvgmColorFrame.upload(cvmColorFrame_);
	//processing the frame
	//a) cvgmBlurred(i+1) = Gaussian(cvgmImage); // gaussian filter the input image 
	_pBlurFilter->apply(_cvgmColorFrame, _cvgmBlurredCurr, cv::Rect(0, 0, _cvgmColorFrame.cols, _cvgmColorFrame.rows));
	//b) cvgmResponse = ExtractSalientPixels(cvgmBlurred(i+1))
	unsigned int uTotalSalientPoints = btl::device::semidense::cudaCalcSaliency(_cvgmBlurredCurr, ucContrastThresold, fSaliencyThreshold, &_cvgmSaliency, &_cvgmInitKeyPointLocation);
	uTotalSalientPoints = std::min( uTotalSalientPoints, _uInitMaxKeyPoints );
	//c) for each PixelLocation in cvgmParitclesResponse(i)
	unsigned int uDeletedPoints = btl::device::semidense::cudaTrack(_cvgmBlurredPrev, _cvgmParticleResponsesPrev,_cvgmParticlesAgePrev,_cvgmParticlesVelocityPrev, _cvgmBlurredCurr,&_cvgmSaliency,&_cvgmParticlesAgeCurr,&_cvgmParticlesVelocityCurr);
	//e) KeyPoints, Response = NonMaxSupressionAndCollection(cvgmResponse );
	unsigned int uNewlyAdded = btl::device::semidense::cudaMatchCollectionAndNonMaxSupression(_cvgmInitKeyPointLocation, &uTotalSalientPoints, &_cvgmSaliency, _cvgmMatchedKeyPointsLocations.ptr<short2>(), _cvgmMatchedKeyPointsResponse.ptr<float>(), _cvgmNewlyAddedKeyPointsLocations.ptr<short2>(), _cvgmNewlyAddedKeyPointsResponse.ptr<float>() );
	unsigned int uMatched = uTotalSalientPoints;
	//f) cvgmParticlesResponse(i+1) = Sort( KeyPoint, Response, uDeletePoints)
	btl::device::semidense::thrustSort(_cvgmNewlyAddedKeyPointsLocations.ptr<short2>(),_cvgmNewlyAddedKeyPointsResponse.ptr<float>(),uNewlyAdded);
	uNewlyAdded = std::min( uNewlyAdded, uDeletedPoints );
	//g) collect keypoints
	_cvgmParticleResponsesCurr.setTo(0);
	btl::device::semidense::cudaCollectParticles(_cvgmNewlyAddedKeyPointsLocations.ptr<short2>(),_cvgmNewlyAddedKeyPointsResponse.ptr<float>(),uNewlyAdded,&_cvgmParticleResponsesCurr);
	btl::device::semidense::cudaCollectParticles(_cvgmMatchedKeyPointsLocations.ptr<short2>(),_cvgmMatchedKeyPointsResponse.ptr<float>(),uTotalSalientPoints,&_cvgmParticleResponsesCurr);
	//h) assign the current frame to previous frame
	_cvgmParticleResponsesCurr.copyTo(_cvgmParticleResponsesPrev);
	_cvgmParticlesAgeCurr.copyTo(_cvgmParticlesAgePrev);
	_cvgmParticlesVelocityCurr.copyTo(_cvgmParticlesVelocityPrev);
	_cvgmParticlesAgeCurr.copyTo(_cvgmParticlesAgePrev);
}
void btl::image::semidense::CSemiDenseTracker::tracking( cv::Mat& cvmColorFrame_ )
{
	_cvgmColorFrame.upload(cvmColorFrame_);
	//processing the frame
	//gaussian filter
	_pBlurFilter->apply(_cvgmColorFrame, _cvgmBlurred, cv::Rect(0, 0, _cvgmColorFrame.cols, _cvgmColorFrame.rows));
	//detect key points
	//1.compute the saliency score
	unsigned int uTotalSalientPoints = btl::device::semidense::cudaCalcSaliency(_cvgmBlurred, ucContrastThresold, fSaliencyThreshold, &_cvgmSaliency, &_cvgmInitKeyPointLocation);
	uTotalSalientPoints = std::min( uTotalSalientPoints, _uInitMaxKeyPoints );
	//2.do a non-max suppression and initialize particles ( extract feature descriptors )
	unsigned int uFinalSalientPoints = btl::device::semidense::cudaNonMaxSupression(_cvgmBlurred,_cvgmInitKeyPointLocation, uTotalSalientPoints, &_cvgmSaliency, _cvgmFinalKeyPointsLocations.ptr<short2>(), _cvgmFinalKeyPointsResponse.ptr<float>() );
	uFinalSalientPoints = std::min( uFinalSalientPoints, _uFinalMaxKeyPoints );

	//3.predict and match
	unsigned int uLostParticles = btl::device::semidense::cudaPredictAndMatch(uFinalSalientPoints, _cvgmBlurred, _cvgmSaliency,_cvgmFinalKeyPointsLocations,_cvgmFinalKeyPointsResponse,_cvgmParticlesAgePrev,_cvgmParticlesVelocityPrev,_cvgmParticlesDescriptors);

	//convert saliency score to 3-channel RGB image
	_cvgmSaliency.convertTo(_cvgmBufferC1,CV_8UC1,255);
	cv::gpu::cvtColor(_cvgmBufferC1,_cvgmColorFrame,CV_GRAY2RGB);
	_cvgmColorFrame.download(cvmColorFrame_);
	//render keypoints
	_cvgmFinalKeyPointsLocations.download(_cvmKeyPointsLocations);
	for (unsigned int i=0;i<_uTotalParticles; i++){
		short2 s2Loc = _cvmKeyPointsLocations.ptr<short2>()[i];
		cv::circle(cvmColorFrame_,cv::Point(s2Loc.x,s2Loc.y),0,cv::Scalar(0,0,255.));
	}
	return;
}
