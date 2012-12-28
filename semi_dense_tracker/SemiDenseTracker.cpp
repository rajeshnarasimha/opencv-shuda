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
	unsigned int cudaNonMaxSupression(const cv::gpu::GpuMat& cvgmKeyPointLocation_, const unsigned int uMaxSalientPoints_, 
		const cv::gpu::GpuMat& cvgmSaliency_, short2* ps2devLocations_, float* pfdevResponse_);
	//sort
	void thrustSort(short2* pnLoc_, float* pfResponse_, const unsigned int nCorners_);
	void cudaFastDescriptors(const cv::gpu::GpuMat& cvgmImage_, unsigned int uFinalSalientPoints_, cv::gpu::GpuMat* pcvgmKeyPointsLocations_, cv::gpu::GpuMat* pcvgmParticlesDescriptors_);
	unsigned int cudaPredictAndMatch(const unsigned int uFinalSalientPoints_, const cv::gpu::GpuMat& cvgmImage_,const cv::gpu::GpuMat& cvgmSaliency_, cv::gpu::GpuMat& cvgmFinalKeyPointsLocations_,cv::gpu::GpuMat& cvgmFinalKeyPointsResponse_,cv::gpu::GpuMat& cvgmParticlesAge_,cv::gpu::GpuMat& cvgmParticlesVelocity_, cv::gpu::GpuMat& cvgmParticlesDescriptors_);
	void cudaCollectParticles(const short2* ps2KeyPointsLocations_, const float* pfKeyPointsResponse_, const unsigned int uTotalParticles_, 
		cv::gpu::GpuMat* pcvgmParticleResponses_, cv::gpu::GpuMat* pcvgmParticleDescriptor_, const cv::gpu::GpuMat& cvgmImage_=cv::gpu::GpuMat() );

	unsigned int cudaTrack(const float fMatchThreshold_, const short sSearchRange_, const cv::gpu::GpuMat& cvgmParticleDescriptorsPrev_, const cv::gpu::GpuMat& cvgmParticleResponsesPrev_,const cv::gpu::GpuMat& cvgmParticlesAgePrev_,const cv::gpu::GpuMat& cvgmParticlesVelocityPrev_, const cv::gpu::GpuMat& cvgmBlurredCurr_,
		cv::gpu::GpuMat* pcvgmParticleResponsesCurr_,cv::gpu::GpuMat* pcvgmParticlesAgeCurr_,cv::gpu::GpuMat* pcvgmParticlesVelocityCurr_,cv::gpu::GpuMat* pcvgmParticleDescriptorsCurr_);
	
	unsigned int cudaMatchedAndNewlyAddedKeyPointsCollection(cv::gpu::GpuMat& cvgmKeyPointLocation_, unsigned int* puMaxSalientPoints_, cv::gpu::GpuMat* pcvgmParticleResponsesCurr_, short2* ps2devMatchedKeyPointLocations_, float* pfdevMatchedKeyPointResponse_, short2* ps2devNewlyAddedKeyPointLocations_, float* pfdevNewlyAddedKeyPointResponse_);
}//semidense
}//device
}//btl

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

	_sSearchRange = 10;

}

#define __device__
#define __float2int_rn short
__device__ short2 operator + (const short2 s2O1_, const short2 s2O2_){
	return make_short2(s2O1_.x + s2O2_.x,s2O1_.y + s2O2_.y);
}
__device__ short2 operator - (const short2 s2O1_, const short2 s2O2_){
	return make_short2(s2O1_.x - s2O2_.x,s2O1_.y - s2O2_.y);
}
__device__ short2 operator * (const float fO1_, const short2 s2O2_){
	return make_short2( __float2int_rn(fO1_* s2O2_.x),__float2int_rn( fO1_ * s2O2_.y));
}

float dL1(const int4& n4Descriptor1_, const int4& n4Descriptor2_){
	float fDist = 0.f;
	uchar uD1,uD2;
	for (uchar u=0; u < 4; u++){
		uD1 = (n4Descriptor1_.x >> u*8) & 0xFF;
		uD2 = (n4Descriptor2_.x >> u*8) & 0xFF;
		fDist += abs(uD1 - uD2); 
		uD1 = (n4Descriptor1_.y >> u*8) & 0xFF;
		uD2 = (n4Descriptor2_.y >> u*8) & 0xFF;
		fDist += abs(uD1 - uD2); 
		uD1 = (n4Descriptor1_.z >> u*8) & 0xFF;
		uD2 = (n4Descriptor2_.z >> u*8) & 0xFF;
		fDist += abs(uD1 - uD2); 
		uD1 = (n4Descriptor1_.w >> u*8) & 0xFF;
		uD2 = (n4Descriptor2_.w >> u*8) & 0xFF;
		fDist += abs(uD1 - uD2); 
	}
	fDist /= 16;
	return fDist;
}
void devGetFastDescriptor(const cv::Mat& cvgmImage_, const int r, const int c, int4* pDescriptor_ ){
	pDescriptor_->x = pDescriptor_->y = pDescriptor_->z = pDescriptor_->w = 0;
	uchar Color;
	Color = cvgmImage_.ptr<uchar>(r-3)[c  ];//1
	pDescriptor_->x += Color; 
	pDescriptor_->x = pDescriptor_->x << 8;
	Color = cvgmImage_.ptr<uchar>(r-3)[c+1];//2
	pDescriptor_->x += Color; 
	pDescriptor_->x = pDescriptor_->x << 8;
	Color = cvgmImage_.ptr<uchar>(r-2)[c+2];//3
	pDescriptor_->x += Color; 
	pDescriptor_->x = pDescriptor_->x << 8;
	Color = cvgmImage_.ptr<uchar>(r-1)[c+3];//4
	pDescriptor_->x += Color; 

	Color = cvgmImage_.ptr<uchar>(r  )[c+3];//5
	pDescriptor_->y += Color; 
	pDescriptor_->y = pDescriptor_->y << 8;
	Color = cvgmImage_.ptr<uchar>(r+1)[c+3];//6
	pDescriptor_->y += Color; 
	pDescriptor_->y = pDescriptor_->y << 8;
	Color = cvgmImage_.ptr<uchar>(r+2)[c+2];//7
	pDescriptor_->y += Color; 
	pDescriptor_->y = pDescriptor_->y << 8;
	Color = cvgmImage_.ptr<uchar>(r+3)[c+1];//8
	pDescriptor_->y += Color; 

	Color = cvgmImage_.ptr<uchar>(r+3)[c  ];//9
	pDescriptor_->z += Color; 
	pDescriptor_->z = pDescriptor_->z << 8;
	Color= cvgmImage_.ptr<uchar>(r+3)[c-1];//10
	pDescriptor_->z += Color; 
	pDescriptor_->z = pDescriptor_->z << 8;
	Color= cvgmImage_.ptr<uchar>(r+2)[c-2];//11
	pDescriptor_->z += Color; 
	pDescriptor_->z = pDescriptor_->z << 8;
	Color= cvgmImage_.ptr<uchar>(r+1)[c-3];//12
	pDescriptor_->z += Color; 

	Color= cvgmImage_.ptr<uchar>(r  )[c-3];//13
	pDescriptor_->w += Color; 
	pDescriptor_->w = pDescriptor_->w << 8;
	Color= cvgmImage_.ptr<uchar>(r-1)[c-3];//14
	pDescriptor_->w += Color; 
	pDescriptor_->w = pDescriptor_->w << 8;
	Color= cvgmImage_.ptr<uchar>(r-2)[c-2];//15
	pDescriptor_->w += Color; 
	pDescriptor_->w = pDescriptor_->w << 8;
	Color= cvgmImage_.ptr<uchar>(r-3)[c-1];//16
	pDescriptor_->w += Color; 
	return;
}
__device__ float devMatch(const float fMatchThreshold_, const int4& n4DesPrev_, const cv::Mat& cvgmImage_, const cv::Mat& cvgmScore_,short2* ps2Loc_,int4 *pn4Descriptor_ ){
	float fResponse = 0.f;
	float fBestMatchedResponse;
	short2 s2Loc,s2BestLoc;
	float fMinDist = 300.f;
	int4 n4BestDescriptor;
	//search for the 7x7 neighbourhood for 
	for(short r = -3; r < 4; r++ ){
		for(short c = -3; c < 4; c++ ){
			s2Loc = *ps2Loc_ + make_short2( c, r ); 
			fResponse = cvgmScore_.ptr<float>(s2Loc.y)[s2Loc.x];
			if( fResponse > 0 ){
				int4 n4Des; 
				devGetFastDescriptor(cvgmImage_,s2Loc.y,s2Loc.x,&n4Des);
				float fDist = dL1(n4Des,n4DesPrev_);
				if ( fDist < fMatchThreshold_ ){
					if (  fMinDist > fDist ){
						fMinDist = fDist;
						fBestMatchedResponse = fResponse;
						n4BestDescriptor = n4Des;
						s2BestLoc = s2Loc;		 
					}
				}
			}//if sailent corner exits
		}//for 
	}//for
	if(fMinDist < 300.f){
		*ps2Loc_ = s2BestLoc;
		*pn4Descriptor_ = n4BestDescriptor;
		return fBestMatchedResponse;
	}
	else
		return -1.f;
}


void increase(const int nCycle_,int* pnIdx_ );
void decrease(const int nCycle_,int* pnIdx_ );

void btl::image::semidense::CSemiDenseTracker::initialize( cv::Mat& cvmColorFrame_ )
{
	_nFrameIdx = 0;
	_cvgmColorFrame.upload(cvmColorFrame_);
	cv::gpu::cvtColor(_cvgmColorFrame,_cvgmGrayFrame,cv::COLOR_RGB2GRAY);
	_cvgmSaliency.create(cvmColorFrame_.size(),CV_32FC1);
	_cvgmInitKeyPointLocation.create(1, _uMaxKeyPointsBeforeNonMax, CV_16SC2);
	_cvgmFinalKeyPointsLocationsAfterNonMax.create(1, _uMaxKeyPointsAfterNonMax, CV_16SC2);//short2 location;
	_cvgmFinalKeyPointsResponseAfterNonMax.create(1, _uMaxKeyPointsAfterNonMax, CV_32FC1);// float corner strength(response);  

	_cvgmMatchedKeyPointsLocations.create(1, _uTotalParticles, CV_16SC2);
	_cvgmMatchedKeyPointsResponse.create(1, _uTotalParticles, CV_32FC1);
	_cvgmNewlyAddedKeyPointsLocations.create(1, _uMaxKeyPointsAfterNonMax, CV_16SC2);
	_cvgmNewlyAddedKeyPointsResponse.create(1, _uMaxKeyPointsAfterNonMax, CV_32FC1);

	//init particles
	_cvgmParticleResponsesPrev.create(cvmColorFrame_.size(),CV_32FC1);_cvgmParticleResponsesPrev.setTo(0);
	_cvgmParticlesVelocityPrev.create(cvmColorFrame_.size(),CV_16SC2);_cvgmParticlesVelocityPrev.setTo(cv::Scalar::all(0));//float velocity; 
	_cvgmParticlesAgePrev.create(cvmColorFrame_.size(),CV_8UC1);	  _cvgmParticlesAgePrev.setTo(0);//uchar age;
	_cvgmParticleDescriptorsPrev.create(cvmColorFrame_.size(),CV_32SC4);_cvgmParticleDescriptorsPrev.setTo(cv::Scalar::all(0));

	_cvgmParticleResponsesCurr.create(cvmColorFrame_.size(),CV_32FC1);_cvgmParticleResponsesCurr.setTo(0);
	_cvgmParticlesVelocityCurr.create(cvmColorFrame_.size(),CV_16SC2);_cvgmParticlesVelocityCurr.setTo(cv::Scalar::all(0));//float velocity; 
	_cvgmParticlesAgeCurr.create(cvmColorFrame_.size(),CV_8UC1);	  _cvgmParticlesAgeCurr.setTo(0);//uchar age;
	_cvgmParticleDescriptorsCurr.create(cvmColorFrame_.size(),CV_32SC4);_cvgmParticleDescriptorsCurr.setTo(cv::Scalar::all(0));

	//allocate filter
	if (_pBlurFilter.empty()){
		_pBlurFilter = cv::gpu::createGaussianFilter_GPU(CV_8UC1, cv::Size(_uGaussianKernelSize, _uGaussianKernelSize), _fSigma, _fSigma, cv::BORDER_REFLECT_101);
	}

	//processing the frame
	//0.gaussian filter 
	//a) cvgmBlurred(i) = Gaussian(cvgmImage); // gaussian filter the input image 
	_pBlurFilter->apply(_cvgmGrayFrame, _cvgmBlurredPrev, cv::Rect(0, 0, _cvgmGrayFrame.cols, _cvgmGrayFrame.rows));
	//detect key points

	//1.compute the saliency score 
	//b) cvgmResponse = ExtractSalientPixels(cvgmBlurred(i));
	unsigned int uTotalSalientPoints = btl::device::semidense::cudaCalcSaliency(_cvgmBlurredPrev, _ucContrastThresold, _fSaliencyThreshold, 
		&_cvgmSaliency,&_cvgmInitKeyPointLocation);
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
	btl::device::semidense::cudaCollectParticles(_cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(),_cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>(),_uTotalParticles,
		&_cvgmParticleResponsesPrev,&_cvgmParticleDescriptorsPrev,_cvgmBlurredPrev);
	
	_cvgmParticlesVelocityPrev.download(_cvmKeyPointVelocitys[_nFrameIdx]);
	increase(30,&_nFrameIdx);
	cvmColorFrame_.setTo(cv::Scalar::all(255));
	cv::putText(cvmColorFrame_, "Matthieu 2012", cv::Point(10, 15), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(1.,0.,0.) );
	return;
}

void btl::image::semidense::CSemiDenseTracker::displayCandidates( cv::Mat& cvmColorFrame_ ){
	cv::Mat cvmKeyPoint;
	_cvgmFinalKeyPointsLocationsAfterNonMax.download(cvmKeyPoint);
	for (unsigned int i=0;i<_uFinalSalientPoints; i++){
		short2 ptCurr = cvmKeyPoint.ptr<short2>()[i];
		cv::circle(cvmColorFrame_,cv::Point(ptCurr.x,ptCurr.y),2,cv::Scalar(0,0,255.));
	}
	cv::putText(cvmColorFrame_, "Fast Keypoints", cv::Point(10, 15), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(1.,0.,0.) );
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
	unsigned int uTotalSalientPoints = btl::device::semidense::cudaCalcSaliency(_cvgmBlurredCurr, _ucContrastThresold, _fSaliencyThreshold, &_cvgmSaliency, &_cvgmInitKeyPointLocation);
	uTotalSalientPoints = std::min( uTotalSalientPoints, _uMaxKeyPointsBeforeNonMax );
	
	//do a non-max suppression and collect the candidate particles into a temporary vectors( extract feature descriptors ) 
	unsigned int uFinalSalientPoints = btl::device::semidense::cudaNonMaxSupression(_cvgmInitKeyPointLocation, uTotalSalientPoints, _cvgmSaliency, 
		_cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(), _cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>() );
	_uFinalSalientPoints = uFinalSalientPoints = std::min( uFinalSalientPoints, unsigned int(_uMaxKeyPointsAfterNonMax) );
	_cvgmSaliency.setTo(0.f);//clear saliency scores
	//redeploy the saliency matrix
	btl::device::semidense::cudaCollectParticles(_cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(),_cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>(),uFinalSalientPoints,
		&_cvgmSaliency,NULL);

	//track particles in previous frame by searching the candidates of current frame. 
	//Note that _cvgmSaliency is the input as well as output, tracked particles are marked as negative scores
	_cvgmParticleDescriptorsCurr.setTo(cv::Scalar::all(0));_cvgmParticlesAgeCurr.setTo(0);_cvgmParticlesVelocityCurr.setTo(cv::Scalar::all(0));//clear all memory
	unsigned int uDeletedPoints = btl::device::semidense::cudaTrack(10.f,_sSearchRange,
		_cvgmParticleDescriptorsPrev, _cvgmParticleResponsesPrev, _cvgmParticlesAgePrev, _cvgmParticlesVelocityPrev, _cvgmBlurredCurr, 
		&_cvgmSaliency, &_cvgmParticlesAgeCurr, &_cvgmParticlesVelocityCurr, &_cvgmParticleDescriptorsCurr);
	//separate tracked particles and rest of candidates. Note that saliency scores are updated 
	//Note that _cvgmSaliency is the input as well as output, after the tracked particles are separated with rest of candidates, their negative saliency
	//scores are recovered into positive scores
	unsigned int uNewlyAdded = btl::device::semidense::cudaMatchedAndNewlyAddedKeyPointsCollection(_cvgmFinalKeyPointsLocationsAfterNonMax, &uFinalSalientPoints, 
		&_cvgmSaliency, //s
		_cvgmMatchedKeyPointsLocations.ptr<short2>(), _cvgmMatchedKeyPointsResponse.ptr<float>(), //tracked particles
		_cvgmNewlyAddedKeyPointsLocations.ptr<short2>(), _cvgmNewlyAddedKeyPointsResponse.ptr<float>() ); //rest of candidates
	uNewlyAdded = std::min(uNewlyAdded,_uMaxKeyPointsAfterNonMax);
	unsigned int uMatched = std::min(uFinalSalientPoints,_uTotalParticles);
	//rest of candidates are ranked according to their strength, and some of them are selected as new particles and added into the set of particles
	btl::device::semidense::thrustSort(_cvgmNewlyAddedKeyPointsLocations.ptr<short2>(),_cvgmNewlyAddedKeyPointsResponse.ptr<float>(),uNewlyAdded);
	uNewlyAdded = std::min( uNewlyAdded, _uTotalParticles - uMatched );
	//g) collect particles
	_cvgmParticleResponsesCurr.setTo(0);
	btl::device::semidense::cudaCollectParticles(_cvgmMatchedKeyPointsLocations.ptr<short2>(),_cvgmMatchedKeyPointsResponse.ptr<float>(),uMatched,
		&_cvgmParticleResponsesCurr,NULL);//collect tracked particles
	btl::device::semidense::cudaCollectParticles(_cvgmNewlyAddedKeyPointsLocations.ptr<short2>(),_cvgmNewlyAddedKeyPointsResponse.ptr<float>(),uNewlyAdded,
		&_cvgmParticleResponsesCurr,&_cvgmParticleDescriptorsCurr,_cvgmBlurredCurr);//collect new particles and calc their descriptors
	
	//h) assign the current frame to previous frame
	_cvgmParticleResponsesCurr.copyTo(_cvgmParticleResponsesPrev);
	_cvgmParticleDescriptorsCurr.copyTo(_cvgmParticleDescriptorsPrev);
	_cvgmParticlesAgeCurr.copyTo(_cvgmParticlesAgePrev);
	_cvgmParticlesVelocityCurr.copyTo(_cvgmParticlesVelocityPrev);
	_cvgmBlurredCurr.copyTo(_cvgmBlurredPrev);

	//render keypoints
	_cvgmMatchedKeyPointsLocations.download(_cvmKeyPointsLocations);
	_cvgmParticlesAgeCurr.download(_cvmKeyPointsAge);
	//store velocity
	_cvgmParticlesVelocityCurr.download(_cvmKeyPointVelocitys[_nFrameIdx]);
	cvmColorFrame_.setTo(cv::Scalar::all(255));
	for (unsigned int i=0;i<uMatched; i++){
		short2 ptCurr = _cvmKeyPointsLocations.ptr<short2>()[i];
		cv::circle(cvmColorFrame_,cv::Point(ptCurr.x,ptCurr.y),1,cv::Scalar(0,0,255.));
		//render trajectory 
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

	cv::putText(cvmColorFrame_, "Matthieu 2012", cv::Point(10, 15), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(1.,0.,0.) );
	increase(30, &_nFrameIdx);

	return;	
}

void btl::image::semidense::CSemiDenseTracker::trackTest( cv::Mat& cvmColorFrame_ )
{
	_cvgmColorFrame.upload(cvmColorFrame_);
	cv::gpu::cvtColor(_cvgmColorFrame,_cvgmGrayFrame,cv::COLOR_RGB2GRAY);
	//processing the frame
	//a) cvgmBlurred(i+1) = Gaussian(cvgmImage); // gaussian filter the input image 
	_pBlurFilter->apply(_cvgmGrayFrame, _cvgmBlurredCurr, cv::Rect(0, 0, _cvgmColorFrame.cols, _cvgmColorFrame.rows));
	//b) cvgmResponse = ExtractSalientPixels(cvgmBlurred(i+1))
	unsigned int uTotalSalientPoints = btl::device::semidense::cudaCalcSaliency(_cvgmBlurredCurr, _ucContrastThresold, _fSaliencyThreshold, &_cvgmSaliency, &_cvgmInitKeyPointLocation);
	uTotalSalientPoints = std::min( uTotalSalientPoints, _uMaxKeyPointsBeforeNonMax );
	//2.do a non-max suppression and initialize particles ( extract feature descriptors ) 
	//c) cvgmSupressed, KeyPoints, Response = NonMaxSupression(cvgmResponse);
	unsigned int uFinalSalientPoints = btl::device::semidense::cudaNonMaxSupression(_cvgmInitKeyPointLocation, uTotalSalientPoints, _cvgmSaliency, _cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(), _cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>() );
	uFinalSalientPoints = std::min( uFinalSalientPoints, unsigned int(_uMaxKeyPointsAfterNonMax) );
	_cvgmSaliency.setTo(0.f);
	btl::device::semidense::cudaCollectParticles(_cvgmFinalKeyPointsLocationsAfterNonMax.ptr<short2>(),_cvgmFinalKeyPointsResponseAfterNonMax.ptr<float>(),uFinalSalientPoints,&_cvgmSaliency,NULL);

	cv::Mat cvmBlurredCurr; _cvgmBlurredCurr.download(cvmBlurredCurr);
	cv::Mat cvmBlurredPrev; _cvgmBlurredPrev.download(cvmBlurredPrev);
	cv::Mat cvmParticleResponsePrev; _cvgmParticleResponsesPrev.download(cvmParticleResponsePrev);
	cv::Mat cvmParticlesVelocityPrev; _cvgmParticlesVelocityPrev.download(cvmParticlesVelocityPrev);
	cv::Mat cvmSaliency; _cvgmSaliency.download(cvmSaliency);
	cv::Mat cvmParticlesVelocityCurr; _cvgmParticlesVelocityCurr.download(cvmParticlesVelocityCurr);
	cv::Mat cvmParticlesAgeCurr; _cvgmParticlesAgeCurr.download(cvmParticlesAgeCurr);
	cv::Mat cvmParticlesAgePrev; _cvgmParticlesAgePrev.download(cvmParticlesAgePrev);
	cv::Mat cvmParticleResponsesCurr; _cvgmParticleResponsesCurr.download(cvmParticleResponsesCurr);
	cv::Mat cvmParticleDescriptorPrev; _cvgmParticleDescriptorsPrev.download(cvmParticleDescriptorPrev); 
	cv::Mat cvmParticleDescriptorCurr; cvmParticleDescriptorCurr.create(_cvgmParticleDescriptorsPrev.size(),CV_32SC4 );

	float _fRho = 0.75f;
	int nDelete = 0;
	int nMatched= 0;
	for (int c = 3; c<cvmBlurredPrev.cols-4; c++ )
	for (int r = 3; r<cvmBlurredPrev.rows-4; r++ ){
		//if IsParticle( PixelLocation, cvgmParitclesResponse(i) )
		if ( cvmParticleResponsePrev.ptr<float>(r)[c] < 0.2f ) continue;
		//A) PredictLocation = PixelLocation + ParticleVelocity(i, PixelLocation);
		short2 s2PredictLoc = make_short2(c,r);// + cvmParticlesVelocityPrev.ptr<short2>(r)[c];
		//check whether the predicted point is inside the image
		if (s2PredictLoc.x >=12 && s2PredictLoc.x < cvmBlurredPrev.cols-13 && s2PredictLoc.y >=12 && s2PredictLoc.y < cvmBlurredPrev.rows-13)
		{
			//B) ActualLocation = Match(PredictLocation, cvgmBlurred(i),cvgmBlurred(i+1));
			int4 n4DesPrev = cvmParticleDescriptorPrev.ptr<int4>(s2PredictLoc.y)[s2PredictLoc.x];//;	devGetFastDescriptor(cvmBlurredPrev,r,c,&n4DesPrev);
			int4 n4DenCurr;
			float fResponse = devMatch( 70.f, n4DesPrev, cvmBlurredCurr, cvmSaliency, &s2PredictLoc, &n4DenCurr );

			if(fResponse > 0)
			{
				nMatched++;
				cvmParticleDescriptorCurr.ptr<int4>(s2PredictLoc.y)[s2PredictLoc.x]  = n4DenCurr;
				short2 s2VeCur = _fRho * (s2PredictLoc - make_short2(c,r)) + (1.f - _fRho)* cvmParticlesVelocityPrev.ptr<short2>(r)[c];//update velocity
				cvmParticlesVelocityCurr.ptr<short2>(s2PredictLoc.y)[s2PredictLoc.x] = s2VeCur;
				cvmParticlesAgeCurr.ptr			    (s2PredictLoc.y)[s2PredictLoc.x] = cvmParticlesAgePrev.ptr(s2PredictLoc.y)[s2PredictLoc.x] + 1; //update age
				cvmParticleResponsesCurr.ptr<float> (s2PredictLoc.y)[s2PredictLoc.x] = -fResponse; //update response and location //marked as matched and it will be corrected in NoMaxAndCollection
			}
			else{//C) if no match found 
				nDelete ++;
				//atomicInc(&_devuCounter, (unsigned int)(-1));//deleted particle counter increase by 1
			}//lost
		}
		else{
			nDelete ++;
		}
	}

	//c) for each PixelLocation in cvgmParitclesResponse(i)
	_cvgmParticleDescriptorsCurr.setTo(cv::Scalar::all(0));
	unsigned int uDeletedPoints = btl::device::semidense::cudaTrack(70.f,_sSearchRange,_cvgmParticleDescriptorsPrev, _cvgmParticleResponsesPrev,_cvgmParticlesAgePrev,_cvgmParticlesVelocityPrev, _cvgmBlurredCurr,
		&_cvgmSaliency,&_cvgmParticlesAgeCurr,&_cvgmParticlesVelocityCurr,&_cvgmParticleDescriptorsCurr);
	//e) KeyPoints, Response = NonMaxSupressionAndCollection(cvgmResponse );
	unsigned int uNewlyAdded = btl::device::semidense::cudaMatchedAndNewlyAddedKeyPointsCollection(_cvgmInitKeyPointLocation, &uTotalSalientPoints, &_cvgmSaliency, _cvgmMatchedKeyPointsLocations.ptr<short2>(), _cvgmMatchedKeyPointsResponse.ptr<float>(), _cvgmNewlyAddedKeyPointsLocations.ptr<short2>(), _cvgmNewlyAddedKeyPointsResponse.ptr<float>() );
	unsigned int uMatched = uTotalSalientPoints;
	uNewlyAdded = std::min(uNewlyAdded,_uMaxKeyPointsAfterNonMax/10);
	//f) cvgmParticlesResponse(i+1) = Sort( KeyPoint, Response, uDeletePoints)
	btl::device::semidense::thrustSort(_cvgmNewlyAddedKeyPointsLocations.ptr<short2>(),_cvgmNewlyAddedKeyPointsResponse.ptr<float>(),uNewlyAdded);
	uNewlyAdded = std::min( uNewlyAdded, _uTotalParticles - uMatched );
	//g) collect keypoints
	_cvgmParticleResponsesCurr.setTo(0);
	/*btl::device::semidense::cudaCollectParticles(_cvgmMatchedKeyPointsLocations.ptr<short2>(),_cvgmMatchedKeyPointsResponse.ptr<float>(),uMatched,
		&_cvgmParticleResponsesCurr,&_cvgmParticleResponsesCurr,_cvgmBlurredCurr);
	btl::device::semidense::cudaCollectParticles(_cvgmNewlyAddedKeyPointsLocations.ptr<short2>(),_cvgmNewlyAddedKeyPointsResponse.ptr<float>(),uNewlyAdded,&_cvgmParticleResponsesCurr);
	*/
	btl::device::semidense::cudaCollectParticles(_cvgmMatchedKeyPointsLocations.ptr<short2>(),_cvgmMatchedKeyPointsResponse.ptr<float>(),uMatched,
		&_cvgmParticleResponsesCurr,NULL);
	btl::device::semidense::cudaCollectParticles(_cvgmNewlyAddedKeyPointsLocations.ptr<short2>(),_cvgmNewlyAddedKeyPointsResponse.ptr<float>(),uNewlyAdded,
		&_cvgmParticleResponsesCurr,&_cvgmParticleDescriptorsCurr,_cvgmBlurredCurr);
	
	//h) assign the current frame to previous frame
	_cvgmParticleResponsesCurr.copyTo(_cvgmParticleResponsesPrev);
	_cvgmParticlesAgeCurr.copyTo(_cvgmParticlesAgePrev);
	_cvgmParticlesVelocityCurr.copyTo(_cvgmParticlesVelocityPrev);
	_cvgmBlurredCurr.copyTo(_cvgmBlurredPrev);
	_cvgmParticleDescriptorsCurr.copyTo(_cvgmParticleDescriptorsPrev);

	//convert saliency score to 3-channel RGB image
	/*_cvgmParticleResponsesPrev.convertTo(_cvgmBufferC1,CV_8UC1,255);
	cv::gpu::cvtColor(_cvgmBufferC1,_cvgmColorFrame,CV_GRAY2RGB);
	_cvgmColorFrame.download(cvmColorFrame_);*/
	//render keypoints
	_cvgmMatchedKeyPointsLocations.download(_cvmKeyPointsLocations);
	for (unsigned int i=0;i<_uTotalParticles; i++){
		short2 s2Loc = _cvmKeyPointsLocations.ptr<short2>()[i];
		cv::circle(cvmColorFrame_,cv::Point(s2Loc.x,s2Loc.y),2,cv::Scalar(0,0,255.));
	}
}













