#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/legacy/legacy.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>

#include <Eigen/Core>

#include "SemiDenseTracker.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "Helper.hpp"
//#include "OtherUtil.hpp"
#include "TestCudaFast.h"
#include "TrackerSimpleFreak.h"
#include "FullFrameAlignment.cuh"

using namespace btl::image::semidense;

bool sort_pred ( const cv::DMatch& m1_, const cv::DMatch& m2_ )
{
	return m1_.distance < m2_.distance;
}

btl::image::CTrackerSimpleFreak::CTrackerSimpleFreak(unsigned int uPyrHeight_)
	:CSemiDenseTracker(uPyrHeight_)
{
	_usTotal = 300;
}

bool btl::image::CTrackerSimpleFreak::initialize( boost::shared_ptr<cv::Mat> _acvmShrPtrPyrBW[4] )
{
	//allocate surf
	_pSurf.reset(new cv::SURF(100,4,2,false,true) );
	//allocate freak
	_pFreak.reset(new cv::FREAK()); 
	//detect key points
	(*_pSurf)(*_acvmShrPtrPyrBW[0], cv::Mat(), _vKeypointsPrev);
	//extract freak
	_pFreak->compute( *_acvmShrPtrPyrBW[0], _vKeypointsPrev, _cvmDescriptorPrev );
	
	return true;
}


void btl::image::CTrackerSimpleFreak::track(boost::shared_ptr<cv::Mat> _acvmShrPtrPyrBW[4] )
{
	//detect key points
	_vKeypointsCurr.clear();
	(*_pSurf)(*_acvmShrPtrPyrBW[0], cv::Mat(), _vKeypointsCurr);
	
	//extract freak
	_pFreak->compute( *_acvmShrPtrPyrBW[0], _vKeypointsCurr, _cvmDescriptorCurr );
	
	//match
	_cMatcher.match(_cvmDescriptorPrev, _cvmDescriptorCurr, _vMatches);  
	//sort
	std::sort (_vMatches.begin(), _vMatches.end(), sort_pred);
	//backup previous
	_cvmDescriptorPrev.copyTo(_cvmDescriptor1);
	_vKeypoints1.resize(_vKeypointsPrev.size());
	std::copy( _vKeypointsPrev.begin(), _vKeypointsPrev.end(), _vKeypoints1.begin() );
	//copy current to previous
	_cvmDescriptorCurr.copyTo(_cvmDescriptorPrev);
	_vKeypointsPrev.resize(_vKeypointsCurr.size());
	std::copy ( _vKeypointsCurr.begin(), _vKeypointsCurr.end(), _vKeypointsPrev.begin() );
	return;	
}


bool btl::image::CTrackerSimpleFreak::initialize( boost::shared_ptr<cv::gpu::GpuMat> acgvmShrPtrPyrBW_[4], const cv::Mat& cvmMaskCurr_ )
{
	for (unsigned int n=0; n <_uPyrHeight; n++)	{
		_acgvmShrPtrPyrBWPrev[0] = acgvmShrPtrPyrBW_[0];
	}
	cvmMaskCurr_.copyTo( _cvmMaskPrev);
	return true;
}

void btl::image::CTrackerSimpleFreak::track( const Eigen::Matrix3f& eimHomoInit_,
											 boost::shared_ptr<cv::gpu::GpuMat> acvgmShrPtrPyrBW_[4],
											 const cv::Mat& cvmMaskCurr_,
											 Eigen::Matrix3f* peimHomo_ ){
												 
	//from coarse to fine
	*peimHomo_ =  eimHomoInit_;
	cv::gpu::GpuMat cvgmMaskPrev; cvgmMaskPrev.upload(_cvmMaskPrev);
	cv::gpu::GpuMat cvgmMaskCurr; cvgmMaskCurr.upload(cvmMaskCurr_);
	cv::gpu::GpuMat cvgmBuffer;
	for (unsigned int uLevel = _uPyrHeight-1; uLevel >= -1; uLevel--){
		for (int n = 0; n < 5; n++){ //iterations
			btl::device::cudaFullFrame( *_acgvmShrPtrPyrBWPrev[uLevel], cvgmMaskPrev, *acvgmShrPtrPyrBW_[uLevel], cvgmMaskCurr, peimHomo_->data(), &cvgmBuffer);
			//solve out the delta homography
			Eigen::Matrix3f eimDeltaHomo; eimDeltaHomo.setZero();
			extractHomography(cvgmBuffer,&eimDeltaHomo);
			*peimHomo_ += eimDeltaHomo; 
		}//iterations
	}//for pyramid
	return;
}

void btl::image::CTrackerSimpleFreak::extractHomography(const cv::gpu::GpuMat& cvgmBuffer_,Eigen::Matrix3f* peimDeltaHomo_){

}

void btl::image::CTrackerSimpleFreak::displayCandidates( cv::Mat& cvmColorFrame_ ){
	
	return;
}
cv::Mat btl::image::CTrackerSimpleFreak::calcHomography(const cv::Mat& cvmMaskCurr_, const cv::Mat& cvmMaskPrev_) {
	if(_vMatches.size()<=4){
		std::cout << "Not KeyPoint detected";
		return cv::Mat();
	}
	unsigned int uTotal = std::min( unsigned int(_vMatches.size()), unsigned int(_usTotal) );
	cv::Mat cvmKeyPointPrev; cvmKeyPointPrev.create( 1, (int)uTotal, CV_32FC2 );
	cv::Mat cvmKeyPointCurr; cvmKeyPointCurr.create( 1, (int)uTotal, CV_32FC2 );
	for (int i=0;i<cvmKeyPointCurr.cols; i++){
		int nCurrIdx = _vMatches[i].trainIdx;
		int nPrevIdx = _vMatches[i].queryIdx;

		cvmKeyPointCurr.ptr<float2>()[i].x = _vKeypointsCurr[nCurrIdx].pt.x;
		cvmKeyPointCurr.ptr<float2>()[i].y = _vKeypointsCurr[nCurrIdx].pt.y;
		cvmKeyPointPrev.ptr<float2>()[i].x = _vKeypoints1[nPrevIdx].pt.x;
		cvmKeyPointPrev.ptr<float2>()[i].y = _vKeypoints1[nPrevIdx].pt.y;
	}
	//calc Homography
	return cv::findHomography(cvmKeyPointPrev,cvmKeyPointCurr,CV_RANSAC,1);
}

void btl::image::CTrackerSimpleFreak::display(cv::Mat& cvmColorFrame_) {
	cvmColorFrame_.setTo(cv::Scalar::all(255));
	unsigned int uTotal = std::min( unsigned int(_vMatches.size()), unsigned int(_usTotal) );
	for (unsigned int i=0;i<uTotal; i+=1){
		int nCurrIdx = _vMatches[i].trainIdx;
		int nPrevIdx = _vMatches[i].queryIdx;
		
		if( _vKeypoints1[nPrevIdx].octave == 0)	cv::circle(cvmColorFrame_,_vKeypoints1[nPrevIdx].pt,1,cv::Scalar(0,0,255.));
		if( _vKeypoints1[nPrevIdx].octave == 1)	cv::circle(cvmColorFrame_,_vKeypoints1[nPrevIdx].pt,1,cv::Scalar(0,255.,0));
		if( _vKeypoints1[nPrevIdx].octave == 2)	cv::circle(cvmColorFrame_,_vKeypoints1[nPrevIdx].pt,1,cv::Scalar(255.,0,0));
		if( _vKeypoints1[nPrevIdx].octave == 3)	cv::circle(cvmColorFrame_,_vKeypoints1[nPrevIdx].pt,1,cv::Scalar(255.,0,255));

		if( _vKeypointsCurr[nCurrIdx].octave == 0) cv::line(cvmColorFrame_, _vKeypoints1[nPrevIdx].pt, _vKeypointsCurr[nCurrIdx].pt, cv::Scalar(0,0,255));
		if( _vKeypointsCurr[nCurrIdx].octave == 1) cv::line(cvmColorFrame_, _vKeypoints1[nPrevIdx].pt, _vKeypointsCurr[nCurrIdx].pt, cv::Scalar(0,255,0));
		if( _vKeypointsCurr[nCurrIdx].octave == 2) cv::line(cvmColorFrame_, _vKeypoints1[nPrevIdx].pt, _vKeypointsCurr[nCurrIdx].pt, cv::Scalar(255,0,0));
		if( _vKeypointsCurr[nCurrIdx].octave == 3) cv::line(cvmColorFrame_, _vKeypoints1[nPrevIdx].pt, _vKeypointsCurr[nCurrIdx].pt, cv::Scalar(255,0,255));
	}
	return;
}












