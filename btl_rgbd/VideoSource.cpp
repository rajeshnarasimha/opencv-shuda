
#include <string>
#include <vector>

#include <gl/freeglut.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
//boost
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <Eigen/Core>

#include "OtherUtil.hpp"
#include "Converters.hpp"
#include "EigenUtil.hpp"
#include "Camera.h"
#include "Kinect.h"
#include "GLUtil.h"
#include "PlaneObj.h"
#include "Histogram.h"
#include "SemiDenseTracker.h"
#include "SemiDenseTrackerOrb.h"
#include "KeyFrame.h"
#include "VideoSource.h"

namespace btl{ namespace video
{



VideoSource::VideoSource(const std::string& strCameraParam_, ushort uResolution_, ushort uPyrHeight_,const Eigen::Vector3f& eivCw_ )
{
	_nMode = SIMPLE_CAPTURING;
	_strCameraParam = strCameraParam_;
	_uResolution = uResolution_;
	_uPyrHeight = uPyrHeight_;
	_uFrameIdx = 0;
	_eivCw = eivCw_;
	_fScale = 1.f;
}

VideoSource::~VideoSource()
{

}


void VideoSource::init()
{
	_uFrameIdx = -1;
	_pVideo.reset( new cv::VideoCapture(_strVideoFileName) ); 
	//import camera parameters
	_pCamera.reset(new btl::image::SCamera(_strCameraParam,0));
	//allocate
	_pCurrFrame.reset( new btl::kinect::CKeyFrame( _pCamera.get(),_uResolution,_uPyrHeight, _eivCw ) );
}

void VideoSource::getNextFrame(int* pnStatus_)
{
	_uFrameIdx++;
	//capture a new frame
	_pVideo->read(_cvmRGB); //load into cpu buffer
	//ensure the capture image is valid
	if (_cvmRGB.empty()){
		_pVideo->set(CV_CAP_PROP_POS_AVI_RATIO,0);//replay at the end of the video
		_pVideo->read(_cvmRGB);
		_uFrameIdx = 0;
	}
	//resize
	cv::resize(_cvmRGB,*_pCurrFrame->_acvmShrPtrPyrRGBs[0],cv::Size(0,0),_fScale ,_fScale );	

	_pCurrFrame->_acvgmShrPtrPyrRGBs[0]->upload(*_pCurrFrame->_acvmShrPtrPyrRGBs[0]); //load into gpu buffer
	cv::gpu::cvtColor(*_pCurrFrame->_acvgmShrPtrPyrRGBs[0],_cvgmRGB,CV_BGR2RGB); 
	_pCurrFrame->_acvgmShrPtrPyrRGBs[0]->setTo(cv::Scalar::all(0));//clear(RGB)
	//undistort the frame
	cv::gpu::remap(_cvgmRGB, *_pCurrFrame->_acvgmShrPtrPyrRGBs[0], _pCamera->_cvgmMapX, _pCamera->_cvgmMapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT  ); //undistort image
	cv::gpu::cvtColor(*_pCurrFrame->_acvgmShrPtrPyrRGBs[0],*_pCurrFrame->_acvgmShrPtrPyrBWs[0],CV_RGB2GRAY); //convert to gray image
	_pCurrFrame->_acvgmShrPtrPyrRGBs[0]->download(*_pCurrFrame->_acvmShrPtrPyrRGBs[0]);//from bgr to rgb
	_pCurrFrame->_acvgmShrPtrPyrBWs[0]->download(*_pCurrFrame->_acvmShrPtrPyrBWs[0]);

	gpuBuildPyramid();

	return;
}

void VideoSource::gpuBuildPyramid(){
	for (unsigned int n=1; n< _uPyrHeight; n++){
		cv::gpu::resize(*_pCurrFrame->_acvgmShrPtrPyrBWs[n-1],*_pCurrFrame->_acvgmShrPtrPyrBWs[n],cv::Size(0,0),.5f,.5f );	
		cv::gpu::resize(*_pCurrFrame->_acvgmShrPtrPyrRGBs[n-1],*_pCurrFrame->_acvgmShrPtrPyrRGBs[n],cv::Size(0,0),.5f,.5f );
		_pCurrFrame->_acvgmShrPtrPyrRGBs[n]->download(*_pCurrFrame->_acvmShrPtrPyrRGBs[n]);
		_pCurrFrame->_acvgmShrPtrPyrBWs[n]->download(*_pCurrFrame->_acvmShrPtrPyrBWs[n]);
	}
	return;
}


}//namespace video
}//namespace btl