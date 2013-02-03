/**
* @file VideoSourceKinect.cpp
* @brief load of data from a kinect device 
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.0
* @date 2011-02-23
*/
#define INFO
//gl
#include <gl/glew.h>
#include <gl/freeglut.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
//opencv
#include <opencv2/gpu/gpu.hpp>
//boost
#include <boost/scoped_ptr.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
//eigen
#include <Eigen/Core>
//openni
#include <XnCppWrapper.h>
//self
#include "Camera.h"
#include "Utility.hpp"
#include "Kinect.h"
#include "GLUtil.h"
#include "PlaneObj.h"
#include "Histogram.h"
#include "SemiDenseTracker.h"
#include "SemiDenseTrackerOrb.h"
#include "KeyFrame.h"
#include "CyclicBuffer.h"
#include "VideoSourceKinect.hpp"
#include "VideoSourceKinectSimulator.hpp"
#include "cuda/CudaLib.h"

#include <iostream>
#include <string>
#include <limits>

using namespace btl::utility;

namespace btl{ namespace kinect
{


VideoSourceKinectSimulator::VideoSourceKinectSimulator (ushort uResolution_, ushort uPyrHeight_, bool bUseNIRegistration_,const Eigen::Vector3f& eivCw_)
:VideoSourceKinect(uResolution_, uPyrHeight_, bUseNIRegistration_,eivCw_){
	_fNear = 0.3f;
	_fFar = 4.f;
}
VideoSourceKinectSimulator::~VideoSourceKinectSimulator()
{
}

void VideoSourceKinectSimulator::exportRawDepth() const{
	cv::FileStorage cFSWrite( "raw_depth.yml", cv::FileStorage::WRITE );
	cFSWrite << "RawDepth" << _cvmDepth;
	cFSWrite.release();
}
void VideoSourceKinectSimulator::captureScreen(){
	//view port # 1.
	glReadBuffer(GL_FRONT);
	glReadPixels(0, __aKinectH[_uResolution], __aKinectW[_uResolution], __aKinectH[_uResolution], GL_RGB,	   GL_UNSIGNED_BYTE, _cvmUndistRGB.data);
	glDrawBuffer(GL_BACK);
	_cvgmUndistRGB.upload(_cvmUndistRGB);
	cv::gpu::cvtColor(_cvgmUndistRGB,_cvgmUndistRGB,CV_RGB2BGR);
	btl::device::cudaConvertGL2CV(_cvgmUndistRGB,&_cvgmRGB);
	_cvgmRGB.download(_cvmUndistRGB);
	cv::imwrite(std::string("1.bmp"),_cvmUndistRGB);
	//view port # 4.
	glReadBuffer(GL_FRONT);
	glReadPixels(__aKinectW[_uResolution],0, __aKinectW[_uResolution], __aKinectH[_uResolution], GL_RGB,	   GL_UNSIGNED_BYTE, _cvmUndistRGB.data);
	glDrawBuffer(GL_BACK);
	_cvgmUndistRGB.upload(_cvmUndistRGB);
	cv::gpu::cvtColor(_cvgmUndistRGB,_cvgmUndistRGB,CV_RGB2BGR);
	btl::device::cudaConvertGL2CV(_cvgmUndistRGB,&_cvgmRGB);
	_cvgmRGB.download(_cvmUndistRGB);
	cv::imwrite(std::string("2.bmp"),_cvmUndistRGB);
}
void VideoSourceKinectSimulator::getNextFrame(tp_frame eFrameType_, int* pnRecordingStatus_)
{
	// must be called from the main display call back function of the glut 
	//_pFrame->assignRTfromGL();
	//capture color and depth
	glReadBuffer(GL_FRONT);
	glReadPixels(0, __aKinectH[_uResolution], __aKinectW[_uResolution], __aKinectH[_uResolution], GL_DEPTH_COMPONENT, GL_FLOAT, _cvmDepth.data);
	glReadPixels(0, __aKinectH[_uResolution], __aKinectW[_uResolution], __aKinectH[_uResolution], GL_RGB, GL_UNSIGNED_BYTE, _cvmRGB.data);
	//glReadPixels(0, 0, screenWidth, screenHeight, GL_DEPTH_COMPONENT24, GL_UNSIGNED_INT, _pZBuffer); //doesnt work
	// render to the framebuffer //////////////////////////
	glDrawBuffer(GL_BACK);

	_cvgmDepth.upload(_cvmDepth);
	_cvgmRGB.upload(_cvmRGB);

	Eigen::Matrix4d eimProj;
	glGetDoublev(GL_PROJECTION_MATRIX,eimProj.data());
	PRINT(eimProj);

	btl::device::cudaConvertZValue2Depth(_cvgmDepth,_fNear,_fFar,&_cvgmUndistDepth);
	//processZBuffer(_cvmDepth, &*_pFrame->_acvmShrPtrPyrBWs[0]);
	//float fMax1, fMin1;
	//btl::utility::findMatMaxMin(_cvmDepth,&fMax1,&fMin1);
	//PRINT(fMin1);
	//PRINT(fMax1);
	//_cvgmUndistDepth.download(_cvmUndistDepth);
	//float fMax2, fMin2;
	//btl::utility::findMatMaxMin(_cvmUndistDepth,&fMax2,&fMin2);
	//PRINT(fMin2);
	//PRINT(fMax2);

	//bilateral filtering (comments off the following three lines to get raw depth map image of kinect)
	btl::device::cudaDepth2Disparity2(_cvgmUndistDepth,_fCutOffDistance, &*_pFrame->_acvgmShrPtrPyr32FC1Tmp[0]);//convert depth from mm to m
	btl::device::cudaBilateralFiltering(*_pFrame->_acvgmShrPtrPyr32FC1Tmp[0],_fSigmaSpace,_fSigmaDisparity,&*_pFrame->_acvgmShrPtrPyrDisparity[0]);
	btl::device::cudaDisparity2Depth(*_pFrame->_acvgmShrPtrPyrDisparity[0],&*_pFrame->_acvgmShrPtrPyrDepths[0]);
	//get pts and nls
	btl::device::unprojectRGBCVm(*_pFrame->_acvgmShrPtrPyrDepths[0],_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v, 0,&*_pFrame->_acvgmShrPtrPyrPts[0] );
	btl::device::cudaFastNormalEstimation(*_pFrame->_acvgmShrPtrPyrPts[0],&*_pFrame->_acvgmShrPtrPyrNls[0]);//_vcvgmPyrNls[0]);
	//generate black and white
	btl::device::cudaConvertGL2CV(_cvgmRGB,&*_pFrame->_acvgmShrPtrPyrRGBs[0]);
	cv::gpu::cvtColor(*_pFrame->_acvgmShrPtrPyrRGBs[0],*_pFrame->_acvgmShrPtrPyrBWs[0],cv::COLOR_RGB2GRAY);

	//down-sampling
	for( unsigned int i=1; i<_uPyrHeight; i++ )	{
		_pFrame->_acvgmShrPtrPyrRGBs[i]->setTo(0);
		cv::gpu::pyrDown(*_pFrame->_acvgmShrPtrPyrRGBs[i-1],*_pFrame->_acvgmShrPtrPyrRGBs[i]);
		cv::gpu::cvtColor(*_pFrame->_acvgmShrPtrPyrRGBs[i],*_pFrame->_acvgmShrPtrPyrBWs[i],cv::COLOR_RGB2GRAY);
		_pFrame->_acvgmShrPtrPyr32FC1Tmp[i]->setTo(std::numeric_limits<float>::quiet_NaN());
		btl::device::cudaPyrDown( *_pFrame->_acvgmShrPtrPyrDisparity[i-1],_fSigmaDisparity,&*_pFrame->_acvgmShrPtrPyr32FC1Tmp[i]);
		btl::device::cudaBilateralFiltering(*_pFrame->_acvgmShrPtrPyr32FC1Tmp[i],_fSigmaSpace,_fSigmaDisparity,&*_pFrame->_acvgmShrPtrPyrDisparity[i]);
		btl::device::cudaDisparity2Depth(*_pFrame->_acvgmShrPtrPyrDisparity[i],&*_pFrame->_acvgmShrPtrPyrDepths[i]);
		btl::device::unprojectRGBCVm(*_pFrame->_acvgmShrPtrPyrDepths[i],_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v, i,&*_pFrame->_acvgmShrPtrPyrPts[i] );
		btl::device::cudaFastNormalEstimation(*_pFrame->_acvgmShrPtrPyrPts[i],&*_pFrame->_acvgmShrPtrPyrNls[i]);
	}	

	for( unsigned int i=0; i<_uPyrHeight; i++ )	{
		_pFrame->_acvgmShrPtrPyrRGBs[i]->download(*_pFrame->_acvmShrPtrPyrRGBs[i]);
		_pFrame->_acvgmShrPtrPyrBWs[i]->download(*_pFrame->_acvmShrPtrPyrBWs[i]);
		_pFrame->_acvgmShrPtrPyrPts[i]->download(*_pFrame->_acvmShrPtrPyrPts[i]);
		_pFrame->_acvgmShrPtrPyrNls[i]->download(*_pFrame->_acvmShrPtrPyrNls[i]);	
	}
#if !USE_PBO
#endif
	//scale the depth map
	{
		btl::device::scaleDepthCVmCVm(0,_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v,&*_pFrame->_acvgmShrPtrPyrDepths[0]);
	}
	return;
}


void VideoSourceKinectSimulator::processZBuffer(const cv::Mat& cvmDepth_, cv::Mat* pcvmDepthImg_ ) const
{
	float fMax, fMin;
	btl::utility::findMatMaxMin(cvmDepth_,&fMax,&fMin);
	
	float fScale = fMax - fMin;

	const float* pSrc = (const float*)cvmDepth_.data;
	uchar* pDImg = pcvmDepthImg_->data;
	for(int i = 0; i < cvmDepth_.rows; ++i)
	{
		for(int j = 0; j < cvmDepth_.cols; ++j)
		{
			float tmp = ((*pSrc)-fMin)/fScale*255;
			*pDImg = (GLubyte) tmp;
			pDImg++; pSrc++;
		}
	}
	return;
}

void VideoSourceKinectSimulator::setSensorDepthRange() const{
	_pRGBCamera->setGLProjectionMatrix(1,_fNear,_fFar);
}
} //namespace kinect
} //namespace btl
