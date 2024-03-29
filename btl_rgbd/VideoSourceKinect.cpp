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
#include <boost/shared_ptr.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
//eigen
#include <Eigen/Core>
//openni
#include <XnCppWrapper.h>
#include "CyclicBuffer.h"
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
#include "VideoSourceKinect.hpp"
#include "cuda/CudaLib.h"

#include <iostream>
#include <string>
#include <limits>

#ifndef CHECK_RC_
#define CHECK_RC_(rc, what)	\
	BTL_ASSERT(rc == XN_STATUS_OK, (what + std::string(xnGetStatusString(rc))) )
#endif

using namespace btl::utility;

namespace btl{ namespace kinect
{

bool VideoSourceKinect::_bIsSequenceEnds = false;
VideoSourceKinect::VideoSourceKinect (ushort uResolution_, ushort uPyrHeight_, bool bUseNIRegistration_,const Eigen::Vector3f& eivCw_/*float fCwX_, float fCwY_, float fCwZ_*/ )
:_bUseNIRegistration(bUseNIRegistration_),_uResolution(uResolution_),_uPyrHeight(uPyrHeight_)
{
	/*boost::posix_time::ptime _cT0, _cT1;
	boost::posix_time::time_duration _cTDAll;
	_cT0 =  boost::posix_time::microsec_clock::local_time(); 
	_cT1 =  boost::posix_time::microsec_clock::local_time(); 
	_cTDAll = _cT1 - _cT0 ;
	PRINT( _cTDAll.total_milliseconds() );*/
	_nMode = SIMPLE_CAPTURING;
	//allocate

	PRINTSTR("Allocate buffers...")
	_cvmRGB			   .create( __aKinectH[_uResolution], __aKinectW[_uResolution], CV_8UC3 );
	_cvmUndistRGB	   .create( __aKinectH[_uResolution], __aKinectW[_uResolution], CV_8UC3 );
	_cvmDepth		   .create( __aKinectH[_uResolution], __aKinectW[_uResolution], CV_32FC1);
	_cvmUndistDepth	   .create( __aKinectH[_uResolution], __aKinectW[_uResolution], CV_32FC1);
	//_cvmAlignedRawDepth.create( KINECT_HEIGHT, KINECT_WIDTH, CV_32FC1);

	//_cvmIRWorld .create(KINECT_HEIGHT,KINECT_WIDTH,CV_32FC3);
	//_cvmRGBWorld.create(KINECT_HEIGHT,KINECT_WIDTH,CV_32FC3);

	// allocate memory for later use ( registrate the depth with rgb image
	// refreshed for every frame
	// pre-allocate cvgm to increase the speed
	_cvgmIRWorld        .create(__aKinectH[_uResolution], __aKinectW[_uResolution],CV_32FC3);
	_cvgmRGBWorld       .create(__aKinectH[_uResolution], __aKinectW[_uResolution],CV_32FC3);
	_cvgmAlignedRawDepth.create(__aKinectH[_uResolution], __aKinectW[_uResolution],CV_32FC1);
	_cvgm32FC1Tmp       .create(__aKinectH[_uResolution], __aKinectW[_uResolution],CV_32FC1);
	_cvgmUndistDepth    .create(__aKinectH[_uResolution], __aKinectW[_uResolution],CV_32FC1);

	//import camera parameters
	_pRGBCamera.reset(new btl::image::SCamera("XtionRGB.yml"/*btl::kinect::SCamera::CAMERA_RGB*/,_uResolution));
	_pIRCamera .reset(new btl::image::SCamera("XtionIR.yml"/*btl::kinect::SCamera::CAMERA_IR*/, _uResolution));

	importYML();
	_pCurrFrame.reset(new CKeyFrame(_pRGBCamera.get(),_uResolution,_uPyrHeight,eivCw_));

	//other
	//definition of parameters
	_fThresholdDepthInMeter = 0.01f;
	_fSigmaSpace = 1.5;
	_fSigmaDisparity = 1.f/.6f - 1.f/(.6f+_fThresholdDepthInMeter);

	//default centroid follows opencv-convention
	_dXCentroid = _dYCentroid = 0;
	_dZCentroid = 1.0;
	//initialize normal histogram
	//btl::kinect::CKeyFrame::_sNormalHist.init(2);
	//btl::kinect::CKeyFrame::_sDistanceHist.init(30);
	btl::kinect::CKeyFrame::_pSurf.reset(new cv::gpu::SURF_GPU(100));
	btl::kinect::CKeyFrame::_pOrb.reset(new cv::gpu::ORB_GPU(500,2.f,3));
	btl::kinect::CKeyFrame::_pBroxOpticalFlow.reset(new cv::gpu::BroxOpticalFlow(80,100,0.5,3,10,5));

	//
	btl::kinect::CKeyFrame::_pcvgmPrev.reset(new cv::gpu::GpuMat(btl::kinect::__aKinectH[_uResolution],btl::kinect::__aKinectW[_uResolution],CV_32FC1));
	btl::kinect::CKeyFrame::_pcvgmCurr.reset(new cv::gpu::GpuMat(btl::kinect::__aKinectH[_uResolution],btl::kinect::__aKinectW[_uResolution],CV_32FC1));
	btl::kinect::CKeyFrame::_pcvgmU.reset(new cv::gpu::GpuMat(btl::kinect::__aKinectH[_uResolution],btl::kinect::__aKinectW[_uResolution],CV_32FC1));
	btl::kinect::CKeyFrame::_pcvgmV.reset(new cv::gpu::GpuMat(btl::kinect::__aKinectH[_uResolution],btl::kinect::__aKinectW[_uResolution],CV_32FC1));

	extern cv::gpu::GpuMat cvgmTest,cvgmTmp;
	cvgmTest.create(btl::kinect::__aKinectH[_uResolution],btl::kinect::__aKinectW[_uResolution],CV_32FC1);
	cvgmTmp.create(btl::kinect::__aKinectH[_uResolution],btl::kinect::__aKinectW[_uResolution],CV_32FC1);

	_bIsSequenceEnds = false;

	_fCutOffDistance = 3.f;
	std::cout << " Done. " << std::endl;
}
VideoSourceKinect::~VideoSourceKinect()
{
	_cImgGen.Release();
	_cDepthGen.Release();
    _cContext.Release();
	if (_nMode == PLAYING_BACK){
		_cPlayer.UnregisterFromEndOfFileReached(_handle);
		_cPlayer.Release();
	}
}

void VideoSourceKinect::initKinect()
{
	_nMode = SIMPLE_CAPTURING;
	PRINTSTR("Initialize RGBD camera...");
	XnStatus nRetVal = XN_STATUS_OK;
	//_cContext inizialization 
	nRetVal = _cContext.Init();						CHECK_RC_(nRetVal, "Initialize _cContext"); 
	
	setResolution(_uResolution);
	//RGB node creation 
	nRetVal = _cImgGen.Create(_cContext);			CHECK_RC_(nRetVal, "Create rgb generator fail"); 
	//IR node creation 
	nRetVal = _cDepthGen.Create(_cContext);			CHECK_RC_(nRetVal, "Create depth generator"); 
	_cContext.StopGeneratingAll();
	// set as the highest resolution 0 for 480x640 
	//register the depth generator with the image generator
	if (_bUseNIRegistration){
		nRetVal = _cDepthGen.GetAlternativeViewPointCap().SetViewPoint ( _cImgGen );	CHECK_RC_ ( nRetVal, "Getting and setting AlternativeViewPoint failed: " ); 
		// Set Hole Filter
		_cDepthGen.SetIntProperty("HoleFilter", TRUE);
		// Frame Sync
		if ( true && _cImgGen.IsCapabilitySupported(XN_CAPABILITY_FRAME_SYNC)){
			if (_cImgGen.GetFrameSyncCap().CanFrameSyncWith(_cImgGen)){
				nRetVal = _cImgGen.GetFrameSyncCap().FrameSyncWith(_cDepthGen);			CHECK_RC_(nRetVal, "Frame sync");
			}//if (_cDepthGen.GetFrameSyncCap().CanFrameSyncWith(_cImgGen))
		}//if ( _cDepthGen.IsCapabilitySupported(XN_CAPABILITY_FRAME_SYNC))
	}//if (_bUseNIRegistration)
	nRetVal = _cImgGen.SetMapOutputMode(_sModeVGA); 	CHECK_RC_(nRetVal, "Depth SetMapOutputMode XRes for 240, YRes for 320 and FPS for 30"); 
	nRetVal = _cDepthGen.SetMapOutputMode(_sModeVGA);	CHECK_RC_(nRetVal, "Depth SetMapOutputMode XRes for 640, YRes for 480 and FPS for 30"); 
	//nRetVal = _cDepthGen.StartGenerating();			CHECK_RC_(nRetVal, "Start generating Depth fail");
	nRetVal = _cContext.StartGeneratingAll();			CHECK_RC_(nRetVal, "Start generating data: " );
	PRINTSTR(" Done.");

	return;
}
void VideoSourceKinect::initRecorder(std::string& strPath_, ushort nTimeInSecond_){
	initKinect();
	_nMode = RECORDING;
	PRINTSTR("Initialize RGBD data recorder...");
	_pCyclicBuffer.reset(new CCyclicBuffer(_cContext,_cDepthGen,_cImgGen));
	_pCyclicBuffer->Initialize(".", nTimeInSecond_);//strPath_.c_str()

	_nLastDepthTime = 0;
	_nLastImageTime = 0;
	_nMissedDepthFrames = 0;
	_nMissedImageFrames = 0;
	_nDepthFrames = 0;
	_nImageFrames = 0;
	PRINTSTR(" Done.");
}
void VideoSourceKinect::initPlayer(std::string& strPathFileName_, bool bRepeat_){
	_nMode = PLAYING_BACK;
	PRINTSTR("Initialize Player recorder...");
	//when you need to replay the file, call this function
	_bIsSequenceEnds = false;
	XnStatus nRetVal = _cContext.Init(); CHECK_RC_(nRetVal, "Initialize _cContext");
	nRetVal = _cContext.OpenFileRecording(strPathFileName_.c_str(), _cPlayer ); CHECK_RC_(nRetVal, "Open oni file");
	_cPlayer.SetRepeat(bRepeat_);
	_cPlayer.RegisterToEndOfFileReached(VideoSourceKinect::playEndCallback, NULL, _handle);
	//locate relevant production node from the context
	xn::NodeInfoList list;
	nRetVal = _cContext.EnumerateExistingNodes(list);
	if (nRetVal == XN_STATUS_OK)
	{
		for (xn::NodeInfoList::Iterator it = list.Begin(); it != list.End(); ++it)
		{
			switch ((*it).GetDescription().Type)
			{
			case XN_NODE_TYPE_DEPTH:
				(*it).GetInstance(_cDepthGen);
				break;
			case XN_NODE_TYPE_IMAGE:
				(*it).GetInstance(_cImgGen);
				break;
			case XN_NODE_TYPE_PLAYER:
				(*it).GetInstance(_cPlayer);
			}
		}
	}//if (nRetVal == XN_STATUS_OK)

	PRINTSTR(" Done.");
	return;
}//initPlayer()

void VideoSourceKinect::importYML()
{
	//_pRGBCamera->importYML();
	//_pIRCamera->importYML();

	// create and open a character archive for output
#if __linux__
	cv::FileStorage cFSRead( "/space/csxsl/src/opencv-shuda/Data/kinect_intrinsics.yml", cv::FileStorage::READ );
#else if _WIN32 || _WIN64
	cv::FileStorage cFSRead ( "C:\\csxsl\\src\\opencv-shuda\\Data\\xtion_intrinsics.yml", cv::FileStorage::READ );
#endif
	cv::Mat cvmRelativeRotation,cvmRelativeTranslation;
	cFSRead ["cvmRelativeRotation"] >> cvmRelativeRotation;
	cFSRead ["cvmRelativeTranslation"] >> cvmRelativeTranslation;
	cFSRead.release();

	//prepare camera parameters
	cv::Mat  mRTrans = cvmRelativeRotation.t();
	cv::Mat vRT = mRTrans * cvmRelativeTranslation;

	_aR[0] = (float)mRTrans.at<double> ( 0, 0 );
	_aR[1] = (float)mRTrans.at<double> ( 0, 1 );
	_aR[2] = (float)mRTrans.at<double> ( 0, 2 );
	_aR[3] = (float)mRTrans.at<double> ( 1, 0 );
	_aR[4] = (float)mRTrans.at<double> ( 1, 1 );
	_aR[5] = (float)mRTrans.at<double> ( 1, 2 );
	_aR[6] = (float)mRTrans.at<double> ( 2, 0 );
	_aR[7] = (float)mRTrans.at<double> ( 2, 1 );
	_aR[8] = (float)mRTrans.at<double> ( 2, 2 );

	_aRT[0] = (float)vRT.at<double> ( 0 );
	_aRT[1] = (float)vRT.at<double> ( 1 );
	_aRT[2] = (float)vRT.at<double> ( 2 );
}

void VideoSourceKinect::findRange(const cv::gpu::GpuMat& cvgmMat_)
{
	cv::Mat cvmMat;
	cvgmMat_.download(cvmMat);
	findRange(cvmMat);
}
void VideoSourceKinect::findRange(const cv::Mat& cvmMat_)
{
	//BTL_ASSERT(cvmMat_.type()==CV_32F,"findRange() must be CV_32F");
	int N,s;
	switch(cvmMat_.type())
	{
	case CV_32F:
		N = cvmMat_.cols*cvmMat_.rows;
		s = 1;
		break;
	case CV_32FC3:
		N = cvmMat_.cols*cvmMat_.rows*3;
		s = 3;
		break;
	}
	float* pData = (float*) cvmMat_.data;
	float fMin =  1.0e+20f;
	float fMax = -1.0e+20f;
	for( int i=s-1; i< N; i+=s)
	{
		float tmp = *pData ++;
		fMin = fMin > tmp? tmp : fMin;
		fMax = fMax < tmp? tmp : fMax;
	}
	PRINT(fMax);
	PRINT(fMin);
	return;
}

void VideoSourceKinect::getNextFrame(int* pnStatus_){
	float fTimeLeft;
	switch(_nMode){
	case SIMPLE_CAPTURING:
	case PLAYING_BACK:
		getNextFrameNormal(&*pnStatus_);
		break;
	case RECORDING://record the captured sequence from the camera
		getNextFrameRecording(&*pnStatus_,&fTimeLeft);
		break;
	default:
		PRINTSTR("Illigal mode.");
		_nMode = SIMPLE_CAPTURING;
	}
}

void VideoSourceKinect::getNextFrameRecording(int* pnStatus_, float* pfTimeLeft_){
	XnStatus nRetVal = 0;
	switch (*pnStatus_&MASK1)
	{
	case PAUSE://restart
		if (!_cImgMD.WritableRGB24Data()){
			nRetVal = _cContext.WaitAndUpdateAll();	CHECK_RC_ ( nRetVal, "UpdateData failed: " );
		}
		//do nothing
		break;
	case CONTINUE:
		nRetVal = _cContext.WaitAndUpdateAll();	CHECK_RC_ ( nRetVal, "UpdateData failed: " );
		break;
	default:
		PRINTSTR("convert to the status of CONTINUE");
		*pnStatus_ = (*pnStatus_&(~MASK1))|CONTINUE; //set status as continue
		nRetVal = _cContext.WaitAndUpdateAll();	CHECK_RC_ ( nRetVal, "UpdateData failed: " );
		break;
	}
	// these two lines are required for getting a stable image and depth.
	_cImgGen.GetMetaData ( _cImgMD );
	_cDepthGen.GetMetaData( _cDepthMD );

	switch (*pnStatus_&MASK_RECORDER)
	{
	case START_RECORDING://restart
		_pCyclicBuffer->restart();
		_pCyclicBuffer->Update(_cDepthMD, _cImgMD, &*pfTimeLeft_);
		*pnStatus_ = (*pnStatus_&(~MASK_RECORDER))|CONTINUE_RECORDING;
		break;
	case STOP_RECORDING://pause
		*pfTimeLeft_ = _pCyclicBuffer->getTimeLeft();
		break;
		//do nothing
	case CONTINUE_RECORDING://recording continue
		_pCyclicBuffer->Update(_cDepthMD, _cImgMD, &*pfTimeLeft_);
		break;
	case DUMP_RECORDING://dump
		_pCyclicBuffer->Dump(_strDumpFileName);
		*pfTimeLeft_=0;
		_pCyclicBuffer->restart();
		*pnStatus_ = (*pnStatus_&(~MASK_RECORDER))|STOP_RECORDING;
		break;
	default:
		*pnStatus_ = (*pnStatus_&(~MASK_RECORDER))|STOP_RECORDING;
	}
	// Check for missed frames
	//depth
	++_nDepthFrames;
	XnUInt64 nTimestamp = _cDepthGen.GetTimestamp();
	if (_nLastDepthTime != 0 && nTimestamp - _nLastDepthTime > 35000)	{
		int missed = (int)(nTimestamp-_nLastDepthTime)/32000 - 1;
		printf("Missed depth: %llu -> %llu = %d > 35000 - %d frames\n",
			_nLastDepthTime, nTimestamp, XnUInt32(nTimestamp-_nLastDepthTime), missed);
		_nMissedDepthFrames += missed;
	}//if (_nLastDepthTime != 0 && nTimestamp - _nLastDepthTime > 35000)	
	_nLastDepthTime = nTimestamp;
	//image
	++_nImageFrames;
	nTimestamp = _cImgGen.GetTimestamp();
	if (_nLastImageTime != 0 && nTimestamp - _nLastImageTime > 35000)	{
		int missed = (int)(nTimestamp-_nLastImageTime)/32000 - 1;
		printf("Missed image: %llu -> %llu = %d > 35000 - %d frames\n",
			_nLastImageTime, nTimestamp, XnUInt32(nTimestamp-_nLastImageTime), missed);
		_nMissedImageFrames += missed;
	}//if (_nLastImageTime != 0 && nTimestamp - _nLastImageTime > 35000)
	_nLastImageTime = nTimestamp;

	_pCurrFrame->initRT();

	cv::Mat cvmRGB(__aKinectH[_uResolution],__aKinectW[_uResolution],CV_8UC3, (unsigned char*)_cImgMD.WritableRGB24Data());
	cv::Mat cvmDep(__aKinectH[_uResolution],__aKinectW[_uResolution],CV_16UC1,(unsigned short*)_cDepthMD.WritableData());
	cvmRGB.copyTo(_cvmRGB);
	cvmDep.convertTo(_cvmDepth,CV_32FC1);
	//mail capturing function
	if (_bUseNIRegistration)
		gpuBuildPyramidUseNICVm(_fCutOffDistance);
	else
		gpuBuildPyramidCVm();

    return;
}

void VideoSourceKinect::getNextFrameNormal(int* pnStatus_)
{
	if(_bIsSequenceEnds) { *pnStatus_ = PAUSE; _bIsSequenceEnds = false; }
	XnStatus nRetVal = 0;
	switch (*pnStatus_&MASK1)
	{
	case PAUSE://restart
		if (!_cImgMD.WritableRGB24Data()){
			nRetVal = _cContext.WaitAndUpdateAll();	CHECK_RC_ ( nRetVal, "UpdateData failed: " );
		}
		//do nothing
		break;
	case CONTINUE:
		nRetVal = _cContext.WaitAndUpdateAll();	if (nRetVal != 65568 /*end of file*/) CHECK_RC_ ( nRetVal, "UpdateData failed: " );
		break;
	default:
		PRINTSTR("convert to the status of CONTINUE");
		*pnStatus_ = (*pnStatus_&(~MASK1))|CONTINUE;
		nRetVal = _cContext.WaitAndUpdateAll();	CHECK_RC_ ( nRetVal, "UpdateData failed: " );
		break;
	}
	// these two lines are required for getting a stable image and depth.
	_cImgGen.GetMetaData ( _cImgMD );
	_cDepthGen.GetMetaData( _cDepthMD );
	cv::Mat cvmRGB(__aKinectH[_uResolution],__aKinectW[_uResolution],CV_8UC3, (unsigned char*)_cImgMD.WritableRGB24Data());
	cv::Mat cvmDep(__aKinectH[_uResolution],__aKinectW[_uResolution],CV_16UC1,(unsigned short*)_cDepthMD.WritableData());
	cvmRGB.copyTo(_cvmRGB);
	cvmDep.convertTo(_cvmDepth,CV_32FC1);
	//mail capturing function
	if (_bUseNIRegistration)
		gpuBuildPyramidUseNICVm(_fCutOffDistance);
	else
		gpuBuildPyramidCVm();
	_pCurrFrame->initRT();
    return;
}

void VideoSourceKinect::buildPyramid(btl::utility::tp_coordinate_convention eConvention_ ){
	/*_pFrame->_eConvention = eConvention_;
	// not fullly understand the lense distortion model used by OpenNI.
	//undistortRGB( _cvmRGB, &*_pFrame->_acvmShrPtrPyrRGBs[0] );
	cv::remap(_cvmRGB,_cvmUndistRGB,_pRGBCamera->_cvmMapX,_pRGBCamera->_cvmMapY, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	//undistortIR( _cvmDepth, &_cvmUndistDepth );
	cv::remap(_cvmDepth,_cvmUndistDepth,_pIRCamera->_cvmMapX,_pIRCamera->_cvmMapY, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	alignDepthWithRGB(_cvmUndistDepth,&*_pFrame->_acvmPyrDepths[0]);
	btl::utility::bilateralFilterInDisparity<float>(&*_pFrame->_acvmPyrDepths[0],_fSigmaDisparity,_fSigmaSpace);
	unprojectRGB(*_pFrame->_acvmPyrDepths[0],0, &*_pFrame->_acvmShrPtrPyrPts[0],eConvention_);
	fastNormalEstimation(*_pFrame->_acvmShrPtrPyrPts[0],&*_pFrame->_acvmShrPtrPyrNls[0]);
	for( unsigned int i=1; i<_uPyrHeight; i++ )	{
		cv::pyrDown(*_pFrame->_acvmShrPtrPyrRGBs[i-1],*_pFrame->_acvmShrPtrPyrRGBs[i]);
		btl::utility::downSampling<float>(*_pFrame->_acvmPyrDepths[i-1],&*_pFrame->_acvmPyrDepths[i]);
		btl::utility::bilateralFilterInDisparity<float>(&*_pFrame->_acvmPyrDepths[i],_fSigmaDisparity,_fSigmaSpace);
		unprojectRGB(*_pFrame->_acvmPyrDepths[i],i, &*_pFrame->_acvmShrPtrPyrPts[i],eConvention_);
		fastNormalEstimation(*_pFrame->_acvmShrPtrPyrPts[i],&*_pFrame->_acvmShrPtrPyrNls[i]);
	}*/
}
void VideoSourceKinect::gpuBuildPyramidUseNICVm(float fCutOffDistance_ ){
	_cvgmRGB.upload(_cvmRGB);
	_cvgmDepth.upload(_cvmDepth);
	_pCurrFrame->_acvgmShrPtrPyrRGBs[0]->setTo(0);//clear(RGB)
	cv::gpu::remap(_cvgmRGB, *_pCurrFrame->_acvgmShrPtrPyrRGBs[0], _pRGBCamera->_cvgmMapX, _pRGBCamera->_cvgmMapY, cv::INTER_NEAREST, cv::BORDER_CONSTANT  );
	
	_cvgmUndistDepth.setTo(std::numeric_limits<float>::quiet_NaN());//clear(_cvgmUndistDepth)
	cv::gpu::remap(_cvgmDepth, _cvgmUndistDepth, _pRGBCamera->_cvgmMapX, _pRGBCamera->_cvgmMapY, cv::INTER_NEAREST, cv::BORDER_CONSTANT  );
	//bilateral filtering (comments off the following three lines to get raw depth map image of kinect)
	btl::device::cudaDepth2Disparity2(_cvgmUndistDepth,fCutOffDistance_, &*_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[0]);//convert depth from mm to m
	btl::device::cudaBilateralFiltering(*_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[0],_fSigmaSpace,_fSigmaDisparity,&*_pCurrFrame->_acvgmShrPtrPyrDisparity[0]);
	btl::device::cudaDisparity2Depth(*_pCurrFrame->_acvgmShrPtrPyrDisparity[0],&*_pCurrFrame->_acvgmShrPtrPyrDepths[0]);
	//get pts and nls
	btl::device::unprojectRGBCVm(*_pCurrFrame->_acvgmShrPtrPyrDepths[0],_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v, 0,&*_pCurrFrame->_acvgmShrPtrPyrPts[0] );
	btl::device::cudaFastNormalEstimation(*_pCurrFrame->_acvgmShrPtrPyrPts[0],&*_pCurrFrame->_acvgmShrPtrPyrNls[0]);//_vcvgmPyrNls[0]);
	//generate black and white
	cv::gpu::cvtColor(*_pCurrFrame->_acvgmShrPtrPyrRGBs[0],*_pCurrFrame->_acvgmShrPtrPyrBWs[0],cv::COLOR_RGB2GRAY);

	//down-sampling
	for( unsigned int i=1; i<_uPyrHeight; i++ )	{
		_pCurrFrame->_acvgmShrPtrPyrRGBs[i]->setTo(0);
		cv::gpu::pyrDown(*_pCurrFrame->_acvgmShrPtrPyrRGBs[i-1],*_pCurrFrame->_acvgmShrPtrPyrRGBs[i]);
		cv::gpu::cvtColor(*_pCurrFrame->_acvgmShrPtrPyrRGBs[i],*_pCurrFrame->_acvgmShrPtrPyrBWs[i],cv::COLOR_RGB2GRAY);
		_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[i]->setTo(std::numeric_limits<float>::quiet_NaN());
		//btl::device::cudaPyrDown( *_pFrame->_acvgmShrPtrPyrDisparity[i-1],_fSigmaDisparity,&*_pFrame->_acvgmShrPtrPyr32FC1Tmp[i]);//need to compare the performance of cudaPyDown() with resize()
		cv::gpu::resize(*_pCurrFrame->_acvgmShrPtrPyrDisparity[i-1],*_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[i],_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[i]->size(),0,0,cv::INTER_LINEAR);
		btl::device::cudaBilateralFiltering(*_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[i],_fSigmaSpace,_fSigmaDisparity,&*_pCurrFrame->_acvgmShrPtrPyrDisparity[i]);
		btl::device::cudaDisparity2Depth(*_pCurrFrame->_acvgmShrPtrPyrDisparity[i],&*_pCurrFrame->_acvgmShrPtrPyrDepths[i]);
		btl::device::unprojectRGBCVm(*_pCurrFrame->_acvgmShrPtrPyrDepths[i],_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v, i,&*_pCurrFrame->_acvgmShrPtrPyrPts[i] );
		btl::device::cudaFastNormalEstimation(*_pCurrFrame->_acvgmShrPtrPyrPts[i],&*_pCurrFrame->_acvgmShrPtrPyrNls[i]);
	}	

	for( unsigned int i=0; i<_uPyrHeight; i++ )	{
		_pCurrFrame->_acvgmShrPtrPyrRGBs[i]->download(*_pCurrFrame->_acvmShrPtrPyrRGBs[i]);
		_pCurrFrame->_acvgmShrPtrPyrBWs[i]->download(*_pCurrFrame->_acvmShrPtrPyrBWs[i]);
		_pCurrFrame->_acvgmShrPtrPyrPts[i]->download(*_pCurrFrame->_acvmShrPtrPyrPts[i]);
		_pCurrFrame->_acvgmShrPtrPyrNls[i]->download(*_pCurrFrame->_acvmShrPtrPyrNls[i]);
	}
	//scale the depth map
	btl::device::scaleDepthCVmCVm(0,_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v,&*_pCurrFrame->_acvgmShrPtrPyrDepths[0]);

	return;
}
void VideoSourceKinect::gpuBuildPyramidCVm( ){
	_cvgmRGB.upload(_cvmRGB);
	_cvgmDepth.upload(_cvmDepth);
	_pCurrFrame->_acvgmShrPtrPyrRGBs[0]->setTo(0);//clear(RGB)
	cv::gpu::remap(_cvgmRGB, *_pCurrFrame->_acvgmShrPtrPyrRGBs[0], _pRGBCamera->_cvgmMapX, _pRGBCamera->_cvgmMapY, cv::INTER_NEAREST, cv::BORDER_CONSTANT  );
	_cvgmUndistDepth.setTo(std::numeric_limits<float>::quiet_NaN());//clear(_cvgmUndistDepth)
	cv::gpu::remap(_cvgmDepth, _cvgmUndistDepth, _pIRCamera->_cvgmMapX, _pIRCamera->_cvgmMapY, cv::INTER_NEAREST, cv::BORDER_CONSTANT  );
	gpuAlignDepthWithRGB( _cvgmUndistDepth, &*_pCurrFrame->_acvgmShrPtrPyrDepths[0] );//_cvgmAlignedRawDepth cleaned inside
	//bilateral filtering (comments off the following three lines to get raw depth map image of kinect)
	btl::device::cudaDepth2Disparity(*_pCurrFrame->_acvgmShrPtrPyrDepths[0], &*_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[0]);
	btl::device::cudaBilateralFiltering(*_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[0],_fSigmaSpace,_fSigmaDisparity,&*_pCurrFrame->_acvgmShrPtrPyrDisparity[0]);
	btl::device::cudaDisparity2Depth(*_pCurrFrame->_acvgmShrPtrPyrDisparity[0],&*_pCurrFrame->_acvgmShrPtrPyrDepths[0]);
	//get pts and nls
	btl::device::unprojectRGBCVm(*_pCurrFrame->_acvgmShrPtrPyrDepths[0],_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v, 0,&*_pCurrFrame->_acvgmShrPtrPyrPts[0] );
	btl::device::cudaFastNormalEstimation(*_pCurrFrame->_acvgmShrPtrPyrPts[0],&*_pCurrFrame->_acvgmShrPtrPyrNls[0]);//_vcvgmPyrNls[0]);
	//generate black and white
	cv::gpu::cvtColor(*_pCurrFrame->_acvgmShrPtrPyrRGBs[0],*_pCurrFrame->_acvgmShrPtrPyrBWs[0],cv::COLOR_RGB2GRAY);

	//down-sampling
	for( unsigned int i=1; i<_uPyrHeight; i++ )	{
		_pCurrFrame->_acvgmShrPtrPyrRGBs[i]->setTo(0);
		cv::gpu::pyrDown(*_pCurrFrame->_acvgmShrPtrPyrRGBs[i-1],*_pCurrFrame->_acvgmShrPtrPyrRGBs[i]);
		cv::gpu::cvtColor(*_pCurrFrame->_acvgmShrPtrPyrRGBs[i],*_pCurrFrame->_acvgmShrPtrPyrBWs[i],cv::COLOR_RGB2GRAY);
		_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[i]->setTo(std::numeric_limits<float>::quiet_NaN());
		btl::device::cudaPyrDown( *_pCurrFrame->_acvgmShrPtrPyrDisparity[i-1],_fSigmaDisparity,&*_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[i]);
		btl::device::cudaBilateralFiltering(*_pCurrFrame->_acvgmShrPtrPyr32FC1Tmp[i],_fSigmaSpace,_fSigmaDisparity,&*_pCurrFrame->_acvgmShrPtrPyrDisparity[i]);
		btl::device::cudaDisparity2Depth(*_pCurrFrame->_acvgmShrPtrPyrDisparity[i],&*_pCurrFrame->_acvgmShrPtrPyrDepths[i]);
		btl::device::unprojectRGBCVm(*_pCurrFrame->_acvgmShrPtrPyrDepths[i],_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v, i,&*_pCurrFrame->_acvgmShrPtrPyrPts[i] );
		btl::device::cudaFastNormalEstimation(*_pCurrFrame->_acvgmShrPtrPyrPts[i],&*_pCurrFrame->_acvgmShrPtrPyrNls[i]);
	}	

	for( unsigned int i=0; i<_uPyrHeight; i++ )	{
		_pCurrFrame->_acvgmShrPtrPyrRGBs[i]->download(*_pCurrFrame->_acvmShrPtrPyrRGBs[i]);
		_pCurrFrame->_acvgmShrPtrPyrBWs[i]->download(*_pCurrFrame->_acvmShrPtrPyrBWs[i]);
		_pCurrFrame->_acvgmShrPtrPyrPts[i]->download(*_pCurrFrame->_acvmShrPtrPyrPts[i]);
		_pCurrFrame->_acvgmShrPtrPyrNls[i]->download(*_pCurrFrame->_acvmShrPtrPyrNls[i]);	
	}
#if !USE_PBO
#endif
	//scale the depth map
	{
		btl::device::scaleDepthCVmCVm(0,_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v,&*_pCurrFrame->_acvgmShrPtrPyrDepths[0]);
		//for testing scaleDepthCVmCVm();
		//cv::Mat cvmTest,cvmTestScaled;
		//_pFrame->_acvgmShrPtrPyrDepths[i]->download(cvmTest);
		//btl::device::scaleDepthCVmCVm(i,_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v,&*_pFrame->_acvgmShrPtrPyrDepths[i]);
		//_pFrame->_acvgmShrPtrPyrDepths[i]->download(cvmTestScaled);
		//const float* pD = (const float*)cvmTest.data;
		//const float* pDS= (const float*)cvmTestScaled.data;
		//int nStep = 1<<i;
		//for (int r=0;r<cvmTest.rows;r++)
		//for (int c=0;c<cvmTest.cols;c++){
		//	float fRatio = *pDS++ / *pD++;
		//	float fTanX= (c*nStep-_pRGBCamera->_u)/_pRGBCamera->_fFx;
		//	float fTanY= (r*nStep-_pRGBCamera->_v)/_pRGBCamera->_fFy;
		//	float fSec = std::sqrt(fTanX*fTanX+fTanY*fTanY+1);
		//	if (fRatio==fRatio){
		//		BTL_ASSERT(std::fabs(fSec-fRatio)<0.00001,"scaleDepthCVmCVm() error");
		//	}
		//}
	}
	/*
	for (unsigned int i=_uResolution; i<_uPyrHeight;i++){

		//clear plane obj
		_pFrame->_vPlaneObjsDistanceNormal[i].clear();
		////check VNMap
		//cv::gpu::GpuMat cvgmCheckResults;
		//btl::device::checkNVMap(*_pFrame->_acvgmShrPtrPyrPts[i],*_pFrame->_acvgmShrPtrPyrNls[i],&cvgmCheckResults);
		//cv::Mat cvmResult;
		//cvgmCheckResults.download(cvmResult);
		//const short* pRe = (const short*) cvmResult.data;
		//int nRe = 0;
		//for (int i=0; i<btl::kinect::__aKinectWxH[i];i++){
		//	if(*pRe++>0) nRe++; 
		//}
		//if (nRe > 0)
		//{
		//	PRINT(nRe);
		//}
	}*/

	return;
}
void VideoSourceKinect::gpuAlignDepthWithRGB( const cv::gpu::GpuMat& cvgmUndistortDepth_ , cv::gpu::GpuMat* pcvgmAligned_){
	//clean data containers
	//unproject the depth map to IR coordinate and cudaUnprojectIRCVCV() no need to preset quiet_NaN
	btl::device::cudaUnprojectIRCVCV(cvgmUndistortDepth_,_pIRCamera->_fFx,_pIRCamera->_fFy,_pIRCamera->_u,_pIRCamera->_v, &_cvgmIRWorld);
	//findRange(_cvgmIRWorld);
	//transform from IR coordinate to RGB coordinate no need to preset quiet_NaN
	btl::device::cudaTransformIR2RGBCVCV(_cvgmIRWorld, _aR, _aRT, &_cvgmRGBWorld);
	//findRange(_cvgmRGBWorld);
	//project RGB coordinate to image to register the depth with rgb image
	btl::device::cudaProjectRGBCVCV(_cvgmRGBWorld, _pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v, pcvgmAligned_ );
	//findRange(*pcvgmAligned_);
	return;
}
void VideoSourceKinect::alignDepthWithRGB( const cv::Mat& cvUndistortDepth_ , cv::Mat* pcvAligned_)
{
	// initialize the Registered depth as NULLs
	//pcvAligned_->setTo(0);
	//unproject the depth map to IR coordinate
	//unprojectIR      ( cvUndistortDepth_, &_cvmIRWorld );
	//transform from IR coordinate to RGB coordinate
	//transformIR2RGB  ( _cvmIRWorld, &_cvmRGBWorld );
	//project RGB coordinate to image to register the depth with rgb image
	//projectRGB       ( _cvmRGBWorld,&(*pcvAligned_) );
	return;
}
void VideoSourceKinect::unprojectIR ( const cv::Mat& cvmDepth_, cv::Mat* pcvmIRWorld_)
{
	float* pWorld_ = (float*) pcvmIRWorld_->data;
	const float* pCamera_=  (const float*) cvmDepth_.data;
	for(int r = 0; r<cvmDepth_.rows; r++)
	for(int c = 0; c<cvmDepth_.cols; c++)
	{
		if ( 400.f < *pCamera_ && *pCamera_ < 3000.f){
			* ( pWorld_ + 2 ) = ( *pCamera_ ) / 1000.f; //convert to meter z 5 million meter is added according to experience. as the OpenNI
			//coordinate system is defined w.r.t. the camera plane which is 0.5 centimeters in front of the camera center
			*   pWorld_		  = ( c - _pIRCamera->_u ) / _pIRCamera->_fFx * *(pWorld_+2); // + 0.0025;     //x by experience.
			* ( pWorld_ + 1 ) = ( r - _pIRCamera->_v ) / _pIRCamera->_fFy * *(pWorld_+2); // - 0.00499814; //y the value is esimated using CCalibrateKinectExtrinsics::calibDepth(
		}
		else{
			*(pWorld_) = *(pWorld_+1) = *(pWorld_+2) = 0;
		}
		pCamera_ ++;
		pWorld_ += 3;
	}
}
void VideoSourceKinect::transformIR2RGB  ( const cv::Mat& cvmIRWorld, cv::Mat* pcvmRGBWorld_ ){
	//_aR[0] [1] [2]
	//   [3] [4] [5]
	//   [6] [7] [8]
	//_aT[0]
	//   [1]
	//   [2]
	//  pRGB_ = _aR * ( pIR_ - _aT )
	//  	  = _aR * pIR_ - _aR * _aT
	//  	  = _aR * pIR_ - _aRT

	float* pRGB_ = (float*) pcvmRGBWorld_->data;
	const float* pIR_=  (float*) cvmIRWorld.data;
	for(int r = 0; r<cvmIRWorld.rows; r++)
	for(int c = 0; c<cvmIRWorld.cols; c++){
		if( 0.4f < fabsf( * ( pIR_ + 2 ) ) && fabsf( * ( pIR_ + 2 ) ) < 3.f ) {
			* pRGB_++ = _aR[0] * *pIR_ + _aR[1] * *(pIR_+1) + _aR[2] * *(pIR_+2) - _aRT[0];
			* pRGB_++ = _aR[3] * *pIR_ + _aR[4] * *(pIR_+1) + _aR[5] * *(pIR_+2) - _aRT[1];
			* pRGB_++ = _aR[6] * *pIR_ + _aR[7] * *(pIR_+1) + _aR[8] * *(pIR_+2) - _aRT[2];
		}
		else {
			* pRGB_++ = 0;
			* pRGB_++ = 0;
			* pRGB_++ = 0;
		}
		pIR_ += 3;
	}
}
void VideoSourceKinect::projectRGB ( const cv::Mat& cvmRGBWorld_, cv::Mat* pcvAlignedRGB_ ){
	//cout << "projectRGB() starts." << std::endl;
	unsigned short nX, nY;
	int nIdx1,nIdx2;
	float dX,dY,dZ;

	CHECK( CV_32FC1 == pcvAlignedRGB_->type(), "the depth pyramid level 1 must be CV_32FC1" );
	float* pWorld_ = (float*) cvmRGBWorld_.data;
	float* pDepth = (float*) pcvAlignedRGB_->data;

	for ( int i = 0; i < KINECT_WxH; i++ ){
		dX = *   pWorld_;
		dY = * ( pWorld_ + 1 );
		dZ = * ( pWorld_ + 2 );
		if (0.4 < fabsf( dZ ) && fabsf( dZ ) < 3){
			// get 2D image projection in RGB image of the XYZ in the world
			nX = cvRound( _pRGBCamera->_fFx * dX / dZ + _pRGBCamera->_u );
			nY = cvRound( _pRGBCamera->_fFy * dY / dZ + _pRGBCamera->_v );
			// set 2D rgb XYZ
			if ( nX >= 0 && nX < KINECT_WIDTH && nY >= 0 && nY < KINECT_HEIGHT ){
				nIdx1= nY * KINECT_WIDTH + nX; //1 channel
				nIdx2= ( nIdx1 ) * 3; //3 channel
				pDepth    [ nIdx1   ] = float(dZ);
				//PRINT( nX ); PRINT( nY ); PRINT( pWorld_ );
			}
		}
		pWorld_ += 3;
	}
	return;
}
void VideoSourceKinect::unprojectRGB ( const cv::Mat& cvmDepth_, int nLevel, cv::Mat* pcvmPts_, tp_coordinate_convention eConvention_ /*= GL*/ )
{
	pcvmPts_->setTo(0);
	// pCamer format
	// 0 x (c) 1 y (r) 2 d
	//the pixel coordinate is defined w.r.t. camera reference, which is defined as x-left, y-downward and z-forward. It's
	//a right hand system. i.e. opencv-default reference system;
	//unit is meter
	//when rendering the point using opengl's camera reference which is defined as x-left, y-upward and z-backward. the
	//for example: glVertex3d ( Pt(0), -Pt(1), -Pt(2) ); i.e. opengl-default reference system
	int nScale = 1 << nLevel;

	float *pDepth = (float*) cvmDepth_.data;
	float *pWorld_ = (float*) pcvmPts_->data;
	float fD;
	for ( int r = 0; r < cvmDepth_.rows; r++ )
	for ( int c = 0; c < cvmDepth_.cols; c++ )	{
		fD = *pDepth++;
		if( 0.4 < fabsf( fD ) && fabsf( fD ) < 3 ){
			* ( pWorld_ + 2 ) = fD;
			*   pWorld_		  = ( c*nScale - _pRGBCamera->_u ) / _pRGBCamera->_fFx * fD; // + 0.0025;     //x by experience.
			* ( pWorld_ + 1 ) = ( r*nScale - _pRGBCamera->_v ) / _pRGBCamera->_fFy * fD; // - 0.00499814; //y the value is esimated using CCalibrateKinectExtrinsics::calibDepth(
			//convert from opencv convention to opengl convention
			if (btl::utility::BTL_GL == eConvention_){
				* ( pWorld_ + 1 ) = -*( pWorld_ + 1 );
				* ( pWorld_ + 2 ) = -*( pWorld_ + 2 );
			}
		}
		else{
			* pWorld_ = *(pWorld_+1) = *(pWorld_+2) = 0.f;
		}
		pWorld_ += 3;
	}
	return;
}

void VideoSourceKinect::fastNormalEstimation(const cv::Mat& cvmPts_, cv::Mat* pcvmNls_)
{
	pcvmNls_->setTo(0);
	Eigen::Vector3f n1, n2, n3;

	const float* pPt_ = (const float*) cvmPts_.data;
	float* pNl_ = (float*) pcvmNls_->data;
	//calculate normal
	for( int r = 0; r < cvmPts_.rows; r++ )
	for( int c = 0; c < cvmPts_.cols; c++ )
	{
		if (c == cvmPts_.cols-1 || r == cvmPts_.rows-1) {
			pPt_+=3;
			pNl_+=3;
			continue;
		}
		if ( c == 420 && r == 60 ) {
			int x = 0;
		}
		// skip the right and bottom boarder line
		Eigen::Vector3f pti  ( pPt_[0],pPt_[1],pPt_[2] );
		Eigen::Vector3f pti1 ( pPt_[3],pPt_[4],pPt_[5] ); //left
		Eigen::Vector3f ptj1 ( pPt_[cvmPts_.cols*3],pPt_[cvmPts_.cols*3+1],pPt_[cvmPts_.cols*3+2] ); //down

		if( fabs( pti(2) ) > 0.0000001 && fabs( pti1(2) ) > 0.0000001 && fabs( ptj1(2) ) > 0.0000001 ) {
			n1 = pti1 - pti;
			n2 = ptj1 - pti;
			n3 = n1.cross(n2);
			float fNorm = n3.norm() ;
			if ( fNorm > SMALL ) {
				n3/=fNorm;
				if ( -n3(0)*pti[0] - n3(1)*pPt_[1] - n3(2)*pPt_[2] <0 ) {
					pNl_[0] = -n3(0);
					pNl_[1] = -n3(1);
					pNl_[2] = -n3(2);
				}
				else{
					pNl_[0] = n3(0);
					pNl_[1] = n3(1);
					pNl_[2] = n3(2);
				}
			}//if dNorm
		} //if

		pPt_+=3;
		pNl_+=3;
	}
	return;
}

void VideoSourceKinect::setResolution(ushort uResolutionLevel_){
	 _uResolution = uResolutionLevel_;
	XnStatus nRetVal = XN_STATUS_OK;
	
	_sModeVGA.nFPS = 30; 
	switch(_uResolution){
	case 3:
		_sModeVGA.nXRes = 80; 
		_sModeVGA.nYRes = 60; 
	case 2:
		_sModeVGA.nXRes = 160; 
		_sModeVGA.nYRes = 120; 
		break;
	case 1:
		_sModeVGA.nXRes = 320; 
		_sModeVGA.nYRes = 240; 
		break;
	case 0:
		_sModeVGA.nXRes = 640; 
		_sModeVGA.nYRes = 480; 
		break;
	}

	return;
}

} //namespace kinect
} //namespace btl
