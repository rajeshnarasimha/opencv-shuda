/**
* @file VideoSourceKinect.cpp
* @brief load of data from a kinect device 
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.0
* @date 2011-02-23
*/
#define INFO
//#define TIMER
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
#include "Histogram.h"
#include "KeyFrame.h"
#include "VideoSourceKinect.hpp"
#include "cuda/CudaLib.h"

#include <iostream>
#include <string>
#include <limits>


#define CHECK_RC(rc, what)	\
	BTL_ASSERT(rc == XN_STATUS_OK, (what + std::string(xnGetStatusString(rc))) )

using namespace btl::utility;

namespace btl{ namespace kinect
{

VideoSourceKinect::VideoSourceKinect ()
//:CCalibrateKinect()
{
    std::cout << "  VideoSourceKinect: Opening Kinect..." << std::endl;

    XnStatus nRetVal = XN_STATUS_OK;
    //initialize OpenNI context
    nRetVal = _cContext.Init();
    CHECK_RC ( nRetVal, "Initialize context: " );
    //create a image generator
    nRetVal =  _cImgGen.Create ( _cContext );
    CHECK_RC ( nRetVal, "Create image generator: " );
    //create a depth generator
    nRetVal =  _cDepthGen.Create ( _cContext );
    CHECK_RC ( nRetVal, "Create depth generator: " );
    //start generating data
    nRetVal = _cContext.StartGeneratingAll();
    CHECK_RC ( nRetVal, "Start generating data: " );
	//register the depth generator with the image generator
    //nRetVal = _cDepthGen.GetAlternativeViewPointCap().SetViewPoint ( _cImgGen );
	//CHECK_RC ( nRetVal, "Getting and setting AlternativeViewPoint failed: " ); 
	PRINTSTR("Kinect connected");
	//import camera parameters
	_pRGBCamera.reset(new SCamera(btl::kinect::SCamera::CAMERA_RGB));
	_pIRCamera .reset(new SCamera(btl::kinect::SCamera::CAMERA_IR));
	importYML();
	PRINTSTR("data holder constructed...");
	_pFrame.reset(new CKeyFrame(_pRGBCamera.get()));
	//allocate
    _cvmRGB			   .create( KINECT_HEIGHT, KINECT_WIDTH, CV_8UC3 );
	_cvmUndistRGB	   .create( KINECT_HEIGHT, KINECT_WIDTH, CV_8UC3 );
	_cvmDepth		   .create( KINECT_HEIGHT, KINECT_WIDTH, CV_32FC1);
	_cvmUndistDepth	   .create( KINECT_HEIGHT, KINECT_WIDTH, CV_32FC1);
	_cvmAlignedRawDepth.create( KINECT_HEIGHT, KINECT_WIDTH, CV_32FC1);

	_cvmIRWorld .create(KINECT_HEIGHT,KINECT_WIDTH,CV_32FC3);
	_cvmRGBWorld.create(KINECT_HEIGHT,KINECT_WIDTH,CV_32FC3);


	// allocate memory for later use ( registrate the depth with rgb image
	// refreshed for every frame
	// pre-allocate cvgm to increase the speed
	_cvgmIRWorld        .create(KINECT_HEIGHT,KINECT_WIDTH,CV_32FC3);
	_cvgmRGBWorld       .create(KINECT_HEIGHT,KINECT_WIDTH,CV_32FC3);
	_cvgmAlignedRawDepth.create(KINECT_HEIGHT,KINECT_WIDTH,CV_32FC1);
	_cvgm32FC1Tmp       .create(KINECT_HEIGHT,KINECT_WIDTH,CV_32FC1);

	_cvgmUndistDepth    .create(KINECT_HEIGHT,KINECT_WIDTH,CV_32FC1);

	for(int i=0; i<4; i++)	{
		int nRows = KINECT_HEIGHT>>i; 
		int nCols = KINECT_WIDTH>>i;
		//device
		_vcvgmPyrDisparity .push_back(cv::gpu::GpuMat(nRows,nCols,CV_32FC1));
		_vcvgmPyr32FC1Tmp  .push_back(cv::gpu::GpuMat(nRows,nCols,CV_32FC1));
		//host
		//rendering (static)
		btl::kinect::CKeyFrame::_acvgmShrPtrAA[i].reset(new cv::gpu::GpuMat(nRows,nCols,CV_32FC3));
		btl::kinect::CKeyFrame::_acvmShrPtrAA[i].reset(new cv::Mat(nRows,nCols,CV_32FC3));
		PRINTSTR("construct pyrmide level:");
		PRINT(i);
	}

	//other
	//definition of parameters
	_fThresholdDepthInMeter = 0.01f;
	_fSigmaSpace = 1;
	_fSigmaDisparity = 1.f/.6f - 1.f/(.6f+_fThresholdDepthInMeter);
	_uPyrHeight = 1;

	//default centroid follows opencv-convention
	_dXCentroid = _dYCentroid = 0;
	_dZCentroid = 1.0;
	//initialize normal histogram
	btl::kinect::CKeyFrame::_sNormalHist.init(2);
	btl::kinect::CKeyFrame::_sDistanceHist.init(30);

	std::cout << " Done. " << std::endl;
}
VideoSourceKinect::~VideoSourceKinect()
{
    _cContext.Release();
}
void VideoSourceKinect::importYML()
{
	//_pRGBCamera->importYML();
	//_pIRCamera->importYML();

	// create and open a character archive for output
#if __linux__
	cv::FileStorage cFSRead( "/space/csxsl/src/opencv-shuda/Data/kinect_intrinsics.yml", cv::FileStorage::READ );
#else if _WIN32 || _WIN64
	cv::FileStorage cFSRead ( "C:\\csxsl\\src\\opencv-shuda\\Data\\kinect_intrinsics.yml", cv::FileStorage::READ );
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
void VideoSourceKinect::getNextFrame(tp_frame eFrameType_)
{
    //get next frame
#ifdef TIMER	
	// timer on
	_cT0 =  boost::posix_time::microsec_clock::local_time(); 
#endif
	//
    XnStatus nRetVal = _cContext.WaitAndUpdateAll();
    CHECK_RC ( nRetVal, "UpdateData failed: " );
	// these two lines are required for getting a stable image and depth.
    _cImgGen.GetMetaData ( _cImgMD );
    _cDepthGen.GetMetaData( _cDepthMD );

    const XnRGB24Pixel* pRGBImg  = _cImgMD.RGB24Data();
	     unsigned char* pRGB = _cvmRGB.data;
    const unsigned short* pDepth   = (unsigned short*)_cDepthMD.Data();
	      float* pcvDepth = (float*)_cvmDepth.data;
    
    //XnStatus nRetVal = _cContext.WaitOneUpdateAll( _cIRGen );
    //CHECK_RC ( nRetVal, "UpdateData failed: " );
		  
	for( unsigned int i = 0; i < __aKinectWxH[0]; i++,pRGBImg++){
        // notice that OpenCV is use BGR order
        *pRGB++ = uchar(pRGBImg->nRed);
        *pRGB++ = uchar(pRGBImg->nGreen);
        *pRGB++ = uchar(pRGBImg->nBlue);
		*pcvDepth++ = *pDepth++;
    }
    switch( eFrameType_ ){
		case CPU_PYRAMID_CV:
			buildPyramid( btl::utility::BTL_CV );
			break;
		case GPU_PYRAMID_CV:
			gpuBuildPyramid( btl::utility::BTL_CV );
			break;
		case CPU_PYRAMID_GL:
			buildPyramid( btl::utility::BTL_GL );
			break;
		case GPU_PYRAMID_GL:
			gpuBuildPyramid( btl::utility::BTL_GL );
			break;
    }
#ifdef TIMER
// timer off
	_cT1 =  boost::posix_time::microsec_clock::local_time(); 
 	_cTDAll = _cT1 - _cT0 ;
	_fFPS = 1000.f/_cTDAll.total_milliseconds();
	PRINT( _fFPS );
#endif
	//cout << " getNextFrame() ends."<< endl;
    return;
}
void VideoSourceKinect::buildPyramid(btl::utility::tp_coordinate_convention eConvention_ ){
	_pFrame->_eConvention = eConvention_;
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
	}
}

void VideoSourceKinect::gpuBuildPyramid(btl::utility::tp_coordinate_convention eConvention_ ){
	_pFrame->_eConvention = eConvention_;
	_cvgmRGB.upload(_cvmRGB);
	_cvgmDepth.upload(_cvmDepth);
	_pFrame->_acvgmShrPtrPyrRGBs[0]->setTo(0);//clear(RGB)
	cv::gpu::remap(_cvgmRGB, *_pFrame->_acvgmShrPtrPyrRGBs[0], _pRGBCamera->_cvgmMapX, _pRGBCamera->_cvgmMapY, cv::INTER_NEAREST, cv::BORDER_CONSTANT  );
	_cvgmUndistDepth.setTo(std::numeric_limits<float>::quiet_NaN());//clear(_cvgmUndistDepth)
	cv::gpu::remap(_cvgmDepth, _cvgmUndistDepth, _pIRCamera->_cvgmMapX, _pIRCamera->_cvgmMapY, cv::INTER_NEAREST, cv::BORDER_CONSTANT  );
	gpuAlignDepthWithRGB( _cvgmUndistDepth, &_cvgmAlignedRawDepth );//_cvgmAlignedRawDepth cleaned inside
	_vcvgmPyr32FC1Tmp[0].setTo(std::numeric_limits<float>::quiet_NaN());
	btl::device::cudaDepth2Disparity(_cvgmAlignedRawDepth, &_vcvgmPyr32FC1Tmp[0]);
	_vcvgmPyrDisparity[0].setTo(std::numeric_limits<float>::quiet_NaN());
	btl::device::cudaBilateralFiltering(_vcvgmPyr32FC1Tmp[0],_fSigmaSpace,_fSigmaDisparity,&_vcvgmPyrDisparity[0]);
	_pFrame->_acvgmPyrDepths[0]->setTo(std::numeric_limits<float>::quiet_NaN());
	btl::device::cudaDisparity2Depth(_vcvgmPyrDisparity[0],&*_pFrame->_acvgmPyrDepths[0]);
	_pFrame->_acvgmShrPtrPyrPts[0]->setTo(std::numeric_limits<float>::quiet_NaN());
	btl::device::cudaUnprojectRGBCVBOTH(*_pFrame->_acvgmPyrDepths[0],_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v, 0,&*_pFrame->_acvgmShrPtrPyrPts[0],eConvention_);
	_pFrame->_acvgmShrPtrPyrNls[0]->setTo(std::numeric_limits<float>::quiet_NaN());
	btl::device::cudaFastNormalEstimation(*_pFrame->_acvgmShrPtrPyrPts[0],&*_pFrame->_acvgmShrPtrPyrNls[0]);//_vcvgmPyrNls[0]);
	_pFrame->_acvgmShrPtrPyrRGBs[0]->download(*_pFrame->_acvmShrPtrPyrRGBs[0]);
	_pFrame->_acvgmShrPtrPyrPts[0]->download(*_pFrame->_acvmShrPtrPyrPts[0]);
	_pFrame->_acvgmShrPtrPyrNls[0]->download(*_pFrame->_acvmShrPtrPyrNls[0]);
	cv::gpu::cvtColor(*_pFrame->_acvgmShrPtrPyrRGBs[0],*_pFrame->_acvgmShrPtrPyrBWs[0],cv::COLOR_RGB2GRAY);
	_pFrame->_acvgmShrPtrPyrBWs[0]->download(*_pFrame->_acvmShrPtrPyrBWs[0]);
	for( unsigned int i=1; i<_uPyrHeight; i++ )	{
		_pFrame->_acvgmShrPtrPyrRGBs[i]->setTo(0);
		cv::gpu::pyrDown(*_pFrame->_acvgmShrPtrPyrRGBs[i-1],*_pFrame->_acvgmShrPtrPyrRGBs[i]);
		_pFrame->_acvgmShrPtrPyrRGBs[i]->download(*_pFrame->_acvmShrPtrPyrRGBs[i]);
		cv::gpu::cvtColor(*_pFrame->_acvgmShrPtrPyrRGBs[i],*_pFrame->_acvgmShrPtrPyrBWs[i],cv::COLOR_RGB2GRAY);
		_pFrame->_acvgmShrPtrPyrBWs[i]->download(*_pFrame->_acvmShrPtrPyrBWs[i]);
		_vcvgmPyr32FC1Tmp[i].setTo(std::numeric_limits<float>::quiet_NaN());
		btl::device::cudaPyrDown( _vcvgmPyrDisparity[i-1],_fSigmaDisparity,&_vcvgmPyr32FC1Tmp[i]);
		_vcvgmPyrDisparity[i].setTo(std::numeric_limits<float>::quiet_NaN());
		btl::device::cudaBilateralFiltering(_vcvgmPyr32FC1Tmp[i],_fSigmaSpace,_fSigmaDisparity,&_vcvgmPyrDisparity[i]);
		_pFrame->_acvgmPyrDepths[i]->setTo(std::numeric_limits<float>::quiet_NaN());
		btl::device::cudaDisparity2Depth(_vcvgmPyrDisparity[i],&*_pFrame->_acvgmPyrDepths[i]);
		_pFrame->_acvgmShrPtrPyrPts[i]->setTo(std::numeric_limits<float>::quiet_NaN());
		btl::device::cudaUnprojectRGBCVBOTH(*_pFrame->_acvgmPyrDepths[i],_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v, i,&*_pFrame->_acvgmShrPtrPyrPts[i],eConvention_);
		_pFrame->_acvgmShrPtrPyrPts[i]->download(*_pFrame->_acvmShrPtrPyrPts[i]);
		_pFrame->_acvgmShrPtrPyrNls[i]->setTo(std::numeric_limits<float>::quiet_NaN());
		btl::device::cudaFastNormalEstimation(*_pFrame->_acvgmShrPtrPyrPts[i],&*_pFrame->_acvgmShrPtrPyrNls[i]);
		_pFrame->_acvgmShrPtrPyrNls[i]->download(*_pFrame->_acvmShrPtrPyrNls[i]);	
	}	
	//scale the depthmap
	for (int i=0; i<_uPyrHeight;i++){
		btl::device::scaleDepthCVmCVm(i,_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v,&*_pFrame->_acvgmPyrDepths[i]);
		//for testing scaleDepthCVmCVm();
		//cv::Mat cvmTest,cvmTestScaled;
		//_pFrame->_acvgmPyrDepths[i]->download(cvmTest);
		//btl::device::scaleDepthCVmCVm(i,_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v,&*_pFrame->_acvgmPyrDepths[i]);
		//_pFrame->_acvgmPyrDepths[i]->download(cvmTestScaled);
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
	return;
}
void VideoSourceKinect::gpuAlignDepthWithRGB( const cv::gpu::GpuMat& cvgmUndistortDepth_ , cv::gpu::GpuMat* pcvgmAligned_){
	//clean data containers
	_cvgmIRWorld.setTo(std::numeric_limits<float>::quiet_NaN());
	//unproject the depth map to IR coordinate
	btl::device::cudaUnprojectIRCVCV(cvgmUndistortDepth_,_pIRCamera->_fFx,_pIRCamera->_fFy,_pIRCamera->_u,_pIRCamera->_v, &_cvgmIRWorld);
	//findRange(_cvgmIRWorld);
	_cvgmRGBWorld.setTo(std::numeric_limits<float>::quiet_NaN());
	//transform from IR coordinate to RGB coordinate
	btl::device::cudaTransformIR2RGBCVCV(_cvgmIRWorld, _aR, _aRT, &_cvgmRGBWorld);
	//findRange(_cvgmRGBWorld);
	pcvgmAligned_->setTo(std::numeric_limits<float>::quiet_NaN());
	//project RGB coordinate to image to register the depth with rgb image
	//cv::gpu::GpuMat cvgmAligned_(cvgmUndistortDepth_.size(),CV_32FC1);
	btl::device::cudaProjectRGBCVCV(_cvgmRGBWorld, _pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v, pcvgmAligned_ );
	//findRange(*pcvgmAligned_);
	return;
}
void VideoSourceKinect::alignDepthWithRGB( const cv::Mat& cvUndistortDepth_ , cv::Mat* pcvAligned_)
{
	// initialize the Registered depth as NULLs
	pcvAligned_->setTo(0);
	//unproject the depth map to IR coordinate
	unprojectIR      ( cvUndistortDepth_, &_cvmIRWorld );
	//transform from IR coordinate to RGB coordinate
	transformIR2RGB  ( _cvmIRWorld, &_cvmRGBWorld );
	//project RGB coordinate to image to register the depth with rgb image
	projectRGB       ( _cvmRGBWorld,&(*pcvAligned_) );
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



} //namespace kinect
} //namespace btl
