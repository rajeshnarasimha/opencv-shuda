/**
* @file VideoSourceKinect.cpp
* @brief load of data from a kinect device 
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.0
* @date 2011-02-23
*/
//#define INFO
#include <opencv2/gpu/gpu.hpp>
#include "VideoSourceKinect.hpp"
#include "Utility.hpp"
#include "cuda/CudaLib.h"

#include <iostream>
#include <string>


#define CHECK_RC(rc, what)	\
	BTL_ASSERT(rc == XN_STATUS_OK, (what + std::string(xnGetStatusString(rc))) )

using namespace btl::utility;

namespace btl{ namespace extra { namespace videosource
{

VideoSourceKinect::VideoSourceKinect ()
:CCalibrateKinect()
{
    std::cout << "  VideoSource_Linux: Opening Kinect..." << std::endl;

    XnStatus nRetVal = XN_STATUS_OK;
    //initialize OpenNI context
/*    
    nRetVal = _cContext.InitFromXmlFile("/space/csxsl/src/btl-shuda/Kinect.xml"); 
    CHECK_RC ( nRetVal, "Initialize context: " );
    nRetVal = _cContext.FindExistingNode(XN_NODE_TYPE_IR, _cIRGen); 
    CHECK_RC ( nRetVal, "Find existing node: " );
*/    
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

    _cvmRGB.create( KINECT_HEIGHT, KINECT_WIDTH, CV_8UC3 );
	_cvmDepth.create( KINECT_HEIGHT, KINECT_WIDTH, CV_16UC1 );
	_cvmAlignedRawDepth = cv::Mat::zeros(KINECT_HEIGHT,KINECT_WIDTH,CV_32F);

	// allocate memory for later use ( registrate the depth with rgb image
	_pIRWorld = new double[ KINECT_WxHx3 ]; //XYZ w.r.t. IR camera reference system
	_pPxDIR	  = new unsigned short[ KINECT_WxHx3 ]; //pixel coordinate and depth 
	// refreshed for every frame
	_pRGBWorld    = new double[ KINECT_WxHx3 ];//X,Y,Z coordinate of depth w.r.t. RGB camera reference system
	_pRGBWorldRGB = new double[ KINECT_WxHx3 ];//aligned to RGB image of the X,Y,Z coordinate
	// pre-allocate cvgm to increase the speed
	_cvgmIRWorld          .create(KINECT_HEIGHT,KINECT_WIDTH,CV_32FC3);
	_cvgmRGBWorld         .create(KINECT_HEIGHT,KINECT_WIDTH,CV_32FC3);
	_cvgmAlignedRawDepth  .create(KINECT_HEIGHT,KINECT_WIDTH,CV_32FC1);
	_cvgm32FC1Tmp         .create(KINECT_HEIGHT,KINECT_WIDTH,CV_32FC1);

	//disparity
	for(int i=0; i<4; i++)
	{
		int nRows = KINECT_HEIGHT>>i; 
		int nCols = KINECT_WIDTH>>i;
		//device
		_vcvgmPyrDepths    .push_back(cv::gpu::GpuMat(nRows,nCols,CV_32FC1));
		_vcvgmPyrDisparity .push_back(cv::gpu::GpuMat(nRows,nCols,CV_32FC1));
		_vcvgmPyrRGBs      .push_back(cv::gpu::GpuMat(nRows,nCols,CV_8UC3));
		_vcvgmPyr32FC1Tmp  .push_back(cv::gpu::GpuMat(nRows,nCols,CV_32FC1));
		_vcvgmPyrPts	   .push_back(cv::gpu::GpuMat(nRows,nCols,CV_32FC3));
		_vcvgmPyrNls	   .push_back(cv::gpu::GpuMat(nRows,nCols,CV_32FC3));
		//host
		_vcvmPyrDepths.push_back(cv::Mat(nRows,nCols,CV_32FC1));
		_vcvmPyrRGBs  .push_back(cv::Mat(nRows,nCols,CV_8UC3 ));
		_vcvmPyrPts   .push_back(cv::Mat(nRows,nCols,CV_32FC3));
		_vcvmPyrNls   .push_back(cv::Mat(nRows,nCols,CV_32FC3));
	}

	//other
    _ePreFiltering = RAW; 
	//definition of parameters
	_dThresholdDepth = 10;
	_fSigmaSpace = 4.5;
	_fSigmaDisparity = 1./600 - 1./(600+_dThresholdDepth);
	_uPyrHeight = 1;

	//default centroid follows opencv-convention
	_dXCentroid = _dYCentroid = 0;
	_dZCentroid = 1.0;

	std::cout << " Done. " << std::endl;
}

VideoSourceKinect::~VideoSourceKinect()
{
    _cContext.Release();
	delete [] _pIRWorld;
	delete [] _pPxDIR;
	delete [] _pRGBWorld;
	delete [] _pRGBWorldRGB;
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
void VideoSourceKinect::getNextFrame()
{
    //get next frame
    //set as _frame
	//cout << " getNextFrame() start."<< endl;
	cv::gpu::GpuMat& cvgmAlignedDepthL0 = _vcvgmPyrDepths[0];
	cv::gpu::GpuMat& cvgmDisparity      = _vcvgmPyrDisparity[0];

    XnStatus nRetVal = _cContext.WaitAndUpdateAll();
    CHECK_RC ( nRetVal, "UpdateData failed: " );
	// these two lines are required for getting a stable image and depth.
    _cImgGen.GetMetaData ( _cImgMD );
    _cDepthGen.GetMetaData( _cDepthMD );

    const XnRGB24Pixel* pRGBImg  = _cImgMD.RGB24Data();
	     unsigned char* pRGB = _cvmRGB.data;
    const unsigned short* pDepth   = (unsigned short*)_cDepthMD.Data();
	      unsigned short* pcvDepth = (unsigned short*)_cvmDepth.data;
    
    //XnStatus nRetVal = _cContext.WaitOneUpdateAll( _cIRGen );
    //CHECK_RC ( nRetVal, "UpdateData failed: " );
		  
	for( unsigned int i = 0; i < __aKinectWxH[0]; i++)
	{
        // notice that OpenCV is use BGR order
        *pRGB++ = uchar(pRGBImg->nRed);
        *pRGB++ = uchar(pRGBImg->nGreen);
        *pRGB++ = uchar(pRGBImg->nBlue);
		pRGBImg++;

		*pcvDepth++ = *pDepth++;
    }
	// not fullly understand the lense distortion model used by OpenNI.
	//undistortRGB( _cvmRGB, _cvmUndistRGB );
	//undistortIR( _cvmDepth, _cvmUndistDepth );

	_cvgmRGB.upload(_cvmRGB);
	_cvgmDepth.upload(_cvmDepth);
	gpuUndistortRGB(_cvgmRGB,&_vcvgmPyrRGBs[0]);
	gpuUndistortIR (_cvgmDepth,&_cvgmUndistDepth);
	
	//cvtColor( _cvmUndistRGB, _cvmUndistBW, CV_RGB2GRAY );

#ifdef TIMER	
	// timer on
	_cT0 =  boost::posix_time::microsec_clock::local_time(); 
#endif
    switch( _ePreFiltering )
    {
		case RAW: //default
			gpuAlignDepthWithRGB( _cvgmUndistDepth, &_cvgmAlignedRawDepth );
			_cvgmAlignedRawDepth.download(_cvmAlignedRawDepth);
			//alignDepthWithRGB( _cvmUndistDepth, &_cvmAlignedDepthL0 );
			break;
        case GAUSSIAN:
			PRINT(_fSigmaSpace);
			gpuAlignDepthWithRGB( _cvgmUndistDepth, &cvgmAlignedDepthL0 );
			cvgmAlignedDepthL0.download(_cvmAlignedRawDepth);
			//alignDepthWithRGB( _cvmUndistDepth, &_cvmAlignedDepthL0 );
			{
				cv::Mat cvmGaussianFiltered;
				cv::GaussianBlur(_cvmAlignedRawDepth, cvmGaussianFiltered, cv::Size(0,0), _fSigmaSpace, _fSigmaSpace);
				_cvmAlignedRawDepth = cvmGaussianFiltered;
			}
			break;
        case GAUSSIAN_C1:
			PRINT(_dThresholdDepth);
			PRINT(_fSigmaSpace);
			gpuAlignDepthWithRGB( _cvgmUndistDepth, &cvgmAlignedDepthL0 );
			cvgmAlignedDepthL0.download(_cvmAlignedRawDepth);
			//alignDepthWithRGB( _cvmUndistDepth, &_cvmAlignedDepthL0 );
			{
				cv::Mat cvmGaussianFiltered;
				cv::GaussianBlur(_cvmAlignedRawDepth, cvmGaussianFiltered, cv::Size(0,0), _fSigmaSpace, _fSigmaSpace);
				btl::utility::filterDepth <float> ( _dThresholdDepth, (cv::Mat_<float>)cvmGaussianFiltered, (cv::Mat_<float>*)&_cvmAlignedRawDepth );
			}
			break;
        case GAUSSIAN_C1_FILTERED_IN_DISPARTY:
			PRINT(_fSigmaDisparity);
			PRINT(_fSigmaSpace);
			gpuAlignDepthWithRGB( _cvgmUndistDepth, &cvgmAlignedDepthL0 );
			cvgmAlignedDepthL0.download(_cvmAlignedRawDepth);
			//alignDepthWithRGB( _cvmUndistDepth, &_cvmAlignedDepthL0  );
			btl::utility::gaussianC1FilterInDisparity<float>( &_cvmAlignedRawDepth, _fSigmaDisparity, _fSigmaSpace );
            break;
		case BILATERAL_FILTERED_IN_DISPARTY:
			PRINT(_fSigmaDisparity);
			PRINT(_fSigmaSpace);
			gpuAlignDepthWithRGB( _cvgmUndistDepth, &_vcvgmPyrDepths[0] );
			_vcvgmPyrRGBs[0].download(_vcvmPyrRGBs[0]);
			_vcvgmPyrDepths[0].download(_vcvmPyrDepths[0]);
			unprojectRGBGL(_vcvmPyrDepths[0],);
			//gpuFastNormalEstimationGL(0,&_vcvgmPyrPts[0],&_vcvgmPyrNls[0]);
			_vcvgmPyrPts[0].download(_vcvmPyrPts[0]);
			_vcvgmPyrNls[0].download(_vcvmPyrNls[0]);
			//btl::cuda_util::cudaDepth2Disparity(_cvgmAlignedRawDepth, &_cvgm32FC1Tmp );
			//btl::cuda_util::cudaBilateralFiltering(_cvgm32FC1Tmp,_fSigmaSpace,_fSigmaDisparity,&_vcvgmPyr32FC1Tmp[0]);
			//btl::cuda_util::cudaDisparity2Depth(_vcvgmPyr32FC1Tmp[0], &_cvgmAlignedRawDepth );
			//alignDepthWithRGB( _cvmUndistDepth, &_cvmAlignedDepthL0  ); //generate _cvmDepthRGBL0
			//btl::utility::bilateralFilterInDisparity<float>(&_cvmAlignedDepthL0,_dSigmaDisparity,_dSigmaSpace);
			break;
		case PYRAMID_BILATERAL_FILTERED_IN_DISPARTY:
			PRINT(_fSigmaDisparity);
			PRINT(_fSigmaSpace);
			PRINT(_uPyrHeight);
			gpuAlignDepthWithRGB( _cvgmUndistDepth, &_cvgmAlignedRawDepth );
			btl::cuda_util::cudaDepth2Disparity(_cvgmAlignedRawDepth, &_vcvgmPyr32FC1Tmp[0] );
			btl::cuda_util::cudaBilateralFiltering(_vcvgmPyr32FC1Tmp[0],_fSigmaSpace,_fSigmaDisparity,&_vcvgmPyrDisparity[0]);
			btl::cuda_util::cudaDisparity2Depth(_vcvgmPyrDisparity[0],&_vcvgmPyrDepths[0]);
			_vcvgmPyrRGBs[0].download(_vcvmPyrRGBs[0]);
			_vcvgmPyrDepths[0].download(_vcvmPyrDepths[0]);
			gpuFastNormalEstimationGL(0,&_vcvgmPyrPts[0],&_vcvgmPyrNls[0]);
			_vcvgmPyrPts[0].download(_vcvmPyrPts[0]);
			_vcvgmPyrNls[0].download(_vcvmPyrNls[0]);
			for( unsigned int i=1; i<_uPyrHeight; i++ )
			{
				btl::cuda_util::cudaPyrDown( _vcvgmPyrDisparity[i-1],_fSigmaDisparity,&_vcvgmPyr32FC1Tmp[i]);
				btl::cuda_util::cudaBilateralFiltering(_vcvgmPyr32FC1Tmp[i],_fSigmaSpace,_fSigmaDisparity,&_vcvgmPyrDisparity[i]);
				btl::cuda_util::cudaDisparity2Depth(_vcvgmPyrDisparity[i],&_vcvgmPyrDepths[i]);
				cv::gpu::pyrDown(_vcvgmPyrRGBs[i-1],_vcvgmPyrRGBs[i]);
				_vcvgmPyrRGBs[i].download(_vcvmPyrRGBs[i]);
				_vcvgmPyrDepths[i].download(_vcvmPyrDepths[i]);
				gpuFastNormalEstimationGL(i,&_vcvgmPyrPts[i],&_vcvgmPyrNls[i]);
				_vcvgmPyrPts[i].download(_vcvmPyrPts[i]);
				_vcvgmPyrNls[i].download(_vcvmPyrNls[i]);
			}
				//btl::cuda_util::cudaDisparity2Depth(cvgmDisparity, &cvgmAlignedDepthL0 );
				//cvgmAlignedDepthL0.download(_cvmAlignedDepthL0);
				//_cvgmUndistRGB.download(_cvmUndistRGB);
				//alignDepthWithRGB2( _cvmUndistDepth, &_cvmAlignedDepthL0  ); //generate _cvmDepthRGBL0
				//bilateral filtering in disparity domain
				//btl::utility::bilateralFilterInDisparity<float>(&_cvmAlignedDepthL0,_dSigmaDisparity,_dSigmaSpace);
				//_vcvmPyramidDepths.clear();
				//_vcvmPyramidRGBs.clear();
				//_vcvmPyramidDepths.push_back(_cvmAlignedDepthL0);
				//_vcvmPyramidRGBs.push_back(_cvmUndistRGB);
				//for( unsigned int i=1; i<_uPyrHeight; i++ )
				//{
				//
				//cv::Mat cvmAlignedDepth, cvmUndistRGB;
				//depth
				//btl::utility::downSampling<float>(_vcvmPyramidDepths[i-1],&cvmAlignedDepth);
				//btl::utility::bilateralFilterInDisparity<float>(&cvmAlignedDepth,_fSigmaDisparity,_fSigmaSpace);
				//_vcvmPyramidDepths.push_back(cvmAlignedDepth);
				//rgb
				//cv::pyrDown(_vcvmPyramidRGBs[i-1],cvmUndistRGB);
				//_vcvmPyramidRGBs.push_back(cvmUndistRGB);
				//}
			break;
    }
#ifdef TIMER
// timer off
	_cT1 =  boost::posix_time::microsec_clock::local_time(); 
 	_cTDAll = _cT1 - _cT0 ;
	PRINT( _cTDAll );
#endif
	//cout << " getNextFrame() ends."<< endl;
    return;
}
void VideoSourceKinect::gpuAlignDepthWithRGB( const cv::gpu::GpuMat& cvgmUndistortDepth_ , cv::gpu::GpuMat* pcvgmAligned_)
{
	BTL_ASSERT( cvgmUndistortDepth_.type() == CV_16UC1, "VideoSourceKinect::gpuAlignDepthWithRGB() input must be unsigned short CV_16UC1");

	pcvgmAligned_->setTo(0);

	//unproject the depth map to IR coordinate
	gpuUnProjectIR		( cvgmUndistortDepth_,_fFxIR,_fFyIR,_uIR,_vIR, &_cvgmIRWorld );
	//findRange(cvgmIRWorld);
	//transform from IR coordinate to RGB coordinate
	gpuTransformIR2RGB  ( _cvgmIRWorld, &_cvgmRGBWorld );
	//findRange(cvgmRGBWorld);
	//project RGB coordinate to image to register the depth with rgb image
	//cv::gpu::GpuMat cvgmAligned_(cvgmUndistortDepth_.size(),CV_32FC1);
	gpuProjectRGB       ( _cvgmRGBWorld, pcvgmAligned_ );
	//findRange(*pcvgmAligned_);

}
void VideoSourceKinect::gpuUnProjectIR (const cv::gpu::GpuMat& cvgmUndistortDepth_, 
	const double& dFxIR_, const double& dFyIR_, const double& uIR_, const double& vIR_,
	cv::gpu::GpuMat* pcvgmIRWorld_ )
{
	cv::gpu::GpuMat& cvgmIRWorld_ = *pcvgmIRWorld_;
	btl::cuda_util::cudaUnprojectIR(cvgmUndistortDepth_, dFxIR_, dFyIR_, uIR_, vIR_, &cvgmIRWorld_);
	return;
}
void VideoSourceKinect::gpuTransformIR2RGB( const cv::gpu::GpuMat& cvgmIRWorld_, cv::gpu::GpuMat* pcvgmRGBWorld_ )
{
	cv::gpu::GpuMat& cvgmRGBWorld_ = *pcvgmRGBWorld_;
	btl::cuda_util::cudaTransformIR2RGB(cvgmIRWorld_, _aR, _aRT, &cvgmRGBWorld_);
	return;
}
void VideoSourceKinect::gpuProjectRGB( const cv::gpu::GpuMat& cvgmRGBWorld_, cv::gpu::GpuMat* pcvgmAligned_ )
{
	cv::gpu::GpuMat& cvgmAligned_ = *pcvgmAligned_;
	btl::cuda_util::cudaProjectRGB(cvgmRGBWorld_, _fFxRGB, _fFyRGB, _uRGB, _vRGB, &cvgmAligned_ );
	return;
}
void VideoSourceKinect::alignDepthWithRGB2( const cv::Mat& cvUndistortDepth_ , cv::Mat* pcvAligned_)
{
	BTL_ASSERT( cvUndistortDepth_.type() == CV_16UC1, "VideoSourceKinect::align() input must be unsigned short CV_16UC1");
	BTL_ASSERT( cvUndistortDepth_.cols == KINECT_WIDTH && cvUndistortDepth_.rows == KINECT_HEIGHT, "VideoSourceKinect::align() input must be 640x480.");
	//align( (const unsigned short*)cvUndistortDepth_.data );
	const unsigned short* pDepth = (const unsigned short*)cvUndistortDepth_.data;
	// initialize the Registered depth as NULLs
	_cvmAlignedRawDepth.setTo(0);
	cv::Mat cvmIRWorld (cvUndistortDepth_.size(),CV_32FC3);
	//unproject the depth map to IR coordinate
	unprojectIR      ( cvUndistortDepth_, &cvmIRWorld );
	//findRange(cvmIRWorld);
	cv::Mat cvmRGBWorld(cvUndistortDepth_.size(),CV_32FC3);
	//transform from IR coordinate to RGB coordinate
	transformIR2RGB  ( cvmIRWorld, &cvmRGBWorld );
	//findRange(cvmRGBWorld);
	//project RGB coordinate to image to register the depth with rgb image
	projectRGB       ( cvmRGBWorld,&(*pcvAligned_) );
	//findRange(*pcvAligned_);
	return;
}
void VideoSourceKinect::alignDepthWithRGB( const cv::Mat& cvUndistortDepth_ , cv::Mat* pcvAligned_)
{
	BTL_ASSERT( cvUndistortDepth_.type() == CV_16UC1, "VideoSourceKinect::align() input must be unsigned short CV_16UC1");
	BTL_ASSERT( cvUndistortDepth_.cols == KINECT_WIDTH && cvUndistortDepth_.rows == KINECT_HEIGHT, "VideoSourceKinect::align() input must be 640x480.");
	//align( (const unsigned short*)cvUndistortDepth_.data );
	const unsigned short* pDepth = (const unsigned short*)cvUndistortDepth_.data;
	// initialize the Registered depth as NULLs
	_cvmAlignedRawDepth.setTo(0);
	double* pM = _pRGBWorldRGB ;
	for ( int i = 0; i < KINECT_WxH; i++ )
	{
		*pM++ = 0;
		*pM++ = 0;
		*pM++ = 0;
	}
	
	//btl::utility::clearMat<float>(0,&_cvmAlignedDepthL0);

	//collecting depths
	unsigned short* pMovingPxDIR = _pPxDIR;
	//column-major  
	for ( unsigned short r = 0; r < KINECT_HEIGHT; r++ )
		for ( unsigned short c = 0; c < KINECT_WIDTH; c++ )
		{
			*pMovingPxDIR++ = c;  	    //x
			*pMovingPxDIR++ = r;        //y
			*pMovingPxDIR++ = *pDepth++;//depth
		}

	//unproject the depth map to IR coordinate
	unprojectIR      ( _pPxDIR, KINECT_WxH, _pIRWorld );
	//transform from IR coordinate to RGB coordinate
	transformIR2RGB  ( _pIRWorld, KINECT_WxH, _pRGBWorld );
	//project RGB coordinate to image to register the depth with rgb image
	projectRGB       ( _pRGBWorld, KINECT_WxH, _pRGBWorldRGB, &(*pcvAligned_) );
	//cout << "registration() end."<< std::endl;
}
void VideoSourceKinect::unprojectIR ( const cv::Mat& cvmDepth_, cv::Mat* pcvmIRWorld_)
{
	float* pWorld_ = (float*) pcvmIRWorld_->data;
	const unsigned short* pCamera_=  (const unsigned short*) cvmDepth_.data;
	for(int r = 0; r<cvmDepth_.rows; r++)
	for(int c = 0; c<cvmDepth_.cols; c++)
	{
		* ( pWorld_ + 2 ) = ( *pCamera_ + 5 ) / 1000.f; //convert to meter z 5 million meter is added according to experience. as the OpenNI
		//coordinate system is defined w.r.t. the camera plane which is 0.5 centimeters in front of the camera center
		* pWorld_		  = ( c - _uIR ) / _fFxIR * *( pWorld_ + 2 ); // + 0.0025;     //x by experience.
		* ( pWorld_ + 1 ) = ( r - _vIR ) / _fFyIR * *( pWorld_ + 2 ); // - 0.00499814; //y the value is esimated using CCalibrateKinectExtrinsics::calibDepth(
		pCamera_ ++;
		pWorld_ += 3;
	}
}
void VideoSourceKinect::unprojectIR ( const unsigned short* pCamera_, const int& nN_, double* pWorld_ )
{
	// pCamer format
	// 0 x (c) 1 y (r) 2 d
	//the pixel coordinate is defined w.r.t. camera reference, which is defined as x-left, y-downward and z-forward. It's
	//a right hand system. i.e. opencv-default reference system;
	//unit is meter
	//when rendering the point using opengl's camera reference which is defined as x-left, y-upward and z-backward. the
	//for example: glVertex3d ( Pt(0), -Pt(1), -Pt(2) ); i.e. opengl-default reference system
	for ( int i = 0; i < nN_; i++ )
	{
		* ( pWorld_ + 2 ) = ( * ( pCamera_ + 2 ) + 5 ) / 1000.; //convert to meter z 5 million meter is added according to experience. as the OpenNI
		//coordinate system is defined w.r.t. the camera plane which is 0.5 centimeters in front of the camera center
		* pWorld_		  = ( * pCamera_	     - _uIR ) / _fFxIR * *( pWorld_ + 2 ); // + 0.0025;     //x by experience.
		* ( pWorld_ + 1 ) = ( * ( pCamera_ + 1 ) - _vIR ) / _fFyIR * *( pWorld_ + 2 ); // - 0.00499814; //y the value is esimated using CCalibrateKinectExtrinsics::calibDepth(

		pCamera_ += 3;
		pWorld_ += 3;
	}

	return;
}
void VideoSourceKinect::transformIR2RGB  ( const cv::Mat& cvmIRWorld, cv::Mat* pcvmRGBWorld_ )
{
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
	for(int c = 0; c<cvmIRWorld.cols; c++)
	{
		if ( fabs ( * ( pIR_ + 2 ) ) < 0.0001 )
		{
			* pRGB_++ = 0;
			* pRGB_++ = 0;
			* pRGB_++ = 0;
		}
		else
		{
			* pRGB_++ = _aR[0] * *pIR_ + _aR[1] * *(pIR_+1) + _aR[2] * *(pIR_+2) - _aRT[0];
			* pRGB_++ = _aR[3] * *pIR_ + _aR[4] * *(pIR_+1) + _aR[5] * *(pIR_+2) - _aRT[1];
			* pRGB_++ = _aR[6] * *pIR_ + _aR[7] * *(pIR_+1) + _aR[8] * *(pIR_+2) - _aRT[2];
		}

		pIR_ += 3;
	}
}
void VideoSourceKinect::transformIR2RGB ( const double* pIR_, const int& nN_, double* pRGB_ )
{
	//_aR[0] [1] [2]
	//   [3] [4] [5]
	//   [6] [7] [8]
	//_aT[0]
	//   [1]
	//   [2]
	//  pRGB_ = _aR * ( pIR_ - _aT )
	//  	  = _aR * pIR_ - _aR * _aT
	//  	  = _aR * pIR_ - _aRT

	for ( int i = 0; i < nN_; i++ )
	{
		if ( fabs ( * ( pIR_ + 2 ) ) < 0.0001 )
		{
			* pRGB_++ = 0;
			* pRGB_++ = 0;
			* pRGB_++ = 0;
		}
		else
		{
			* pRGB_++ = _aR[0] * *pIR_ + _aR[1] * *(pIR_+1) + _aR[2] * *(pIR_+2) - _aRT[0];
			* pRGB_++ = _aR[3] * *pIR_ + _aR[4] * *(pIR_+1) + _aR[5] * *(pIR_+2) - _aRT[1];
			* pRGB_++ = _aR[6] * *pIR_ + _aR[7] * *(pIR_+1) + _aR[8] * *(pIR_+2) - _aRT[2];
		}

		pIR_ += 3;
	}

	return;
}
void VideoSourceKinect::projectRGB ( const cv::Mat& cvmRGBWorld_, cv::Mat* pcvAlignedRGB_ )
{
	//cout << "projectRGB() starts." << std::endl;
	unsigned short nX, nY;
	int nIdx1,nIdx2;
	float dX,dY,dZ;

	CHECK( CV_32FC1 == pcvAlignedRGB_->type(), "the depth pyramid level 1 must be CV_32FC1" );
	float* pWorld_ = (float*) cvmRGBWorld_.data;
	float* pDepth = (float*) pcvAlignedRGB_->data;

	for ( int i = 0; i < KINECT_WxH; i++ )
	{
		dX = *   pWorld_;
		dY = * ( pWorld_ + 1 );
		dZ = * ( pWorld_ + 2 );
		if ( fabs ( dZ ) > 0.0000001 )
		{
			// get 2D image projection in RGB image of the XYZ in the world
			nX = int( _fFxRGB * dX / dZ + _uRGB + 0.5 );
			nY = int( _fFyRGB * dY / dZ + _vRGB + 0.5 );

			// set 2D rgb XYZ
			if ( nX >= 0 && nX < KINECT_WIDTH && nY >= 0 && nY < KINECT_HEIGHT )
			{
				nIdx1= nY * KINECT_WIDTH + nX; //1 channel
				nIdx2= ( nIdx1 ) * 3; //3 channel
				pDepth    [ nIdx1   ] = float(dZ*1000);
				//PRINT( nX ); PRINT( nY ); PRINT( pWorld_ );
			}
		}
		pWorld_ += 3;
	}
	return;
}
void VideoSourceKinect::projectRGB ( double* pWorld_, const int& nN_, double* pRGBWorld_, cv::Mat* pDepthL1_ )
{
	//1.pWorld_ is the a 640*480 matrix aranged the same way as depth map
	// pRGBWorld_ is another 640*480 matrix aranged the same wey as rgb image.
	// this is much faster than the function
	// eiv2DPt = mK * vPt; eiv2DPt /= eiv2DPt(2);
	// - pWorld is using opencv convention 
	// - unit is meter
	//
	//2.calculate the centroid of the depth map

	//cout << "projectRGB() starts." << std::endl;
	unsigned short nX, nY;
	int nIdx1,nIdx2;
	_dXCentroid = _dYCentroid = _dZCentroid = 0;
	unsigned int uCount = 0;
	double dX,dY,dZ;

	CHECK( CV_32FC1 == pDepthL1_->type(), "the depth pyramid level 1 must be CV_32FC1" );
	float* pDepth = (float*) pDepthL1_->data;
	for ( int i = 0; i < nN_; i++ )
	{
		dX = *pWorld_;
		dY = * ( pWorld_ + 1 );
		dZ = * ( pWorld_ + 2 );
		if ( fabs ( dZ ) > 0.0000001 )
		{
			// get 2D image projection in RGB image of the XYZ in the world
			nX = int( _fFxRGB * dX / dZ + _uRGB + 0.5 );
			nY = int( _fFyRGB * dY / dZ + _vRGB + 0.5 );

			// set 2D rgb XYZ
			if ( nX >= 0 && nX < KINECT_WIDTH && nY >= 0 && nY < KINECT_HEIGHT )
			{
				nIdx1= nY * KINECT_WIDTH + nX; //1 channel
				nIdx2= ( nIdx1 ) * 3; //3 channel
				pDepth    [ nIdx1   ] = float(dZ*1000);
				pRGBWorld_[ nIdx2++ ] = dX ;
				pRGBWorld_[ nIdx2++ ] = dY ;
				pRGBWorld_[ nIdx2   ] = dZ ;
				//PRINT( nX ); PRINT( nY ); PRINT( pWorld_ );
				_dXCentroid += dX;
				_dYCentroid += dY;
				_dZCentroid += dZ;
				uCount ++;
			}
		}

		pWorld_ += 3;
	}
	_dXCentroid /= uCount;
	_dYCentroid /= uCount;
	_dZCentroid /= uCount;
}
void VideoSourceKinect::unprojectRGB ( const cv::Mat& cvmDepth_, double* pWorld_, int nLevel /*= 0*/ )
{
	BTL_ASSERT( CV_32FC1 == cvmDepth_.type(), "VideoSourceKinect::unprojectRGB() cvmDepth_ must be CV_32FC1" );
	BTL_ASSERT( cvmDepth_.channels()==1, "CVUtil::unprojectRGB() require the input cvmDepth is a 1-channel cv::Mat" );

	double* pM = pWorld_ ;
	// initialize the Registered depth as NULLs
	int nN = cvmDepth_.rows*cvmDepth_.cols;
	for ( int i = 0; i < nN; i++ )
	{
		*pM++ = 0;
		*pM++ = 0;
		*pM++ = 0;
	}
	// pCamer format
	// 0 x (c) 1 y (r) 2 d
	//the pixel coordinate is defined w.r.t. camera reference, which is defined as x-left, y-downward and z-forward. It's
	//a right hand system. i.e. opencv-default reference system;
	//unit is meter
	//when rendering the point using opengl's camera reference which is defined as x-left, y-upward and z-backward. the
	//for example: glVertex3d ( Pt(0), -Pt(1), -Pt(2) ); i.e. opengl-default reference system
	int nScale = 1 << nLevel;
		
	float *pDepth = (float*) cvmDepth_.data;
	
	for ( int r = 0; r < cvmDepth_.rows; r++ )
	for ( int c = 0; c < cvmDepth_.cols; c++ )
	{
		* ( pWorld_ + 2 ) = *pDepth++;
		* ( pWorld_ + 2 ) /= 1000.;
		//coordinate system is defined w.r.t. the camera plane which is 0.5 centimeters in front of the camera center
		* pWorld_		  = ( c*nScale - _uRGB ) / _fFxRGB * *( pWorld_ + 2 ); // + 0.0025;     //x by experience.
		* ( pWorld_ + 1 ) = ( r*nScale - _vRGB ) / _fFyRGB * *( pWorld_ + 2 ); // - 0.00499814; //y the value is esimated using CCalibrateKinectExtrinsics::calibDepth(

		pWorld_ += 3;
	}

	return;
}

void VideoSourceKinect::unprojectRGBGL ( const cv::Mat& cvmDepth_, const int& r,const int& c, int nLevel, cv::Mat* pcvmPts_ )
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

	for ( int r = 0; r < cvmDepth_.rows; r++ )
		for ( int c = 0; c < cvmDepth_.cols; c++ )
		{
			* ( pWorld_ + 2 ) = *pDepth++;
			* ( pWorld_ + 2 ) /= 1000.;
			//coordinate system is defined w.r.t. the camera plane which is 0.5 centimeters in front of the camera center
			* pWorld_		  = ( c*nScale - _uRGB ) / _fFxRGB * *( pWorld_ + 2 ); // + 0.0025;     //x by experience.
			* ( pWorld_ + 1 ) = ( r*nScale - _vRGB ) / _fFyRGB * *( pWorld_ + 2 ); // - 0.00499814; //y the value is esimated using CCalibrateKinectExtrinsics::calibDepth(

			pWorld_ += 3;
		}
	return;
}
void VideoSourceKinect::clonePyramid(std::vector<cv::Mat>* pvcvmRGB_, std::vector<cv::Mat>* pvcvmDepth_)
{
	if (pvcvmRGB_)
	{
		pvcvmRGB_->clear();
		for(unsigned int i=0; i<_vcvmPyrRGBs.size(); i++)
		{
			pvcvmRGB_->push_back(_vcvmPyrRGBs[i].clone());
		}
	}
	if (pvcvmDepth_)
	{
		pvcvmDepth_->clear();
		for(unsigned int i=0; i<_vcvmPyrDepths.size(); i++)
		{
			pvcvmDepth_->push_back(_vcvmPyrDepths[i].clone());
		}
	}
	return;
}
void VideoSourceKinect::cloneRawFrame( cv::Mat* pcvmRGB_, cv::Mat* pcvmDepth_ )
{
	if (pcvmRGB_)
	{
		*pcvmRGB_ = _vcvmPyrRGBs[0].clone();
	}
	if (pcvmDepth_)
	{
		*pcvmDepth_ = _cvmAlignedRawDepth.clone();
	}
}
void VideoSourceKinect::gpuFastNormalEstimationGL(const unsigned int& uLevel_,	cv::gpu::GpuMat* pcvgmPts_, cv::gpu::GpuMat* pcvgmNls_ )
{
	cv::gpu::GpuMat& cvgmDepth = _vcvgmPyrDepths[uLevel_];
	btl::cuda_util::cudaUnprojectRGBGL(cvgmDepth,_fFxRGB,_fFyRGB,_uRGB,_vRGB, uLevel_,&(*pcvgmPts_));
	btl::cuda_util::cudaFastNormalEstimationGL(*pcvgmPts_,&(*pcvgmNls_));
}








} //namespace videosource
} //namespace extra
} //namespace btl
