/**
* @file VideoSourceKinect.cpp
* @brief load of data from a kinect device 
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.0
* @date 2011-02-23
*/
#include "VideoSourceKinect.hpp"
#include "Utility.hpp"

#include <iostream>
#include <string>


#define CHECK_RC(rc, what)	\
	BTL_ASSERT(rc == XN_STATUS_OK, (what + std::string(xnGetStatusString(rc))) )

using namespace btl::utility;

namespace btl
{
namespace extra
{
namespace videosource
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
	_cvmUndistRGB.create( KINECT_HEIGHT, KINECT_WIDTH, CV_8UC3 );
	_cvmUndistDepth.create( KINECT_HEIGHT, KINECT_WIDTH, CV_16UC1 );

	_cvmAlignedDepthL0 = cv::Mat::zeros(KINECT_HEIGHT,KINECT_WIDTH,CV_32F);

	// allocate memory for later use ( registrate the depth with rgb image
	_pIRWorld = new double[ KINECT_WxHx3 ]; //XYZ w.r.t. IR camera reference system
	_pPxDIR	  = new unsigned short[ KINECT_WxHx3 ]; //pixel coordinate and depth 
	// refreshed for every frame
	_pRGBWorld    = new double[ KINECT_WxHx3 ];//X,Y,Z coordinate of depth w.r.t. RGB camera reference system
	_pRGBWorldRGB = new double[ KINECT_WxHx3 ];//aligned to RGB image of the X,Y,Z coordinate

    _ePreFiltering = RAW; 
	//definition of parameters
	_dThresholdDepth = 10;
	_dSigmaSpace = 2;
	_uPyrHeight = 1;

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

void VideoSourceKinect::getNextFrame()
{
    //get next frame
    //set as _frame
	//cout << " getNextFrame() start."<< endl;

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

	for ( unsigned int y = 0; y < _cImgMD.YRes(); y++ )
    {
        for ( unsigned int x = 0; x < _cImgMD.XRes(); x++ )
        {
            // notice that OpenCV is use BGR order
            *pRGB++ = uchar(pRGBImg->nRed);
            *pRGB++ = uchar(pRGBImg->nGreen);
            *pRGB++ = uchar(pRGBImg->nBlue);
			pRGBImg++;

			*pcvDepth++ = *pDepth++;
        }
    }

	// not fullly understand the lense distortion model used by OpenNI.
	undistortRGB( _cvmRGB, _cvmUndistRGB );
	undistortRGB( _cvmDepth, _cvmUndistDepth );
	cvtColor( _cvmUndistRGB, _cvmUndistBW, CV_RGB2GRAY );

#ifdef TIMER	
	// timer on
	_cT0 =  boost::posix_time::microsec_clock::local_time(); 
#endif
	_dSigmaDisparity = 1./600 - 1./(600+_dThresholdDepth);
    switch( _ePreFiltering )
    {
		case RAW: //default
			align( _cvmUndistDepth );
			break;
        case GAUSSIAN:
			PRINT(_dSigmaSpace);
			align( _cvmUndistDepth );
			{
				cv::Mat cvmGaussianFiltered;
				cv::GaussianBlur(_cvmAlignedDepthL0, cvmGaussianFiltered, cv::Size(0,0), _dSigmaSpace, _dSigmaSpace);
				_cvmAlignedDepthL0 = cvmGaussianFiltered;
			}
			break;
        case GAUSSIAN_C1:
			PRINT(_dThresholdDepth);
			PRINT(_dSigmaSpace);
			align( _cvmUndistDepth );
			{
				cv::Mat cvmGaussianFiltered;
				cv::GaussianBlur(_cvmAlignedDepthL0, cvmGaussianFiltered, cv::Size(0,0), _dSigmaSpace, _dSigmaSpace);
				btl::utility::filterDepth <float> ( _dThresholdDepth, (cv::Mat_<float>)cvmGaussianFiltered, (cv::Mat_<float>*)&_cvmAlignedDepthL0 );
			}
			break;
        case GAUSSIAN_C1_FILTERED_IN_DISPARTY:
			PRINT(_dSigmaDisparity);
			PRINT(_dSigmaSpace);
			align( _cvmUndistDepth );
			btl::utility::gaussianC1FilterInDisparity<float>( &_cvmAlignedDepthL0, _dSigmaDisparity, _dSigmaSpace );
            break;
		case BILATERAL_FILTERED_IN_DISPARTY:
			PRINT(_dSigmaDisparity);
			PRINT(_dSigmaSpace);
			align( _cvmUndistDepth ); //generate _cvmDepthRGBL0
			btl::utility::bilateralFilterInDisparity<float>(&_cvmAlignedDepthL0,_dSigmaDisparity,_dSigmaSpace);
			break;
		case PYRAMID_BILATERAL_FILTERED_IN_DISPARTY:
			PRINT(_dSigmaDisparity);
			PRINT(_dSigmaSpace);
			PRINT(_uPyrHeight);
	//level 0
			align( _cvmUndistDepth ); //generate _cvmDepthRGBL0
			//bilateral filtering in disparity domain
			btl::utility::bilateralFilterInDisparity<float>(&_cvmAlignedDepthL0,_dSigmaDisparity,_dSigmaSpace);
			_vcvmPyramidDepths.clear();
			_vcvmPyramidRGBs.clear();
			_vcvmPyramidDepths.push_back(_cvmAlignedDepthL0);
			_vcvmPyramidRGBs.push_back(_cvmUndistRGB);
			for( unsigned int i=1; i<_uPyrHeight; i++ )
			{
				cv::Mat cvmAlignedDepth, cvmUndistRGB;
				//depth
				btl::utility::downSampling<float>(_vcvmPyramidDepths[i-1],&cvmAlignedDepth);
				btl::utility::bilateralFilterInDisparity<float>(&cvmAlignedDepth,_dSigmaDisparity,_dSigmaSpace);
				_vcvmPyramidDepths.push_back(cvmAlignedDepth);
				//rgb
				cv::pyrDown(_vcvmPyramidRGBs[i-1],cvmUndistRGB);
				_vcvmPyramidRGBs.push_back(cvmUndistRGB);
			}
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

void VideoSourceKinect::align( const cv::Mat& cvUndistortDepth_ )
{
	BTL_ASSERT( cvUndistortDepth_.type() == CV_16UC1, "VideoSourceKinect::align() input must be unsigned short CV_16UC1");
	BTL_ASSERT( cvUndistortDepth_.cols == KINECT_WIDTH && cvUndistortDepth_.rows == KINECT_HEIGHT, "VideoSourceKinect::align() input must be 640x480.")
	//align( (const unsigned short*)cvUndistortDepth_.data );
	const unsigned short* pDepth = (const unsigned short*)cvUndistortDepth_.data;
	// initialize the Registered depth as NULLs
	double* pM = _pRGBWorldRGB ;
	for ( int i = 0; i < KINECT_WxH; i++ )
	{
		*pM++ = 0;
		*pM++ = 0;
		*pM++ = 0;
	}

	btl::utility::clearMat<float>(0,&_cvmAlignedDepthL0);

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
	projectRGB       ( _pRGBWorld, KINECT_WxH, _pRGBWorldRGB, &_cvmAlignedDepthL0 );

	//cout << "registration() end."<< std::endl;
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
		* pWorld_		  = ( * pCamera_	     - _uIR ) / _dFxIR * *( pWorld_ + 2 ); // + 0.0025;     //x by experience.
		* ( pWorld_ + 1 ) = ( * ( pCamera_ + 1 ) - _vIR ) / _dFyIR * *( pWorld_ + 2 ); // - 0.00499814; //y the value is esimated using CCalibrateKinectExtrinsics::calibDepth(

		pCamera_ += 3;
		pWorld_ += 3;
	}

	return;
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
		if ( abs ( * ( pIR_ + 2 ) ) < 0.0001 )
		{
			* pRGB_++ = 0;
			* pRGB_++ = 0;
			* pRGB_++ = 0;
		}
		else
		{
			* pRGB_++ = _aR[0] * *pIR_ + _aR[1] * * ( pIR_ + 1 ) + _aR[2] * * ( pIR_ + 2 ) - _aRT[0];
			* pRGB_++ = _aR[3] * *pIR_ + _aR[4] * * ( pIR_ + 1 ) + _aR[5] * * ( pIR_ + 2 ) - _aRT[1];
			* pRGB_++ = _aR[6] * *pIR_ + _aR[7] * * ( pIR_ + 1 ) + _aR[8] * * ( pIR_ + 2 ) - _aRT[2];
		}

		pIR_ += 3;
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
			nX = int( _dFxRGB * dX / dZ + _uRGB + 0.5 );
			nY = int( _dFyRGB * dY / dZ + _vRGB + 0.5 );

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
		* ( pWorld_ + 2 ) = cvmDepth_.at<float>(r,c);//*pDepth++;
		* ( pWorld_ + 2 ) /= 1000.;
		//coordinate system is defined w.r.t. the camera plane which is 0.5 centimeters in front of the camera center
		* pWorld_		  = ( c*nScale - _uRGB ) / _dFxRGB * *( pWorld_ + 2 ); // + 0.0025;     //x by experience.
		* ( pWorld_ + 1 ) = ( r*nScale - _vRGB ) / _dFyRGB * *( pWorld_ + 2 ); // - 0.00499814; //y the value is esimated using CCalibrateKinectExtrinsics::calibDepth(

		pWorld_ += 3;
	}

	return;
}

void VideoSourceKinect::clonePyramid(std::vector<cv::Mat>* pvcvmRGB_, std::vector<cv::Mat>* pvcvmDepth_)
{
	if (pvcvmRGB_)
	{
		pvcvmRGB_->clear();
		for(unsigned int i=0; i<_vcvmPyramidRGBs.size(); i++)
		{
			pvcvmRGB_->push_back(_vcvmPyramidRGBs[i].clone());
		}
	}
	if (pvcvmDepth_)
	{
		pvcvmDepth_->clear();
		for(unsigned int i=0; i<_vcvmPyramidDepths.size(); i++)
		{
			pvcvmDepth_->push_back(_vcvmPyramidDepths[i].clone());
		}
	}
	return;
}

void VideoSourceKinect::cloneFrame( cv::Mat* pcvmRGB_, cv::Mat* pcvmDepth_ )
{
	if (pcvmRGB_)
	{
		*pcvmRGB_ = _cvmUndistRGB.clone();
	}
	if (pcvmDepth_)
	{
		*pcvmDepth_ = _cvmAlignedDepthL0.clone();
	}
}





} //namespace videosource
} //namespace extra
} //namespace btl
