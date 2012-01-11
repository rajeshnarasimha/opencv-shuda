/**
* @file VideoSourceKinect.cpp
* @brief load of data from a kinect device 
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.0
* @date 2011-02-23
*/
#include "VideoSourceKinect.hpp"
#include "Converters.hpp"

#include <iostream>
#include <cassert>
#include <string>


#define CHECK_RC(rc, what)											            \
	if (rc != XN_STATUS_OK)											            \
	{																            \
		throw Exception(what + std::string(xnGetStatusString(rc)));\
	}


using namespace std;
using namespace btl;
using namespace utility;

namespace btl
{
namespace extra
{
namespace videosource
{

VideoSourceKinect::VideoSourceKinect ()
:CCalibrateKinect()
{
    cout << "  VideoSource_Linux: Opening Kinect..." << endl;

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
	_cvmUndistRGBL0.create( KINECT_HEIGHT, KINECT_WIDTH, CV_8UC3 );
	_cvmUndistDepth.create( KINECT_HEIGHT, KINECT_WIDTH, CV_16UC1 );
	_cvmUndistFilteredDepth.create( KINECT_HEIGHT, KINECT_WIDTH, CV_16UC1 );

	_cvmAlignedDepthL0 = cv::Mat::zeros(KINECT_HEIGHT,KINECT_WIDTH,CV_32F);
	_cvmAlignedDepthL1 = cv::Mat::zeros(KINECT_HEIGHT/2,KINECT_WIDTH/2,CV_32F);
	_cvmAlignedDepthL2 = cv::Mat::zeros(KINECT_HEIGHT/4,KINECT_WIDTH/4,CV_32F);

	// allocate memory for later use ( registrate the depth with rgb image
	_pIRWorld = new double[ KINECT_WxHx3 ]; //XYZ w.r.t. IR camera reference system
	_pPxDIR	  = new unsigned short[ KINECT_WxHx3 ]; //pixel coordinate and depth 
	// refreshed for every frame
	_pRGBWorld    = new double[ KINECT_WxHx3 ];//X,Y,Z coordinate of depth w.r.t. RGB camera reference system
	_pRGBWorldRGBL0 = new double[ KINECT_WxHx3 ];//aligned to RGB image of the X,Y,Z coordinate
	_pRGBWorldRGBL1 = new double[ KINECT_WxHx3_L1 ];
	_pRGBWorldRGBL2 = new double[ KINECT_WxHx3_L2 ];

    _eMethod = NEW_BILATERAL; 
	//definition of parameters
	_dThresholdDepth = 10;
	_dSigmaSpace = 2;

	cout << " Done. " << endl;
}

VideoSourceKinect::~VideoSourceKinect()
{
    _cContext.Release();
	delete [] _pIRWorld;
	delete [] _pPxDIR;
	delete [] _pRGBWorld;
	delete [] _pRGBWorldRGBL0;
	delete [] _pRGBWorldRGBL1;
	delete [] _pRGBWorldRGBL2;
}

void VideoSourceKinect::getNextFrame()
{
    //get next frame
    //set as _frame
	//cout << " getNextFrame() start."<< endl;
	_vColors.clear();
	_vPts.clear();
	_vNormals.clear();
	
// timer on
//	_cT0 =  boost::posix_time::microsec_clock::local_time(); 

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
	undistortRGB( _cvmRGB, _cvmUndistRGBL0 );
	undistortRGB( _cvmDepth, _cvmUndistDepth );
	cvtColor( _cvmUndistRGBL0, _cvmUndistBW, CV_RGB2GRAY );

    cv::Mat cvDisparity( _cvmUndistDepth.rows, _cvmUndistDepth.cols, CV_32F );
    cv::Mat_<float> cvFilterDisparity( _cvmUndistDepth.rows, _cvmUndistDepth.cols, CV_32F );
    cv::Mat cvThersholdDisparity( _cvmUndistDepth.rows, _cvmUndistDepth.cols, CV_32F );

    cv::Mat_<unsigned short> cvmFilter(_cvmUndistDepth.rows, _cvmUndistDepth.cols, CV_16U );

    switch( _eMethod )
    {
		case NONE: //default
			align( _cvmUndistDepth );
			normalEstimationGL<double, unsigned char>( alignedDepth(), _cvmUndistRGBL0.data, _cvmUndistRGBL0.rows, _cvmUndistRGBL0.cols, &_vColors, &_vPts, &_vNormals );
			break;
        case RAW:
    	    // register the depth with rgb image
    	    align( _cvmUndistDepth );
			normalEstimationGLPCL<double, unsigned char>( alignedDepth(), _cvmUndistRGBL0.data, _cvmUndistRGBL0.rows, _cvmUndistRGBL0.cols, &_vColors, &_vPts, &_vNormals );
            break;
        case C1_CONTINUITY:
            btl::utility::filterDepth <unsigned short> ( _dThresholdDepth, (cv::Mat_<unsigned short>)_cvmUndistDepth, (cv::Mat_<unsigned short>*)&_cvmUndistFilteredDepth );
    	    // register the depth with rgb image
    	    align( _cvmUndistFilteredDepth );
			normalEstimationGLPCL<double, unsigned char>( alignedDepth(), _cvmUndistRGBL0.data, _cvmUndistRGBL0.rows, _cvmUndistRGBL0.cols, &_vColors, &_vPts, &_vNormals );
            break;
        case GAUSSIAN_C1:
        	// filter out depth noise
            cv::GaussianBlur(_cvmUndistDepth, cvmFilter, cv::Size(0,0), _dSigmaSpace, _dSigmaSpace); // filter size has to be an odd number.
	        btl::utility::filterDepth <unsigned short> ( _dThresholdDepth, (cv::Mat_<unsigned short>)cvmFilter, (cv::Mat_<unsigned short>*)&_cvmUndistFilteredDepth );
    	    // register the depth with rgb image
    	    align( _cvmUndistFilteredDepth );
			normalEstimationGLPCL<double, unsigned char>( alignedDepth(), _cvmUndistRGBL0.data, _cvmUndistRGBL0.rows, _cvmUndistRGBL0.cols, &_vColors, &_vPts, &_vNormals );
            break;
        case DISPARIT_GAUSSIAN_C1:
            convert2DisparityDomain< unsigned short >( _cvmUndistDepth, &(cv::Mat_<float>)cvDisparity );
            cv::GaussianBlur(cvDisparity, cvFilterDisparity, cv::Size(0,0), _dSigmaSpace, _dSigmaSpace);
            _dSigmaDisparity = 1./600. - 1./(600.+_dThresholdDepth);
            btl::utility::filterDepth <float> ( _dSigmaDisparity, ( cv::Mat_<float>)cvFilterDisparity, ( cv::Mat_<float>*)&cvThersholdDisparity );
    	    btl::utility::convert2DepthDomain< unsigned short >( cvThersholdDisparity, &_cvmUndistFilteredDepth, CV_16UC1 );
              // register the depth with rgb image
    	    align( _cvmUndistFilteredDepth );
			normalEstimationGLPCL<double, unsigned char>( alignedDepth(), _cvmUndistRGBL0.data, _cvmUndistRGBL0.rows, _cvmUndistRGBL0.cols, &_vColors, &_vPts, &_vNormals );
            break;
        case NEW_GAUSSIAN:
            // filter out depth noise
			// apply some bilateral gaussian filtering
            cv::GaussianBlur(_cvmUndistDepth, cvmFilter, cv::Size(0,0), _dSigmaSpace, _dSigmaSpace); // filter size has to be an odd number.
            align( cvmFilter );
            normalEstimationGL<double, unsigned char>( alignedDepth(), _cvmUndistRGBL0.data, _cvmUndistRGBL0.rows, _cvmUndistRGBL0.cols, &_vColors, &_vPts, &_vNormals );
            break;
		case NEW_BILATERAL:
			// filter out depth noise
			// apply some bilateral gaussian filtering
			btl::utility::convert2DisparityDomain< unsigned short >( _cvmUndistDepth, &(cv::Mat_<float>)cvDisparity );
			_dSigmaDisparity = 1./600. - 1./(600.+_dThresholdDepth);
			cv::bilateralFilter(cvDisparity, cvThersholdDisparity,0, _dSigmaDisparity, _dSigmaSpace); // filter size has to be an odd number.
			PRINT(_dThresholdDepth);
			PRINT(_dSigmaDisparity);
			btl::utility::convert2DepthDomain< unsigned short >( cvThersholdDisparity, &_cvmUndistFilteredDepth, CV_16UC1 );
			align( _cvmUndistFilteredDepth );
			normalEstimationGL<double, unsigned char>( alignedDepth(), _cvmUndistRGBL0.data, _cvmUndistRGBL0.rows, _cvmUndistRGBL0.cols, &_vColors, &_vPts, &_vNormals );
			break;
		case NEW_DEPTH:
			
			_dSigmaDisparity = 1./600 - 1./(600+_dThresholdDepth);
			PRINT(_dSigmaDisparity);
			PRINT(_dSigmaSpace);
	//level 0
			align( _cvmUndistDepth ); //generate _cvmDepthRGBL0
			//bilateral filtering in disparity domain
			btl::utility::bilateralFilterInDisparity<float>(&_cvmAlignedDepthL0,_dSigmaDisparity,_dSigmaSpace);
		//get normals L0
			//unprojectRGB ( _cvmDepthRGBL0, _pRGBWorldRGB );
			//normalEstimationGL<double, unsigned char>( registeredDepth(), _cvUndistImage.data, _cvUndistImage.rows, _cvUndistImage.cols, &_vColors, &_vPts, &_vNormals );
	//level 1
			btl::utility::downSampling<float>(_cvmAlignedDepthL0,&_cvmAlignedDepthL1);
			cv::pyrDown(_cvmUndistRGBL0,_cvmUndistRGBL1);
		//bilateral filtering in disparity domain
			btl::utility::bilateralFilterInDisparity<float>(&_cvmAlignedDepthL1,_dSigmaDisparity,_dSigmaSpace);
		//get normals L1
			//unprojectRGB ( _cvmDepthRGBL1, _pRGBWorldRGBL1, 1 );//float to double
			//normalEstimationGL<double, unsigned char>( _pRGBWorldRGBL1, _cvmUndistDepthL1.data, _cvmUndistDepthL1.rows, _cvmUndistDepthL1.cols, &_vColors, &_vPts, &_vNormals );
	//level 2
			btl::utility::downSampling<float>(_cvmAlignedDepthL1,&_cvmAlignedDepthL2);
			cv::pyrDown(_cvmUndistRGBL1,_cvmUndistRGBL2);
		//bilateral filtering in disparity domain
			btl::utility::bilateralFilterInDisparity<float>(&_cvmAlignedDepthL2,_dSigmaDisparity,_dSigmaSpace);
		//get normals L2
			unprojectRGB ( _cvmAlignedDepthL2, _pRGBWorldRGBL2, 2 );//float to double
			normalEstimationGL<double, unsigned char>( _pRGBWorldRGBL2, _cvmUndistRGBL2.data, _cvmUndistRGBL2.rows, _cvmUndistRGBL2.cols, &_vColors, &_vPts, &_vNormals );

			break;
    }

// timer off
//	_cT1 =  boost::posix_time::microsec_clock::local_time(); 
// 	_cTDAll = _cT1 - _cT0 ;
//	PRINT( _cTDAll );

	//cout << " getNextFrame() ends."<< endl;
    return;
}

void VideoSourceKinect::align( const cv::Mat& cvUndistortDepth_ )
{
	BTL_ASSERT( cvUndistortDepth_.type() == CV_16UC1, "VideoSourceKinect::align() input must be unsigned short CV_16UC1");
	align( (const unsigned short*)cvUndistortDepth_.data );
}

void VideoSourceKinect::align ( const unsigned short* pDepth_ )
{
	// initialize the Registered depth as NULLs
	double* pM = _pRGBWorldRGBL0 ;
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
		*pMovingPxDIR++ = *pDepth_++;//depth
	}

	//unproject the depth map to IR coordinate
	unprojectIR      ( _pPxDIR, KINECT_WxH, _pIRWorld );
	//transform from IR coordinate to RGB coordinate
	transformIR2RGB  ( _pIRWorld, KINECT_WxH, _pRGBWorld );
	//project RGB coordinate to image to register the depth with rgb image
	projectRGB       ( _pRGBWorld, KINECT_WxH, _pRGBWorldRGBL0, &_cvmAlignedDepthL0 );

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

/*
void VideoSourceKinect::buildPyramid ()
{

}
*/

void VideoSourceKinect::unprojectRGB ( const cv::Mat& cvmDepth_, double* pWorld_, int nLevel /*= 0*/ )
{
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

	CHECK( CV_32FC1 == cvmDepth_.type(), "VideoSourceKinect::unprojectRGB() cvmDepth_ must be CV_32FC1" );
	float *pDepth = (float*) cvmDepth_.data;
	
	for ( unsigned int r = 0; r < cvmDepth_.rows; r++ )
	for ( unsigned int c = 0; c < cvmDepth_.cols; c++ )
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


} //namespace videosource
} //namespace extra
} //namespace btl
