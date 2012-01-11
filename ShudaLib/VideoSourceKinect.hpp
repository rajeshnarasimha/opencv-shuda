#ifndef BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCEKINECT
#define BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCEKINECT
/**
* @file VideoSourceKinect.hpp
* @brief APIs for load of data from a kinect devices
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.1 
* 3-17 depth generator added
* @date 2011-03-17
*/
#include <stdexcept>
#include "calibratekinect.hpp"
#include <XnCppWrapper.h>
#include <opencv/highgui.h>
using namespace xn;

namespace btl{
namespace extra{
namespace videosource{

#define KINECT_WIDTH 640
#define KINECT_HEIGHT 480

#define KINECT_WxH 307200
#define KINECT_WxH_L1 76800 
#define KINECT_WxH_L2 19200

#define KINECT_WxHx3 921600
#define KINECT_WxHx3_L1 230400 
#define KINECT_WxHx3_L2 57600

struct Frame
{
 	cv::Mat       _cvImage;
	cv::Mat 	  _cvmBW;
    cv::Mat       _cvDepth;
};

//CCalibrateKinect is help to load camera parameters from 
class VideoSourceKinect : public CCalibrateKinect
{
public:
    VideoSourceKinect();
    virtual ~VideoSourceKinect();

    void getNextFrame();

    // 1. need to call getNextFrame() before hand
    // 2. RGB color channel (rather than BGR as used by cv::imread())
    const cv::Mat&            cvRGB()     const { return  _cvmUndistRGBL0; }
    const cv::Mat*            cvRGBPtr()  const { return &_cvmUndistRGBL0; }
    const cv::Mat&            cvDepth()   const { return  _cvmUndistFilteredDepth; }
    const cv::Mat*            cvDepthPtr()const { return &_cvmUndistFilteredDepth; }
		  cv::Mat*            cvDepthPtr()      { return &_cvmUndistFilteredDepth; }
	const cv::Mat& 			  cvBW()      const { return  _cvmUndistBW; }
	const double* 			  alignedDepth() const {return _pRGBWorldRGBL0; }	// depth coordnate aligned with RGB camera
	//opencv convention
	void centroid( Eigen::Vector3d* peivCentroid_ ) const 
	{
		(*peivCentroid_)(0) = _dXCentroid;
		(*peivCentroid_)(1) = _dYCentroid;
		(*peivCentroid_)(2) = _dZCentroid;
	}
	//opengl convention
	void centroidGL( Eigen::Vector3d* peivCentroid_ ) const 
	{
		(*peivCentroid_)(0) =   _dXCentroid;
		(*peivCentroid_)(1) = - _dYCentroid;
		(*peivCentroid_)(2) = - _dZCentroid;
	}

	void unprojectRGB ( const cv::Mat& cvmDepth_, double* pWorld_, int nLevel = 0 );
	void align( const cv::Mat& cvUndistortDepth_ );
	// convert the depth map/ir camera to be aligned with the rgb camera
	void align( const unsigned short* pDepth_ ); // the 
	void unprojectIR( const unsigned short* pCamera_,const int& nN_, double* pWorld_ );
	void transformIR2RGB( const double* pIR_,const int& nN_, double* pRGB_ );
	void projectRGB ( double* pWorld_, const int& nN_, double* pRGBWorld_, cv::Mat* pDepthL1_ );

    struct Exception : public std::runtime_error
    {
       Exception(const std::string& str) : std::runtime_error(str) {}
    };

    enum {  C1_CONTINUITY, GAUSSIAN_C1, DISPARIT_GAUSSIAN_C1, RAW, NEW_GAUSSIAN, NEW_BILATERAL, NONE, NEW_DEPTH } _eMethod;

	void cloneDepth(double* pDepth_)
	{
		double* pM = _pRGBWorldRGBL0;
		for( int i=0; i< 921600;i++ )
		{
			*pDepth_++ = *pM++; 
		}
	}
protected:

    Context        _cContext;
    ImageGenerator _cImgGen;
    ImageMetaData  _cImgMD;

    DepthGenerator _cDepthGen;
    DepthMetaData  _cDepthMD;

    cv::Mat       _cvmRGB;
	cv::Mat       _cvmUndistRGBL0;
	cv::Mat		  _cvmUndistRGBL1;
	cv::Mat       _cvmUndistRGBL2;

    cv::Mat       _cvmDepth;
	cv::Mat 	  _cvmUndistDepth;
	cv::Mat 	  _cvmUndistFilteredDepth;
	//depth pyramid
	cv::Mat		  _cvmAlignedDepthL0;//640*480
	cv::Mat		  _cvmAlignedDepthL1;//320*240
	cv::Mat		  _cvmAlignedDepthL2;//160*120

	cv::Mat 	  _cvmUndistBW;


	// temporary variables allocated in constructor and released in destructor
	unsigned short* _pPxDIR; //2D coordinate along with depth for ir image, column-major
	double*  _pIRWorld;      //XYZ w.r.t. IR camera reference system 
	double*  _pRGBWorld;     //XYZ w.r.t. RGB camera but indexed in IR image
	// refreshed for every frame
	//X,Y,Z coordinate of depth w.r.t. RGB camera reference system
	//in the format of the RGB image
	double*  _pRGBWorldRGBL0; 
	double*  _pRGBWorldRGBL1;
	double*  _pRGBWorldRGBL2;

	double _dXCentroid, _dYCentroid, _dZCentroid; // the centroid of all depth point defined in RGB camera system (opencv-default camera reference system convention)

public:
    std::vector< Eigen::Vector3d > _vPts;
    std::vector< Eigen::Vector3d > _vNormals;
    std::vector<const unsigned char*> _vColors;
    //cv::Mat _cvColor;
	//paramters
	double _dThresholdDepth; //threshold for filtering depth
	double _dSigmaSpace; //degree of blur for the bilateral filter
	double _dSigmaDisparity;

};

} //namespace videosource
} //namespace extra
} //namespace btl


namespace btl{
namespace extra{

using videosource::VideoSourceKinect;

} //namespace extra
} //namespace btl


#endif //BTL_EXTRA_VIDEOSOURCE_VIDEOSOURCEKINECT
