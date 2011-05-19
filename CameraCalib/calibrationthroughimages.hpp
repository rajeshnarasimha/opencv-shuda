#ifndef CALIBRATION_THROUGH_IMAGES
#define CALIBRATION_THROUGH_IMAGES

#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <btl/Utility/Converters.hpp>
//#include <btl/Camera/CameraModel.hpp>
using namespace btl;
using namespace utility;

namespace shuda
{
class CCalibrationThroughImages
{

public:
    CCalibrationThroughImages()
    {
        _NUM_CORNERS_X = 11;
        _NUM_CORNERS_Y = 7;
        _X = .02f;
        _Y = .02f;
        _dNear = 0.01;
        _dFar  = 10.;
    }

    void main(const boost::filesystem::path& cFullPath_ );
    //btl::camera::CameraModelPinhole btlCameraModelPinHoleK() const; 
    //retriever:
    double                          near() const {return _dNear;}
    double                          far()  const {return _dFar;} 
    const cv::Mat&                  image (unsigned int nView_) const {return _vImages[nView_];}
    const cv::Mat&                  undistortedImg(unsigned int nView_) const {return _vUndistortedImages[nView_];}
    const cv::Mat&                  depth (unsigned int nView_) const {return _vDepthMaps[nView_];}
    const cv::Mat&                  undistortedDepth(unsigned int nView_) const {return _vUndistortedDepthMaps[nView_];}
    const std::vector<cv::Point3f>& pattern(unsigned int nView_)const {return _vv3DCorners[nView_];}
    const Eigen::Vector2i&          imageResolution()           const {return _vImageResolution;}
    Eigen::Matrix3d                 eiMatK()                    const {Eigen::Matrix3d eiMatK; eiMatK << _mK; return eiMatK;}
    const cv::Mat&                  cvMatK()                    const {return _mK;}
    Eigen::Matrix3d                 eiMatDepthK()               const {Eigen::Matrix3d eiMatDepthK; eiMatDepthK << _mDepthK; return eiMatDepthK;}
    const cv::Mat&                  cvMatDepthK()               const {return _mDepthK;}


    const cv::Mat&                  cvMatDistor()               const {return _mDistCoeffs;}
    Eigen::Matrix3d                 eiMatR(unsigned int nView_) const {Eigen::Matrix3d eiMatR; eiMatR << cv::Mat_<double>(_vmRotationMatrices[nView_]); return eiMatR;}
    const cv::Mat&                  cvMatR(unsigned int nView_) const {return _vmRotationMatrices[nView_];}
    const cv::Mat&                  cvVecR(unsigned int nView_) const {return _vmRotationVectors[nView_];}
    Eigen::Vector3d                 eiVecT(unsigned int nView_) const 
    {
        Eigen::Vector3d eiVecT; 
        cv::Mat mVec = _vmTranslationVectors[nView_].t();
        eiVecT << cv::Mat_<double>(mVec);
        return eiVecT;
    }
    const cv::Mat&                  cvMatT(unsigned int nView_) const {return _vmTranslationVectors[nView_];}
    const unsigned int&             views()                     const {return _uViews;}
    const string                    imagePathName(unsigned int nView_) const {return _vstrImagePathName[nView_];}
    const string                    depthPathName(unsigned int nView_) const {return _vstrDepthPathName[nView_];}

	Eigen::Vector2d distortPoint(const Eigen::Vector2d& undistorted)
	{
 	   double xu = undistorted(0);
 	   double yu = undistorted(1);
 	   double xun = _mInvK(0,0)*xu + _mInvK(0,1)*yu + _mInvK(0,2);
 	   double yun = _mInvK(1,0)*xu + _mInvK(1,1)*yu + _mInvK(1,2);
 	   double x2 = xun * xun;
 	   double y2 = yun * yun;
 	   double xy = xun * yun;
 	   double r2 = x2 + y2;
 	   double r4 = r2 * r2;
 	   double r6 = r4 * r2;
 	   double _k1= _mDistCoeffs(0);
 	   double _k2= _mDistCoeffs(1);
 	   double _k3= _mDistCoeffs(2);
 	   double _k4= _mDistCoeffs(3);
 	   double _k5= _mDistCoeffs(4);
 	   double radialDistortion(1.0 + _k1*r2 + _k2*r4 + _k5*r6);
 	   double tangentialDistortionX = (2 * _k3 * xy) + (_k4 * (r2 + 2 * x2));
 	   double tangentialDistortionY = (_k3 * (r2 + 2 * y2)) + (2 * _k4 * xy);
 	   double xdn = (xun * radialDistortion) + tangentialDistortionX;
 	   double ydn = (yun * radialDistortion) + tangentialDistortionY;
 	   double xd = _mK(0,0)*xdn + _mK(0,1)*ydn + _mK(0,2);
 	   double yd = _mK(1,0)*xdn + _mK(1,1)*ydn + _mK(1,2);
 	   Eigen::Vector2d distorted(xd, yd);
 	   return distorted;
	}

    Eigen::Vector2d distortDepthPoint(const Eigen::Vector2d& undistorted)
	{
 	   double xu = undistorted(0);
 	   double yu = undistorted(1);
 	   double xun = _mInvDepthK(0,0)*xu + _mInvDepthK(0,1)*yu + _mInvDepthK(0,2);
 	   double yun = _mInvDepthK(1,0)*xu + _mInvDepthK(1,1)*yu + _mInvDepthK(1,2);
 	   double x2 = xun * xun;
 	   double y2 = yun * yun;
 	   double xy = xun * yun;
 	   double r2 = x2 + y2;
 	   double r4 = r2 * r2;
 	   double r6 = r4 * r2;
 	   double _k1= _mDepthDistCoeffs(0);
 	   double _k2= _mDepthDistCoeffs(1);
 	   double _k3= _mDepthDistCoeffs(2);
 	   double _k4= _mDepthDistCoeffs(3);
 	   double _k5= _mDepthDistCoeffs(4);
 	   double radialDistortion(1.0 + _k1*r2 + _k2*r4 + _k5*r6);
 	   double tangentialDistortionX = (2 * _k3 * xy) + (_k4 * (r2 + 2 * x2));
 	   double tangentialDistortionY = (_k3 * (r2 + 2 * y2)) + (2 * _k4 * xy);
 	   double xdn = (xun * radialDistortion) + tangentialDistortionX;
 	   double ydn = (yun * radialDistortion) + tangentialDistortionY;
 	   double xd = _mDepthK(0,0)*xdn + _mDepthK(0,1)*ydn + _mDepthK(0,2);
 	   double yd = _mDepthK(1,0)*xdn + _mDepthK(1,1)*ydn + _mDepthK(1,2);
 	   Eigen::Vector2d distorted(xd, yd);
 	   return distorted;
	}

	double rawDepthToMeters ( int nRawDepth_ )
	{
	    if ( nRawDepth_ < 2047 )
	    {
	        return 1.0 / ( nRawDepth_ * -0.0030711016 + 3.3309495161 );
	    }
	
	    return 0;
    }
    double depthInMethers ( int nX_, int nY_, unsigned int uView_ )
    {
        const Mat& cvDepth = depth( uView_ ); 
        unsigned char* pDepth = (unsigned char*) cvDepth.data;

    }

private:
    void initializeDepthIntrinsics();
    void loadImages ( const boost::filesystem::path& cFullPath_ );
    void locate2DCorners();
    void define3DCorners ( const float& fX_, const float& fY_ );
    void calibrate ();
/**
* @brief convert rotation vectors into rotation matrices using cv::
*
* @param
* @param argv
*/
    void convertRV2RM();
    void undistortImages();
    void undistortDepthMaps();
    void save();
    void load();

private:
// others
    unsigned int    _uViews;
    Eigen::Vector2i _vImageResolution; //x,y;

// images
    std::vector< cv::Mat > _vImages;
    std::vector< cv::Mat > _vUndistortedImages;
    std::vector< cv::Mat > _vDepthMaps;
    std::vector< cv::Mat > _vUndistortedDepthMaps;
    std::vector< std::string > _vstrImagePathName; //serialized
    std::vector< std::string > _vstrDepthPathName;
// 2D-3D correspondences serialized
    std::vector< std::vector<cv::Point2f> > _vv2DCorners;
    std::vector< std::vector<cv::Point3f> > _vv3DCorners;
// rgb camera intrinsics serialized
    cv::Mat_<double> _mK; 
    cv::Mat_<double> _mInvK; // not serialized
    cv::Mat_<double> _mDistCoeffs;
// depth camera intrinsics not serialized
    cv::Mat_<double> _mDepthK; 
    cv::Mat_<double> _mInvDepthK; 
    cv::Mat_<double> _mDepthDistCoeffs;
    Eigen::Matrix3d  _eimRelativeRotation;
    Eigen::Vector3d  _eivRelativeTranslation;
// for opengl
    double _dNear;
    double _dFar;

// camera extrinsics serialized (double type)
    std::vector< cv::Mat > _vmRotationVectors;
    std::vector< cv::Mat > _vmRotationMatrices;
    std::vector< cv::Mat > _vmTranslationVectors;
// patern constancy serialized
     int _NUM_CORNERS_X;
     int _NUM_CORNERS_Y;
     float _X;
     float _Y;
};
}//shuda

#endif
