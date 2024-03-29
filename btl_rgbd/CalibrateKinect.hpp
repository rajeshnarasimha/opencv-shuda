#ifndef CALIBRATION_THROUGH_IMAGES
#define CALIBRATION_THROUGH_IMAGES
/**
* @file calibratekinect.hpp
* @brief This code is used to calibrate the kinect intrinsics using a checkboard pattern. 
* The input of the calibration includes:
* 1. a sequence of images with a checkboard clearly visible. The sequence of images must contain equal number of  ir
* images and rgb images. the ir images of pattern are taken when the ir projector is coverred and other infrayed light source is using. 
* 2. size of the grid of pattern measured in meter. 
* 3. number of corners in X direction of in Y direction
* The kinect intrinsical parameters to be calibrated including:
* 1. the intrinsics of rgb camera: focal length, principle points and distortion coefficients
* 2. the intrinsics of ir camera:  focal length, principle points and distortion coefficients
* 3. the relative position of ir camera w.r.t. rgb camera including a rotation and a translation. The transform from rgb
* camera coordinate to ir camera coordinate can be formulated as following:
*    - X_ir = R * X_rgb + T
*    where X_ir is a 3D point defined in ir camera coordinate and X_rgb is defined in rgb camera coordinate. R and T are
*    the relative rotation and translation.
*    Given that the rotation and translation from world coordnate to rgb camera coordinate is R_rgb and T_rgb; Similarly, the
*    rotation and translation from world to ir camera coordinate is R_ir and T_ir. 
*    - X_rgb = R_rgb * X + T_rgb 
*    - X_ir  = R_ir  * X + T_ir
*    Therefore the relative pose (R & T) can be calculated by:
*    - X_ir  = R * ( R_rgb * X + T_rgb ) + T =>
*    - X_ir  = R * R_rgb * X + R * T_rgb + T => 
*    - R_ir  = R * R_rgb
*    - T_ir  = R * T_rgb + T
* Other notices:
* 1. the computation of the relative rotation and translation is basically the average between the difference of 
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.0
* @date 2011-03-27
*/

#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "Utility.hpp"
#include <GL/freeglut.h>

namespace btl
{
namespace kinect
{

using namespace btl;
using namespace utility;



class CCalibrateKinect
{

public:
	CCalibrateKinect();
 	~CCalibrateKinect();
	virtual void parseControlYAML();
    virtual void mainFunc(const boost::filesystem::path& cFullPath_ );

    //btl::camera::CameraModelPinhole btlCameraModelPinHoleK() const; 
	
//retriever:
    const cv::Mat&                  rgbImage (unsigned int nView_)      const {return _vRGBs[nView_];}
    const cv::Mat&                  rgbUndistorted(unsigned int nView_) const {return _vRGBUndistorted[nView_];}
	const cv::Mat&                  irImage (unsigned int nView_)       const {return _vIRs[nView_];}
    const cv::Mat&                  irUndistorted(unsigned int nView_)  const {return _vIRUndistorted[nView_];}
    const std::vector<cv::Point3f>&  pattern()        					const {return _vPatterCorners3D;}
    const Eigen::Vector2i&             imageResolution()                   const {return _vImageResolution;}
    const Eigen::Matrix3d&             eiMatRGBK()                         const {return _eimRGBK;}
    const cv::Mat&                  cvMatRGBK()                         const {return _mRGBK;}
    const Eigen::Matrix3d&             eiMatIRK()                          const {return _eimIRK;}
    const cv::Mat&                  cvMatIRK()                          const {return _mIRK;}
	Eigen::Matrix3d					eiMatK(int nCameraType_ ) const;

    const cv::Mat&                  cvMatRGBDistort()                   const {return _mRGBDistCoeffs;}
    const cv::Mat&                  cvMatIRDistort()                    const {return _mIRDistCoeffs;}
    Eigen::Matrix3d                    eiMatRGBR(unsigned int nView_)      const {Eigen::Matrix3d eiMatR; eiMatR << cv::Mat_<double>(_vmRGBRotationMatrices[nView_]); return eiMatR;}
    const cv::Mat&                  cvMatRGBR(unsigned int nView_)      const {return _vmRGBRotationMatrices[nView_];}
    const cv::Mat&                  cvVecRGBR(unsigned int nView_)      const {return _vmRGBRotationVectors[nView_];}
    Eigen::Vector3d                    eiVecRGBT(unsigned int nView_)      const 
    {
        Eigen::Vector3d eiVecT; 
        cv::Mat mVec = _vmRGBTranslationVectors[nView_];//.t();
        eiVecT << cv::Mat_<double>(mVec);
        return eiVecT;
    }
    const cv::Mat&                  cvMatRGBT(unsigned int nView_)      const {return _vmRGBTranslationVectors[nView_];}
    Eigen::Matrix3d                    eiMatIRR(unsigned int nView_)       const {Eigen::Matrix3d eiMatR; eiMatR << cv::Mat_<double>(_vmIRRotationMatrices[nView_]); return eiMatR;}
    const cv::Mat&                  cvMatIRR(unsigned int nView_)       const {return _vmIRRotationMatrices[nView_];}
    const cv::Mat&                  cvVecIRR(unsigned int nView_)       const {return _vmIRRotationVectors[nView_];}

    Eigen::Vector3d                    eiVecIRT(unsigned int nView_)       const 
    {
        Eigen::Vector3d eiVecT; 
        cv::Mat mVec = _vmIRTranslationVectors[nView_];//.t();
        eiVecT << cv::Mat_<double>(mVec);
        return eiVecT;
    }
    const cv::Mat&                  cvMatIRT(unsigned int nView_)       const {return _vmRGBTranslationVectors[nView_];}
    const cv::Mat&                  cvMatRelativeRotation()             const {return _cvmRelativeRotation;}
    const Eigen::Matrix3d&             eiMatRelativeRotation()             const {return _eimRelativeRotation;}
    const cv::Mat&                  cvMatRelativeTranslation()          const {return _cvmRelativeTranslation;}
    const Eigen::Vector3d&             eiVecRelativeTranslation()          const {return _eivRelativeTranslation;}

    const unsigned int&         views()                             const {return _uViews;}
    const std::string                imagePathName(unsigned int nView_)  const {return _vstrRGBPathName[nView_];}
    const std::string                depthPathName(unsigned int nView_)  const {return _vstrIRPathName[nView_];}
    void exportKinectIntrinsics();
    void importKinectIntrinsics();
	void importKinectIntrinsicsYML();
    void undistortImages(const std::vector< cv::Mat >& vImages_,  const cv::Mat_<double>& cvmK_, const cv::Mat_<double>& cvmInvK_, const cv::Mat_<double>& cvmDistCoeffs_, std::vector< cv::Mat >* pvRGBUndistorted ) const;
	void undistortRGB ( const cv::Mat& cvmRGB_, cv::Mat* pcvmUndistorted_ ) const;
	void undistortIR ( const cv::Mat& cvmIR_, cv::Mat* pcvmUndistorted_ ) const;

	void loadImages ( const boost::filesystem::path& cFullPath_, const std::vector< std::string >& vImgNames_, std::vector< cv::Mat >* pvImgs_ ) const;
	void exportImages(const boost::filesystem::path& cFullPath_, const std::vector< std::string >& vImgNames_, const std::vector< cv::Mat >& vImgs_ ) const;
	void gpuUndistortRGB (const cv::gpu::GpuMat& cvgmOrigin_, cv::gpu::GpuMat* pcvgmUndistorde_ ) const;
	void gpuUndistortIR  (const cv::gpu::GpuMat& cvgmOrigin_, cv::gpu::GpuMat* pcvgmUndistorde_ ) const;

protected:	
	void locate2DCorners(const std::vector< cv::Mat >& vImages_,  const int& nX_, const int& nY_, std::vector< std::vector<cv::Point2f> >* pvv2DCorners_, int nPatternType_ = SQUARE) const;
	void definePattern (  const float& fX_, const float& fY_, const int& nX_, const int& nY_, const int& nPatternType_, std::vector<cv::Point3f>* pv3DPatternCorners_ ) const;
	void define3DCorners ( const std::vector<cv::Point3f>& vPattern_, const unsigned int& nViews_, std::vector< std::vector<cv::Point3f> >* pvv3DCorners_  ) const;

	void calibrate ();

/**
* @brief convert rotation std::vectors into rotation matrices using cv::
*/
    void convertRV2RM(const std::vector< cv::Mat >& vMat_, std::vector< cv::Mat >* pvMat_ ) const;
	void generateMapXY4UndistortRGB();
	void generateMapXY4UndistortIR ();

    virtual void save();
    virtual void load();



private:
// others
    unsigned int  _uViews;
// images
    std::vector< cv::Mat > _vRGBs;
    std::vector< cv::Mat > _vRGBUndistorted;
    std::vector< cv::Mat > _vIRs;
    std::vector< cv::Mat > _vIRUndistorted;
    std::vector< std::string > _vstrRGBPathName; //serialized
    std::vector< std::string > _vstrIRPathName;
	std::vector< std::string > _vstrUndistortedRGBPathName; //serialized
    std::vector< std::string > _vstrUndistortedIRPathName;

// 2D-3D correspondences serialized
    std::vector< std::vector<cv::Point2f> > _vvRGB2DCorners;
    std::vector< std::vector<cv::Point2f> > _vvIR2DCorners;
	std::vector< cv::Point3f >         _vPatterCorners3D;
    std::vector< std::vector<cv::Point3f> > _vv3DCorners;
// camera extrinsics serialized (double type)
    //rgb
    std::vector< cv::Mat > _vmRGBRotationVectors;
    std::vector< cv::Mat > _vmRGBRotationMatrices;
    std::vector< cv::Mat > _vmRGBTranslationVectors;
    //ir
    std::vector< cv::Mat > _vmIRRotationVectors;
    std::vector< cv::Mat > _vmIRRotationMatrices;
    std::vector< cv::Mat > _vmIRTranslationVectors;
// patern constancy serialized
    int _NUM_CORNERS_X;
    int _NUM_CORNERS_Y;
    float _X;
    float _Y;
// control flags	
	int _nLoadRGB;
	int _nLoadIR;
	int _nLoadUndistortedRGB;
	int _nLoadUndistortedIR;
	int _nUndistortImages;
	int _nExportUndistortedRGB;

	int _nExportUndistortedDepth;
	int _nCalibrate;
	int _nCalibrateDepth;
	int _nSerializeToXML;
	int _nSerializeFromXML;

	std::string _strImageDirectory;
protected:
    Eigen::Vector2i _vImageResolution; //x,y;
	int _nRows, _nCols;
// camera intrinsics serialized
    //rgb
    cv::Mat_<double> _mRGBK; 
    cv::Mat_<double> _mRGBInvK; // not serialized
    cv::Mat_<double> _mRGBDistCoeffs;
	Eigen::Matrix3d	 _eimRGBK;
	Eigen::Matrix3d  _eimRGBInvK;
    //ir
    cv::Mat_<double> _mIRK; 
    cv::Mat_<double> _mIRInvK; 
    cv::Mat_<double> _mIRDistCoeffs;
	Eigen::Matrix3d	 _eimIRK;
	Eigen::Matrix3d  _eimIRInvK;

    //relative pose
    cv::Mat_<double> _cvmRelativeRotation;
    cv::Mat_<double> _cvmRelativeTranslation;   
	Eigen::Matrix3d  _eimRelativeRotation;
	Eigen::Vector3d  _eivRelativeTranslation;

	cv::Mat          _cvmMapXYRGB; //for undistortion
	cv::Mat			 _cvmMapYRGB; //useless just for calling cv::remap
	cv::Mat          _cvmMapXYIR; //for undistortion
	cv::Mat			 _cvmMapYIR;
	//gpu version
	cv::gpu::GpuMat  _cvgmMapXRGB;
	cv::gpu::GpuMat  _cvgmMapYRGB;
	cv::gpu::GpuMat  _cvgmMapXIR;
	cv::gpu::GpuMat  _cvgmMapYIR;
public:

	// duplicated camera parameters for speed up the VideoSourceKinect::align() in . because Eigen and cv matrix class is very slow.
	// initialized in constructor after load of the _cCalibKinect.
	double _aR[9];	// Relative rotation transpose
	double _aRT[3]; // aRT =_aR * T, the relative translation

	float _fFxIR, _fFyIR, _uIR, _vIR; //_dFxIR, _dFyIR IR camera focal length
	float _fFxRGB,_fFyRGB,_uRGB,_vRGB;

	enum {IR_CAMERA, RGB_CAMERA } 	_nCameraType;
	enum {CIRCLE, SQUARE} 			_nPatternType;
};



/*
class CKinectView
{
public:
	//type
	typedef boost::shared_ptr<CKinectView> tp_shared_ptr;

public:
	CKinectView( CCalibrateKinect& cCK_)
	:_cVS(cCK_)	{}
	void LoadTexture(const cv::Mat& img);
	void setIntrinsics(unsigned int nScaleViewport_, int nCameraType_, double dNear_, double dFar_ );
	void renderCamera (int nCameraType_, const cv::Mat& cvmRGB_, int nCameraRender_ = ALL_CAMERA, double dPhysicalFocalLength_ = .02, bool bRenderTexture_=true ) const / *dPhysicalFocalLength_ = .02 by default * /;
	void renderOnImage( int nX_, int nY_ );

	enum {ALL_CAMERA, NONE_CAMERA} _eCameraRender;

protected:
	GLuint _uTexture;
	CCalibrateKinect& _cVS;
};*/



} //namespace kinect
} //namespace btl

#endif
