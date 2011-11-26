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
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "Converters.hpp"
#include <GL/freeglut.h>

namespace btl
{
namespace extra
{
namespace videosource
{

using namespace btl;
using namespace utility;
using namespace cv;
using namespace Eigen;

class CCalibrateKinect
{

public:
	CCalibrateKinect();
 	~CCalibrateKinect();
	virtual void parseControlYAML();
    virtual void mainFunc(const boost::filesystem::path& cFullPath_ );

    //btl::camera::CameraModelPinhole btlCameraModelPinHoleK() const; 
    void cloneDepth(double* pDepth_)
	{
		double* pM = _pRGBWorldRGB;
		for( int i=0; i< 921600;i++ )
		{
			*pDepth_++ = *pM++; 
		}
	}

//retriever:
	const double* 				registeredDepth() const {return _pRGBWorldRGB; }	
    const Mat&                  rgbImage (unsigned int nView_)      const {return _vRGBs[nView_];}
    const Mat&                  rgbUndistorted(unsigned int nView_) const {return _vRGBUndistorted[nView_];}
	const Mat&                  irImage (unsigned int nView_)       const {return _vIRs[nView_];}
    const Mat&                  irUndistorted(unsigned int nView_)  const {return _vIRUndistorted[nView_];}
    const vector<cv::Point3f>&  pattern()        					const {return _vPatterCorners3D;}
    const Vector2i&             imageResolution()                   const {return _vImageResolution;}
    const Matrix3d&             eiMatRGBK()                         const {return _eimRGBK;}
    const Mat&                  cvMatRGBK()                         const {return _mRGBK;}
    const Matrix3d&             eiMatIRK()                          const {return _eimIRK;}
    const Mat&                  cvMatIRK()                          const {return _mIRK;}
	Matrix3d					eiMatK(int nCameraType_ ) const;

    const Mat&                  cvMatRGBDistort()                   const {return _mRGBDistCoeffs;}
    const Mat&                  cvMatIRDistort()                    const {return _mIRDistCoeffs;}
    Matrix3d                    eiMatRGBR(unsigned int nView_)      const {Matrix3d eiMatR; eiMatR << Mat_<double>(_vmRGBRotationMatrices[nView_]); return eiMatR;}
    const Mat&                  cvMatRGBR(unsigned int nView_)      const {return _vmRGBRotationMatrices[nView_];}
    const Mat&                  cvVecRGBR(unsigned int nView_)      const {return _vmRGBRotationVectors[nView_];}
    Vector3d                    eiVecRGBT(unsigned int nView_)      const 
    {
        Vector3d eiVecT; 
        Mat mVec = _vmRGBTranslationVectors[nView_];//.t();
        eiVecT << Mat_<double>(mVec);
        return eiVecT;
    }
    const Mat&                  cvMatRGBT(unsigned int nView_)      const {return _vmRGBTranslationVectors[nView_];}
    Matrix3d                    eiMatIRR(unsigned int nView_)       const {Matrix3d eiMatR; eiMatR << Mat_<double>(_vmIRRotationMatrices[nView_]); return eiMatR;}
    const Mat&                  cvMatIRR(unsigned int nView_)       const {return _vmIRRotationMatrices[nView_];}
    const Mat&                  cvVecIRR(unsigned int nView_)       const {return _vmIRRotationVectors[nView_];}

    Vector3d                    eiVecIRT(unsigned int nView_)       const 
    {
        Vector3d eiVecT; 
        Mat mVec = _vmIRTranslationVectors[nView_];//.t();
        eiVecT << Mat_<double>(mVec);
        return eiVecT;
    }
    const Mat&                  cvMatIRT(unsigned int nView_)       const {return _vmRGBTranslationVectors[nView_];}
    const Mat&                  cvMatRelativeRotation()             const {return _cvmRelativeRotation;}
    const Matrix3d&             eiMatRelativeRotation()             const {return _eimRelativeRotation;}
    const Mat&                  cvMatRelativeTranslation()          const {return _cvmRelativeTranslation;}
    const Vector3d&             eiVecRelativeTranslation()          const {return _eivRelativeTranslation;}

    const unsigned int&         views()                             const {return _uViews;}
    const string                imagePathName(unsigned int nView_)  const {return _vstrRGBPathName[nView_];}
    const string                depthPathName(unsigned int nView_)  const {return _vstrIRPathName[nView_];}
    void exportKinectIntrinsics();
    void importKinectIntrinsics();
    void undistortImages(const vector< Mat >& vImages_,  const Mat_<double>& cvmK_, const Mat_<double>& cvmInvK_, const Mat_<double>& cvmDistCoeffs_, vector< Mat >* pvRGBUndistorted ) const;
	void undistortRGB(const Mat& cvmRGB_, Mat& Undistorted_ ) const;
	void undistortIR (const Mat& cvmIR_, Mat& Undistorted_ ) const;

	void loadImages ( const boost::filesystem::path& cFullPath_, const std::vector< std::string >& vImgNames_, std::vector< cv::Mat >* pvImgs_ ) const;
	void exportImages(const boost::filesystem::path& cFullPath_, const vector< std::string >& vImgNames_, const std::vector< cv::Mat >& vImgs_ ) const;

	void registration( const unsigned short* pDepth_ );
	void unprojectIR( const unsigned short* pCamera_,const int& nN_, double* pWorld_ );
	void transformIR2RGB( const double* pIR_,const int& nN_, double* pRGB_ );
	void projectRGB( double* pWorld_,const int& nN_, double* ppRGBWorld_ );

protected:	
	void locate2DCorners(const vector< Mat >& vImages_,  const int& nX_, const int& nY_, vector< vector<cv::Point2f> >* pvv2DCorners_, int nPatternType_ = SQUARE) const;
	void definePattern (  const float& fX_, const float& fY_, const int& nX_, const int& nY_, const int& nPatternType_, vector<cv::Point3f>* pv3DPatternCorners_ ) const;
	void define3DCorners ( const vector<cv::Point3f>& vPattern_, const unsigned int& nViews_, vector< vector<cv::Point3f> >* pvv3DCorners_  ) const;

	void calibrate ();

/**
* @brief convert rotation vectors into rotation matrices using cv::
*/
    void convertRV2RM(const vector< Mat >& vMat_, vector< Mat >* pvMat_ ) const;
	void generateMapXY4UndistortRGB();
	void generateMapXY4UndistortIR ();
	void computeFundamental();

    virtual void save();
    virtual void load();


private:
// others
    unsigned int  _uViews;
// images
    vector< Mat > _vRGBs;
    vector< Mat > _vRGBUndistorted;
    vector< Mat > _vIRs;
    vector< Mat > _vIRUndistorted;
    vector< string > _vstrRGBPathName; //serialized
    vector< string > _vstrIRPathName;
	vector< string > _vstrUndistortedRGBPathName; //serialized
    vector< string > _vstrUndistortedIRPathName;

// 2D-3D correspondences serialized
    vector< vector<cv::Point2f> > _vvRGB2DCorners;
    vector< vector<cv::Point2f> > _vvIR2DCorners;
	vector< cv::Point3f >         _vPatterCorners3D;
    vector< vector<cv::Point3f> > _vv3DCorners;
// camera extrinsics serialized (double type)
    //rgb
    vector< Mat > _vmRGBRotationVectors;
    vector< Mat > _vmRGBRotationMatrices;
    vector< Mat > _vmRGBTranslationVectors;
    //ir
    vector< Mat > _vmIRRotationVectors;
    vector< Mat > _vmIRRotationMatrices;
    vector< Mat > _vmIRTranslationVectors;
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

	string _strImageDirectory;
protected:
    Vector2i      _vImageResolution; //x,y;
// camera intrinsics serialized
    //rgb
    Mat_<double> _mRGBK; 
    Mat_<double> _mRGBInvK; // not serialized
    Mat_<double> _mRGBDistCoeffs;
	Matrix3d	 _eimRGBK;
	Matrix3d     _eimRGBInvK;
    //ir
    Mat_<double> _mIRK; 
    Mat_<double> _mIRInvK; 
    Mat_<double> _mIRDistCoeffs;
	Matrix3d	 _eimIRK;
	Matrix3d     _eimIRInvK;

    //relative pose
    Mat_<double> _cvmRelativeRotation;
    Mat_<double> _cvmRelativeTranslation;   
	Matrix3d     _eimRelativeRotation;
	Vector3d     _eivRelativeTranslation;

	Mat          _cvmMapXYRGB; //for undistortion
	Mat          _cvmMapXYIR; //for undistortion
	Mat			 _cvmMapY; //useless just for calling cv::remap

//timer
	boost::posix_time::ptime _cT0, _cT1;
	boost::posix_time::time_duration _cTDAll;
//paramters
	double _dThreshouldDepth;
public:

	// duplicated camera parameters for speed up the code. because Eigen and cv matrix class is very slow.
	// initialized in constructor after load of the _cCalibKinect.
	double _aR[9];	// Relative rotation transpose
	double _aRT[3]; // aRT =_aR * T, the relative translation
	double _fxIR, _fyIR, _uIR, _vIR;
	double _fxRGB,_fyRGB,_uRGB,_vRGB;

	// temporary variables allocated in constructor and released in destructor
	unsigned short* _pPxDIR; //2D coordinate along with depth for ir image
	unsigned short* _pPxRGB; //2D coordinate in rgb image
	double*  _pIRWorld;      //XYZ w.r.t. IR camera reference system 
	double*  _pRGBWorld;
	// refreshed for every frame
	double*  _pRGBWorldRGB; //X,Y,Z coordinate of depth w.r.t. RGB camera reference system
 	//double** _ppRGBWorld;//registered to RGB image of the X,Y,Z coordinate

	enum {IR_CAMERA, RGB_CAMERA } 	_nCameraType;
	enum {CIRCLE, SQUARE} 			_nPatternType;
};

class CKinectView
{
public:
	CKinectView( CCalibrateKinect& cCK_)
	:_cCK(cCK_)
	{
	}

	GLuint LoadTexture(const cv::Mat& img);
	void setIntrinsics(unsigned int nScaleViewport_, int nCameraType_, double dNear_, double dFar_ );
	void renderCamera( GLuint uTexture_, int nCameraType_, int nCameraRender_ = ALL_CAMERA, double dPhysicalFocalLength_ = .02 ) const;
	void renderOnImage( int nX_, int nY_ );

	enum {ALL_CAMERA, NONE_CAMERA} _nCameraRender;


protected:
	CCalibrateKinect& _cCK;
};



} //namespace videosource
} //namespace extra
} //namespace btl

#endif
