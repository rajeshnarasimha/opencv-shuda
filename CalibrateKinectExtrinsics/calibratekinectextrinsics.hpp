#ifndef CALIBRATEKINECTEXTRINSICS_THROUGH_IMAGES
#define CALIBRATEKINECTEXTRINSICS_THROUGH_IMAGES

#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <btl/Utility/Converters.hpp>
#include <btl/extra/VideoSource/calibratekinect.hpp>

/*
#ifndef RGB_CAMERA
	#define IR_CAMERA 0
	#define DEPTH_CAMERA IR_CAMERA //depth camera is ir camera
	#define RGB_CAMERA 1
#endif
*/

//class CCalibrateKinect;
//#include <btl/Camera/CameraModel.hpp>
using namespace btl;
using namespace utility;
using namespace extra;
using namespace videosource;
namespace shuda
{

class CCalibrateKinectExtrinsics : public CCalibrateKinect
{
public:
    CCalibrateKinectExtrinsics():CCalibrateKinect()
    {
        _NUM_CORNERS_X = 8;
        _NUM_CORNERS_Y = 6;
        _X = .03f;
        _Y = .03f;
		_NUM_IMAGES = 22;

		_nLoadRGB = 1;
		_nLoadDepth = 1;
		_nLoadUndistortedRGB = 1;
		_nLoadUndistortedDepth = 1;
		_nExportUndistortedRGB = 1;
		_nExportUndistortedDepth = 1;
		_nUndistortImages = 1;
		_nCalibrateExtrinsics = 1;
	 	_nSerializeToXML = 1;
		_nSerializeFromXML = 1;
		_nCalibrateDepth = 1;
		_nCollect3DPts = 1;
    }
	~CCalibrateKinectExtrinsics()
	{
		for( vector< double* >::iterator it = _vpRGBWorld.begin(); it != _vpRGBWorld.end(); it++ ) 
		{
			delete [] *it;
		}
		_vpRGBWorld.clear();
		_vvpColor.clear();
		_vvpRGB.clear();
	}

    virtual void mainFunc(const boost::filesystem::path& cFullPath_ );
    //btl::camera::CameraModelPinhole btlCameraModelPinHoleK() const; 
    //retriever:
    const cv::Mat&                  image (unsigned int uNthView_) const {return _vImages[uNthView_];}
    const cv::Mat&                  undistortedImg(unsigned int uNthView_) const {return _vUndistortedImages[uNthView_];}
    const cv::Mat&                  depth (unsigned int uNthView_) const {return _vDepthMaps[uNthView_];}
    const cv::Mat&                  undistortedDepth(unsigned int uNthView_) const {return _vUndistortedDepthMaps[uNthView_];}
//    const std::vector<cv::Point3f>& pattern(unsigned int uNthView_)const {return _vv3DCorners[uNthView_];}
    const Eigen::Vector2i&          imageResolution()              const {return _vImageResolution;}


    Eigen::Matrix3d                 eiMatR(unsigned int uNthView_) const {Eigen::Matrix3d eiMatR; eiMatR << cv::Mat_<double>(_vmRotationMatrices[uNthView_]); return eiMatR;}
    const cv::Mat&                  cvMatR(unsigned int uNthView_) const {return _vmRotationMatrices[uNthView_];}
    const cv::Mat&                  cvVecR(unsigned int uNthView_) const {return _vmRotationVectors[uNthView_];}
    Eigen::Vector3d                 eiVecT(unsigned int uNthView_) const 
    {
        Eigen::Vector3d eiVecT; 
        cv::Mat mVec = _vmTranslationVectors[uNthView_].t();
        eiVecT << cv::Mat_<double>(mVec);
        return eiVecT;
    }
    const cv::Mat&                  cvMatT(unsigned int uNthView_) const {return _vmTranslationVectors[uNthView_];}
	const cv::Mat_<double>          cvMatDK() const {return _mDK;}

    const unsigned int&             views()                     const {return _uViews;}
    const string                    imagePathName(unsigned int uNthView_) const {return _vstrImagePathName[uNthView_];}
    const string                    depthPathName(unsigned int uNthView_) const {return _vstrDepthPathName[uNthView_];}

	Eigen::Vector3d 				eiVecT(unsigned int uNthView_, int nCameraType_ ) const;
	Eigen::Matrix3d  				eiMatR(unsigned int uNthView_, int nCameraType_ ) const;
	Matrix< double, 3, 4 > 			calcProjMatrix( unsigned int uNthView_, int nCameraType_ ) const;
	void 							calcAllProjMatrices(int nCameraType_, std::vector< Matrix< double , 3, 4 > >* pveimProjs_ ) const;

	const vector< Vector3d >&       points(unsigned int uNthView_) const {return _vveiv3DPts[uNthView_];} 
	const  Matrix< double , 3, 4 >& prjMatrix(unsigned int uNthView_, int nCameraType_) const
	{
		switch ( nCameraType_ )
		{
			case IR_CAMERA: //DEPTH_CAMERA = IR_CAMERA;
				return _veimDepthProjs[uNthView_];
			case RGB_CAMERA:
				return _veimRGBProjs[uNthView_];
		}
	}
	//double* 						pointPtrs() {return _pRGBWorld; }
	const vector< unsigned char* >& colors(unsigned int uNthView_) const {return _vvpColor[uNthView_];}
	const double* 					pointPtr(int nView_ ) {return _vpRGBWorld[nView_];}
private:
    //void loadImages ( const boost::filesystem::path& cFullPath_, const std::vector< std::string >& vImgNames_, std::vector< cv::Mat >* pvImgs_ ) const;
	void calibrateExtrinsics ();
	void calibDepth();
	void calibDepthFreeNect();
	//void exportImages( const boost::filesystem::path& cFullPath_, const vector< std::string >& vImgNames_, const std::vector< cv::Mat >& vImgs_ ) const;
	void collect3DPt(unsigned int uNthView_, vector< Vector3d >* pveiv3DPts_, vector< unsigned char* >* pvp3DColors_) const;
	void collect3DPtAll(vector< vector< Vector3d > >* pvveiv3DPts_, vector< vector< unsigned char* > >* pvvp3DColors_) const;
	void createDepth ( unsigned int uNthView_, const Mat& cvmDepth_, Mat_<int>* pcvmDepthNew_ ) const;
	void filterDepth (const double& dThreshould_, const Mat& cvmDepth_, Mat_<int>* pcvmDepthNew_ ) const;
	void parseControlYAML();
	void convertDepth ();
	void buildRegistrationTable();
	void exportTable();
	void importTable();

	void collect3DPt();
	/*
	void unprojectIR( const unsigned short* pCamera_,const int& nN_, double* pWorld_ );
	void transformIR2RGB( const double* pIR_,const int& nN_, double* pRGB_ );
	void projectRGB( double* pWorld_,const int& nN_, double* pRGBWorld_ );
	*/

/**
* @brief convert rotation vectors into rotation matrices using cv::
*
* @param
* @param argv
*/
    virtual void save();
    virtual void load();

private:
// others
    unsigned int    _uViews;
    Eigen::Vector2i _vImageResolution; //x,y;
	std::string 	_strFullPath; 
	std::string 	_strImageDirectory;
	double 	        _dDepthThreshold;
// images
    std::vector< cv::Mat > _vImages;
    std::vector< cv::Mat > _vUndistortedImages;
    std::vector< cv::Mat > _vDepthMaps;
    std::vector< cv::Mat > _vUndistortedDepthMaps;
	//std::vector< cv::Mat_<unsigned short> > _vDepthInts;
	std::vector< cv::Mat_<unsigned short> > _vFilteredUndistDepthInts;
	//std::vector< cv::Mat_<int> > _vFilteredDepth;
    std::vector< std::string > _vstrImagePathName; //serialized
    std::vector< std::string > _vstrDepthPathName; //serialized
    std::vector< std::string > _vstrUndistortedImagePathName; 
    std::vector< std::string > _vstrUndistortedDepthPathName; 

// 3D Pts and colors
 	vector< vector< Vector3d > > 	   _vveiv3DPts;
	vector< vector< unsigned char* > > _vvp3DColors;

// 2D-3D correspondences serialized
    std::vector< std::vector<cv::Point2f> > _vv2DCorners;
    std::vector< std::vector<cv::Point3f> > _vv3DCorners;
// camera extrinsics serialized (double type)
    std::vector< cv::Mat > _vmRotationVectors;
    std::vector< cv::Mat > _vmRotationMatrices;
    std::vector< cv::Mat > _vmTranslationVectors;
// camera projection matrices
    std::vector< Matrix< double , 3, 4 > > _veimRGBProjs;
	std::vector< Matrix< double , 3, 4 > > _veimDepthProjs;
// depth converter parameters
    cv::Mat_<double> _mDK;
// patern constancy serialized
    int _NUM_CORNERS_X;
    int _NUM_CORNERS_Y;
    float _X;
    float _Y;
	int _NUM_IMAGES;
// control flags
	int _nLoadRGB;
	int _nLoadDepth;
	int _nLoadUndistortedRGB;
	int _nLoadUndistortedDepth;
	int _nExportUndistortedRGB;
	int _nExportUndistortedDepth;
	int _nUndistortImages;
	int _nCalibrateExtrinsics;
	int _nSerializeToXML;
	int _nSerializeFromXML;
	int _nCalibrateDepth;
	int _nCollect3DPts;

	// for checking table: unnecessary as the plain c++ code is fast enough.
	typedef Eigen::Matrix< short, 2, 1>         		Vector2s_type;
	typedef std::map< unsigned short, Vector2s_type >   map_type;
	typedef Eigen::Matrix< map_type, 480, 640  >    	table_type;
	table_type _mpTable;

	vector<double*>  _vpRGBWorld; //X,Y,Z coordinate in RGB camera reference system
	vector<vector<unsigned char*> > _vvpColor;
	vector<vector<double*> > _vvpRGB;
/*
	// duplicated camera parameters for speed up the code. because Eigen and cv matrix class is very slow.
	double _aR[9];	// Relative rotation transpose
	double _aRT[3]; // aRT* T, the relative translation
	double _fxIR, _fyIR, _uIR, _vIR;
	double _fxRGB,_fyRGB,_uRGB,_vRGB;
	*/
};
}//shuda

#endif
