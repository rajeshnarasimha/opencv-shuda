#include <opencv2/gpu/gpu.hpp>
#include "calibratekinect.hpp"
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/lexical_cast.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <map>
//using namespace cv;

namespace btl
{
namespace extra
{
namespace videosource
{

CCalibrateKinect::CCalibrateKinect()
{
    _NUM_CORNERS_X = 8;
    _NUM_CORNERS_Y = 6;
    _X = .03f;
    _Y = .03f;

    //importKinectIntrinsics();// obsolete
	importKinectIntrinsicsYML();

    //prepare camera parameters
    const Eigen::Vector3d& vT = eiVecRelativeTranslation();
    Eigen::Matrix3d mRTrans = eiMatRelativeRotation().transpose();
    Eigen::Vector3d vRT = mRTrans * vT;

    _aR[0] = mRTrans ( 0, 0 );
    _aR[1] = mRTrans ( 0, 1 );
    _aR[2] = mRTrans ( 0, 2 );
    _aR[3] = mRTrans ( 1, 0 );
    _aR[4] = mRTrans ( 1, 1 );
    _aR[5] = mRTrans ( 1, 2 );
    _aR[6] = mRTrans ( 2, 0 );
    _aR[7] = mRTrans ( 2, 1 );
    _aR[8] = mRTrans ( 2, 2 );

    _aRT[0] = vRT ( 0 );
    _aRT[1] = vRT ( 1 );
    _aRT[2] = vRT ( 2 );

    _dFxIR = eiMatIRK() ( 0, 0 );
    _dFyIR = eiMatIRK() ( 1, 1 );
    _uIR  = eiMatIRK() ( 0, 2 );
    _vIR  = eiMatIRK() ( 1, 2 );

    _dFxRGB = eiMatRGBK() ( 0, 0 );
    _dFyRGB = eiMatRGBK() ( 1, 1 );
    _uRGB  = eiMatRGBK() ( 0, 2 );
    _vRGB  = eiMatRGBK() ( 1, 2 );

	_nPatternType = SQUARE;
    //define 3D pattern corners
    definePattern ( _X, _Y, _NUM_CORNERS_X, _NUM_CORNERS_Y, _nPatternType, &_vPatterCorners3D );

    std::cout << "CCalibrateKinect() done." << std::endl;
    return;
}

CCalibrateKinect::~CCalibrateKinect()
{

}

Eigen::Matrix3d CCalibrateKinect::eiMatK ( int nCameraType_ ) const
{
    Eigen::Matrix3d eimK;

    if ( IR_CAMERA == nCameraType_ )
    {
        eimK = eiMatIRK();
    }
    else if ( RGB_CAMERA == nCameraType_ )
    {
        eimK = eiMatRGBK();
    }
    else
    {
        THROW ( "eiMatK(): unrecognized camera type." );
    }

    return eimK;
}

void CCalibrateKinect::parseControlYAML()
{
    std::ifstream iFile ( "/space/csxsl/src/opencv-shuda/Data/control.yaml" );
    YAML::Parser parser ( iFile );
    YAML::Node doc;
    parser.GetNextDocument ( doc );
    std::map <std::string, int> mpOpr1;

    for ( unsigned i = 0; i < doc.size(); i++ )
    {
        doc[i] >> mpOpr1;
    }

    //PRINT( mpOpr1 );
    //process mpOpr1;
    std::map< std::string, int >::const_iterator cIt1;

    for ( cIt1 = mpOpr1.begin(); cIt1 != mpOpr1.end(); cIt1++ )
    {
        if ( "load rgb images" == ( *cIt1 ).first )
        {
            _nLoadRGB = ( *cIt1 ).second;
        }
        else if ( "load ir images" == ( *cIt1 ).first )
        {
            _nLoadIR = ( *cIt1 ).second;
        }
        else if ( "load Undistorted rgb images" == ( *cIt1 ).first )
        {
            _nLoadUndistortedRGB = ( *cIt1 ).second;
        }
        else if ( "load undistorted ir images" == ( *cIt1 ).first )
        {
            _nLoadUndistortedIR = ( *cIt1 ).second;
        }
        else if ( "undistort images" == ( *cIt1 ).first )
        {
            _nUndistortImages = ( *cIt1 ).second;
        }
        else if ( "export undistorted rgb images" == ( *cIt1 ).first )
        {
            _nExportUndistortedRGB = ( *cIt1 ).second;
        }
        else if ( "export undistorted ir images" == ( *cIt1 ).first )
        {
            _nExportUndistortedDepth = ( *cIt1 ).second;
        }
        else if ( "calibrate" == ( *cIt1 ).first )
        {
            _nCalibrate = ( *cIt1 ).second;
        }
        else if ( "calibrate ir camera offset" == ( *cIt1 ).first )
        {
            _nCalibrateDepth = ( *cIt1 ).second;
        }
        else if ( "export to xml" == ( *cIt1 ).first )
        {
            _nSerializeToXML = ( *cIt1 ).second;
        }
        else if ( "import from xml" == ( *cIt1 ).first )
        {
            _nSerializeFromXML = ( *cIt1 ).second;
        }
    }

	std::map < std::string, std::string> mpType;
    parser.GetNextDocument ( doc );
    doc[0] >> mpType;
    std::string strGridType = mpType["grid type"];
	if ( "circle" == strGridType )
		_nPatternType = CIRCLE;
	else if( "chessboard" == strGridType )
		_nPatternType = SQUARE;
		
    // properties
    std::map < std::string, std::vector< int > > mpCornerCounts;
    parser.GetNextDocument ( doc );
    doc[0] >> mpCornerCounts;
    //PRINT( mpCornerCounts );
    //process mpCornerCounts
    std::vector< int > vCornerCounts;
    vCornerCounts = mpCornerCounts["No. of Corners X Y"];
    _NUM_CORNERS_X = vCornerCounts[0];
    _NUM_CORNERS_Y = vCornerCounts[1];

    std::map < std::string, std::vector< float > > mpUnitLength;
    parser.GetNextDocument ( doc );
    doc[0] >> mpUnitLength;
    //PRINT( mpUnitLength );
    //process mpUnitLength
    std::vector< float > vUnitLength;
    vUnitLength = mpUnitLength["Unit length in X Y"];
    _X = vUnitLength[0];
    _Y = vUnitLength[1];

    std::map < std::string, std::string> mpProperties;
    parser.GetNextDocument ( doc );
    doc[0] >> mpProperties;
    std::string _strImageDirectory = mpProperties["image directory"];
    //PRINT( mpProperties );
	
	std::vector< int > vIdx;
	parser.GetNextDocument ( doc );
	for ( unsigned i = 0; i < doc.size(); i++ )
	{
		doc[i] >> vIdx;
	}

    std::vector< std::string> vRGBNames;
	std::vector< std::string> vDepthNames;
	std::vector< std::string> vUndistRGB;
	std::vector< std::string> vUndistDepth;

	for ( std::vector< int >::const_iterator cit = vIdx.begin(); cit != vIdx.end(); cit ++ )
	{
		std::string strNum = boost::lexical_cast< std::string> ( *cit );
		vRGBNames.	push_back( "rgb"+strNum+".bmp" );
		vDepthNames.push_back( "ir" +strNum+".bmp" );
		vUndistRGB. push_back( "rgbUndistorted"+strNum+".bmp" );
		vUndistDepth.push_back("irUndistorted" +strNum+".bmp" );
	}

    PRINT( vRGBNames );
    PRINT( vDepthNames );
    PRINT( vUndistRGB );
	PRINT( vUndistDepth );

    CHECK ( vRGBNames.size() == vDepthNames.size(), "There must be the same # of rgb and depth, undistorted rgb and depth." );
    CHECK ( vRGBNames.size() == vUndistRGB.size(), "There must be the same # of rgb and depth, undistorted rgb and depth." );
    CHECK ( vRGBNames.size() == vUndistDepth.size(), "There must be the same # of rgb and depth, undistorted rgb and depth." );

    _uViews = vRGBNames.size();

    for ( unsigned int i = 0; i < vRGBNames.size(); i++ )
    {
        _vstrRGBPathName.push_back ( _strImageDirectory + vRGBNames[i] );
        _vstrIRPathName.push_back ( _strImageDirectory + vDepthNames[i] );
        _vstrUndistortedRGBPathName.push_back ( _strImageDirectory + vUndistRGB[i] );
        _vstrUndistortedIRPathName.push_back ( _strImageDirectory + vUndistDepth[i] );
    }

    return;
}

void CCalibrateKinect::mainFunc ( const boost::filesystem::path& cFullPath_ )
{
    parseControlYAML();

    //load images
    if ( 1 == _nLoadRGB )
    {
        loadImages ( cFullPath_, _vstrRGBPathName, &_vRGBs );
        _nCols = _vImageResolution ( 0 ) = _vRGBs[0].cols;
        _nRows = _vImageResolution ( 1 ) = _vRGBs[0].rows;
        std::cout << "rgb images loaded.\n";
    }

    if ( 1 == _nLoadIR )
    {
        //PRINT( _vstrIRPathName );
        loadImages ( cFullPath_, _vstrIRPathName, &_vIRs );
        std::cout << "ir images loaded.\n";
    }

    if ( 1 == _nLoadUndistortedRGB )
    {
        loadImages ( cFullPath_, _vstrUndistortedRGBPathName, &_vRGBUndistorted );
        std::cout << "undistorted rgb images loaded.\n";
    }

    if ( 1 == _nLoadUndistortedIR )
    {
        loadImages ( cFullPath_, _vstrUndistortedIRPathName, &_vIRUndistorted );
        std::cout << "undistorted ir images loaded.\n";
    }

    if ( 1 == _nCalibrate )
    {
        //find corners
        locate2DCorners ( _vRGBs, _NUM_CORNERS_X, _NUM_CORNERS_Y, &_vvRGB2DCorners, _nPatternType );
        locate2DCorners ( _vIRs,  _NUM_CORNERS_X, _NUM_CORNERS_Y, &_vvIR2DCorners, _nPatternType );
        std::cout << "2d corners located \n" ;
		//PRINT( _vvIR2DCorners );

		definePattern ( _X, _Y, _NUM_CORNERS_X, _NUM_CORNERS_Y, _nPatternType, &_vPatterCorners3D );
		//PRINT( _vPatterCorners3D );
		std::cout << "define pattern \n";
        define3DCorners ( _vPatterCorners3D, views(), &_vv3DCorners );
        std::cout << "3d corners defined \n" ;

        //calibration
        calibrate();
        std::cout << "camera calibrated \n" ;

        //convert rotation std::vectors to rotation matrices
        convertRV2RM ( _vmRGBRotationVectors, &_vmRGBRotationMatrices );
        convertRV2RM ( _vmIRRotationVectors, &_vmIRRotationMatrices );

        std::cout << "convertRV2RM() executed \n" ;
    }

    //remove radical distortions
    if ( 1 == _nUndistortImages )
    {
		
        undistortImages ( _vRGBs, _mRGBK, _mRGBInvK, _mRGBDistCoeffs,  &_vRGBUndistorted );
        undistortImages ( _vIRs,  _mIRK,  _mIRInvK,  _mIRDistCoeffs,   &_vIRUndistorted );
        std::cout << "image undistorted.\n";
    }

    if ( 1 == _nExportUndistortedRGB )
    {
        exportImages ( cFullPath_, _vstrUndistortedRGBPathName,  _vRGBUndistorted );
        std::cout << " Undistorted rgb image exported. \n";
    }

    if ( 1 == _nExportUndistortedDepth )
    {
        exportImages ( cFullPath_, _vstrUndistortedIRPathName,  _vIRUndistorted );
        std::cout << " Undistorted depth image exported. \n";
    }

    if ( 1 == _nSerializeToXML )
    {
        save();
        std::cout << "serialized to XML \n" ;
    }

    if ( 1 == _nSerializeFromXML )
    {
        load();
        std::cout << "serialized from XML \n" ;
    }

    exportKinectIntrinsics();
	std::cout << " intrinsics exported \n" ;
    //importKinectIntrinsics();//obsolete
	importKinectIntrinsicsYML();
	std::cout << " intrinsics imported \n" ;
    return;
}
void CCalibrateKinect::loadImages ( const boost::filesystem::path& cFullPath_, const std::vector< std::string >& vImgNames_, std::vector< cv::Mat >* pvImgs_ ) const
{
    pvImgs_->clear();

    std::string strPathName  = cFullPath_.string();

    for ( unsigned int i = 0; i < vImgNames_.size(); i++ )
    {
        std::string strRGBFileName = strPathName + vImgNames_[i]; //saved into the folder from which the KinectCalibrationDemo is being run.
        //PRINT( strRGBFileName );
        CHECK ( boost::filesystem::exists ( strRGBFileName ), "Not found: " + strRGBFileName + "\n" );
        cv::Mat img = cv::imread ( strRGBFileName );
        cvtColor ( img, img, CV_BGR2RGB );
        pvImgs_->push_back ( img );
    }

    return;
}
void CCalibrateKinect::exportImages ( const boost::filesystem::path& cFullPath_, const std::vector< std::string >& vImgNames_, const std::vector< cv::Mat >& vImgs_ ) const
{
    std::string strPathName  = cFullPath_.string();

    for ( unsigned int n = 0; n < vImgNames_.size(); n++ )
    {
        cv::Mat img = vImgs_[n];
        cvtColor ( img, img, CV_RGB2BGR );
        cv::imwrite ( strPathName + vImgNames_[n], vImgs_[n] );
    }

    return;
}


void CCalibrateKinect::locate2DCorners ( const std::vector< cv::Mat >& vImages_, const int& nX_, const int& nY_, std::vector< std::vector<cv::Point2f> >* pvv2DCorners_, int nPatternType_ ) const //nPatternType_ = SQUARE
{

    CHECK ( !vImages_.empty(), "locate2DCorners(): no images.\n" );

    if ( SQUARE == nPatternType_ )
    {
		std::cout << " locate chessboard corners.\n ";
        pvv2DCorners_->clear();

        cv::Size patternSize ( nX_, nY_ );

        for ( unsigned int i = 0; i < vImages_.size(); i++ )
        {
            const cv::Mat& cvFrame = vImages_[i] ;

            std::vector<cv::Point2f> vCurrentCorners;//float 2d point is required by the OpenCV API.
            //locate corners roughly
            bool _bChessBoardCornersFoundThisFrame = cv::findChessboardCorners ( cvFrame, patternSize, vCurrentCorners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS );

            CHECK ( _bChessBoardCornersFoundThisFrame, " No corners are found.\n" );
			PRINT( vCurrentCorners.size() );
            //locate corners in sub-pixel level
            cv::Mat cvFrameGrey;
            cv::cvtColor ( cvFrame, cvFrameGrey, CV_BGR2GRAY );
            cv::cornerSubPix ( cvFrameGrey, vCurrentCorners, cv::Size ( 9, 9 ), cv::Size ( -1, -1 ), cv::TermCriteria ( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1 ) );

            //store the corners inpto a std::vector
            pvv2DCorners_->push_back ( vCurrentCorners );
        }
    }
    else if ( CIRCLE == nPatternType_ )
    {
		std::cout << "locate circle centers.\n" ;
        pvv2DCorners_->clear();

		PRINT( nX_ );
		PRINT( nY_ );

        cv::Size patternSize ( nX_, nY_ );

        for ( unsigned int i = 0; i < vImages_.size(); i++ )
        {
            const cv::Mat& cvFrame = vImages_[i] ;
			cv::Mat cvmTmp;
			cv::cvtColor ( cvFrame, cvmTmp, CV_RGB2GRAY );

            std::vector<cv::Point2f> vCurrentCorners;//float 2d point is required by the OpenCV API.
            //locate corners roughly
            bool _bChessBoardCornersFoundThisFrame = cv::findCirclesGrid ( cvmTmp, patternSize, vCurrentCorners, cv::CALIB_CB_ASYMMETRIC_GRID );

			PRINT( vCurrentCorners.size() );

            //CHECK ( _bChessBoardCornersFoundThisFrame, " No corners are found.\n" );

            //store the corners inpto a std::vector
            pvv2DCorners_->push_back ( vCurrentCorners );
        }
    }

    return;
}

void CCalibrateKinect::definePattern (  const float& fX_, const float& fY_, const int& nX_, const int& nY_,  const int& nPatternType_, std::vector<cv::Point3f>* pv3DPatternCorners_ ) const
{
    pv3DPatternCorners_->clear();

	if( SQUARE == nPatternType_ )
	{
	    for ( int r = 0; r < nY_; r++ )
        for ( int c = 0; c < nX_; c++ )
        {
            pv3DPatternCorners_->push_back ( cv::Point3f ( c * fX_, r * fY_,  0.0f ) );
        }
	}
	else if( CIRCLE == nPatternType_ )
	{
		float fXHalf = fX_/2;
		float fYHalf = fY_/2;
		for ( int r = 0; r < nY_; r++ )
        for ( int c = 0; c < nX_; c++ )
        {
			float fOffset = r%2?fXHalf:0; // for 0,2,4 .. fOffset = 0;
										  // for 1,3,5 .. fOffset = fXHalf;
            pv3DPatternCorners_->push_back ( cv::Point3f ( c * fX_ + fOffset , r * fYHalf,  0.0f ) );
        }

	}

    return;
}

void CCalibrateKinect::define3DCorners ( const std::vector<cv::Point3f>& vPattern_, const unsigned int& nViews_, std::vector< std::vector<cv::Point3f> >* pvv3DCorners_  ) const
{
    pvv3DCorners_->clear();

    for ( unsigned int i = 0; i < nViews_; ++i )
    {
        pvv3DCorners_->push_back ( vPattern_ );
    }

    return;
}

void CCalibrateKinect::calibrate ()
{
    _mRGBK = cv::Mat_<double> ( 3, 3 );
    _mRGBDistCoeffs = cv::Mat_<double> ( 5, 1 );

    _mRGBK.at<double> ( 0, 0 ) = 1;
    _mRGBK.at<double> ( 1, 1 ) = 1;

    if ( _vRGBs.empty() )
    {
        CError cE;
        cE << CErrorInfo ( " No images have been loaded. \n" );
        throw cE;
    }

    cv::Size cvFrameSize ( _nCols, _nRows );

    //calibrate the rgb camera
    double dErrorRGB = cv::calibrateCamera ( _vv3DCorners, _vvRGB2DCorners, cvFrameSize, _mRGBK, _mRGBDistCoeffs, _vmRGBRotationVectors, _vmRGBTranslationVectors );
	std::cout << "calibrate RGBs.\n";
    double dErrorIR  = cv::calibrateCamera ( _vv3DCorners, _vvIR2DCorners,  cvFrameSize, _mIRK,  _mIRDistCoeffs,  _vmIRRotationVectors,  _vmIRTranslationVectors  );
	std::cout << "calibrate IRs.\n";

    cv::Mat E, F; // E is essential matrix and F is the fundamental matrix
    double dErrorStereo = cv::stereoCalibrate ( _vv3DCorners, _vvRGB2DCorners, _vvIR2DCorners, _mRGBK, _mRGBDistCoeffs, _mIRK, _mIRDistCoeffs, cvFrameSize, _cvmRelativeRotation, _cvmRelativeTranslation, E, F );

    cv::invert ( _mRGBK, _mRGBInvK, cv::DECOMP_SVD );
    cv::invert ( _mIRK, _mIRInvK, cv::DECOMP_SVD );
    _eimRGBK << _mRGBK;
    _eimRGBInvK << _mRGBInvK;
    _eimIRK  << _mIRK;
    _eimIRInvK << _mIRInvK;
    _eimRelativeRotation << _cvmRelativeRotation;
    _eivRelativeTranslation << _cvmRelativeTranslation;
    PRINT( dErrorRGB );
    PRINT( dErrorIR );
    PRINT( dErrorStereo );

    //output calibration results
//    std::cout << "Camera calibrated." << std::endl;
   
    PRINT( _mIRK );
        PRINT( _mRGBDistCoeffs );

        PRINT( _mRGBK );
        PRINT( _mIRDistCoeffs );
    
}

/**
* @brief convert rotation std::vectors into rotation matrices using cv::
*
* @param
* @param argv
*/
void CCalibrateKinect::convertRV2RM ( const std::vector< cv::Mat >& vMat_, std::vector< cv::Mat >* pvMat_ ) const
{
    pvMat_->clear();

    for ( unsigned int n = 0; n < vMat_.size() ; n++ )
    {
        cv::Mat cvmRGB;
        cv::Rodrigues ( vMat_[n], cvmRGB );
        pvMat_->push_back ( cvmRGB );
    }

    /*
        PRINT( _cvmRelativeRotation );
        PRINT( _cvmRelativeTranslation );
        cv::Mat vRT  = (Mat_<double>(1,3) << 0, 0, 0);
        cv::Mat vRR0 = (Mat_<double>(3,1) << 0, 0, 0);
        cv::Mat mRR, mRGBInvR;
        for(unsigned int i = 0; i < views(); i++)
        {
            vRT += _vmIRTranslationVectors[i] - _vmRGBTranslationVectors[i];
            cv::invert( _vmRGBRotationMatrices[i], mRGBInvR, DECOMP_SVD );
            mRR = _vmIRRotationMatrices[i] * mRGBInvR;
            cv::Mat vRR;
            cv::Rodrigues ( mRR, vRR );
            vRR0 += vRR;
        }
        vRT /= views();
        vRR0 /= views();
        cv::Rodrigues ( vRR0, _cvmRelativeRotation );
        _cvmRelativeTranslation = vRT.t();

        PRINT( _cvmRelativeRotation );
        PRINT( vRT );
    */
    return;
}

void CCalibrateKinect::undistortRGB ( const cv::Mat& cvmRGB_, cv::Mat& Undistorted_ ) const
{
    cv::remap ( cvmRGB_, Undistorted_, _cvmMapXYRGB, _cvmMapYRGB, cv::INTER_NEAREST, cv::BORDER_CONSTANT );
}
void CCalibrateKinect::gpuUndistortRGB (const cv::gpu::GpuMat& cvgmOrigin_, cv::gpu::GpuMat* pcvgmUndistorde_ ) const
{
	cv::gpu::remap(cvgmOrigin_, *pcvgmUndistorde_, _cvgmMapXRGB, _cvgmMapYRGB, cv::INTER_NEAREST );
}

void CCalibrateKinect::undistortIR ( const cv::Mat& cvmIR_, cv::Mat& Undistorted_ ) const
{
    cv::remap ( cvmIR_, Undistorted_, _cvmMapXYIR, _cvmMapYRGB, cv::INTER_NEAREST, cv::BORDER_CONSTANT );
}
void CCalibrateKinect::gpuUndistortIR (const cv::gpu::GpuMat& cvgmOrigin_, cv::gpu::GpuMat* pcvgmUndistorde_ ) const
{
	cv::gpu::remap(cvgmOrigin_, *pcvgmUndistorde_, _cvgmMapXRGB, _cvgmMapYRGB, cv::INTER_NEAREST );
}

void CCalibrateKinect::generateMapXY4UndistortRGB()
{
    CHECK ( !_mRGBK.empty(),         "undistortImages(): K matrix cannot be empty.\n" );
    CHECK ( !_mRGBInvK.empty(),      "undistortImages(): inverse of K matrix cannot be empty.\n" );
    CHECK ( !_mRGBDistCoeffs.empty(), "undistortImages(): distortion coefficients cannot be empty.\n" );

    btl::utility::map4UndistortImage<double> ( _nRows, _nCols, _mRGBK, _mRGBInvK, _mRGBDistCoeffs, &_cvmMapXYRGB, &_cvmMapYRGB );
	_cvgmMapXRGB.upload(_cvmMapXYRGB);
	_cvgmMapYRGB.upload(_cvmMapYRGB);
}
void CCalibrateKinect::generateMapXY4UndistortIR()
{
    CHECK ( !_mIRK.empty(),         "undistortImages(): K matrix cannot be empty.\n" );
    CHECK ( !_mIRInvK.empty(),      "undistortImages(): inverse of K matrix cannot be empty.\n" );
    CHECK ( !_mIRDistCoeffs.empty(), "undistortImages(): distortion coefficients cannot be empty.\n" );

    btl::utility::map4UndistortImage<double> ( _nRows, _nCols, _mIRK, _mIRInvK, _mIRDistCoeffs, &_cvmMapXYIR, &_cvmMapYIR );
	_cvgmMapXIR.upload(_cvmMapXYIR);
	_cvgmMapYIR.upload(_cvmMapYIR);
}
void CCalibrateKinect::undistortImages ( const std::vector< cv::Mat >& vImages_,  const cv::Mat_<double>& cvmK_, const cv::Mat_<double>& cvmInvK_, const cv::Mat_<double>& cvmDistCoeffs_, std::vector< cv::Mat >* pvUndistorted_ ) const
{
	std::cout << "undistortImages() "<< std::endl << std::flush;
    CHECK ( !vImages_.empty(),      "undistortImages(): # of undistorted images can not be zero.\n" );
    CHECK ( !cvmK_.empty(),         "undistortImages(): K matrix cannot be empty.\n" );
    CHECK ( !cvmInvK_.empty(),      "undistortImages(): inverse of K matrix cannot be empty.\n" );
    CHECK ( !cvmDistCoeffs_.empty(), "undistortImages(): distortion coefficients cannot be empty.\n" );

    cv::Size cvFrameSize = vImages_[0].size(); //x,y;
    pvUndistorted_->clear();

    for ( unsigned int n = 0; n < vImages_.size(); n++ )
    {
        cv::Mat cvUndistorted;
		std::cout << "distort: "<< n << "-th image.\n"<< std::flush;
        undistortImage ( vImages_[n],  cvmK_, cvmInvK_, cvmDistCoeffs_, &cvUndistorted );
        pvUndistorted_->push_back ( cvUndistorted );

        //string strNum = boost::lexical_cast< std::string> ( n );
        //string strRGBUndistortedFileName = "rgbUndistorted" + strNum + ".bmp";
        //cv::imwrite ( strRGBUndistortedFileName, cvUndistorted );
    }

    return;
}

void CCalibrateKinect::save()
{
    // create and open a character archive for output
    std::ofstream ofs ( "CalibrateKinect.xml" );
    boost::archive::xml_oarchive oa ( ofs );

    //convert non-standard variables into std::vectors

    //rgb
    std::vector< std::vector< double > > stdvRGBKMatrix;
    std::vector< std::vector< double > > stdvRGBDistortionCoeff;
    std::vector< std::vector< std::vector< double > > > stdvRGBRotationVectors;
    std::vector< std::vector< std::vector< double > > > stdvRGBRotationMatrices;
    std::vector< std::vector< std::vector< double > > > stdvRGBTranslationVectors;
    std::vector< std::vector< std::vector< float  > > > stdvRGB2DCorners;
    //ir
    std::vector< std::vector< double > > stdvIRKMatrix;
    std::vector< std::vector< double > > stdvIRDistortionCoeff;
    std::vector< std::vector< std::vector< double > > > stdvIRRotationVectors;
    std::vector< std::vector< std::vector< double > > > stdvIRRotationMatrices;
    std::vector< std::vector< std::vector< double > > > stdvIRTranslationVectors;
    std::vector< std::vector< std::vector< float  > > > stdvIR2DCorners;
    //both
    std::vector< std::vector< int > >    stdvImageResolution;
    std::vector< std::vector< std::vector< float  > > > stdv3DCorners;

    //rgb
    stdvRGBKMatrix           << _mRGBK;
    stdvRGBDistortionCoeff   << _mRGBDistCoeffs;
    stdvRGBRotationVectors   << _vmRGBRotationVectors;
    stdvRGBRotationMatrices  << _vmRGBRotationMatrices;
    stdvRGBTranslationVectors << _vmRGBTranslationVectors;
    stdvRGB2DCorners         << _vvRGB2DCorners;
    //ir
    stdvIRKMatrix           << _mIRK;
    stdvIRDistortionCoeff   << _mIRDistCoeffs;
    stdvIRRotationVectors   << _vmIRRotationVectors;
    stdvIRRotationMatrices  << _vmIRRotationMatrices;
    stdvIRTranslationVectors << _vmIRTranslationVectors;
    stdvIR2DCorners         << _vvIR2DCorners;
    //both
    stdvImageResolution   << _vImageResolution;
    stdv3DCorners         << _vv3DCorners;

    oa << BOOST_SERIALIZATION_NVP ( _NUM_CORNERS_X );
    oa << BOOST_SERIALIZATION_NVP ( _NUM_CORNERS_Y );
    oa << BOOST_SERIALIZATION_NVP ( _X );
    oa << BOOST_SERIALIZATION_NVP ( _Y );
    oa << BOOST_SERIALIZATION_NVP ( _uViews );
    oa << BOOST_SERIALIZATION_NVP ( _vstrRGBPathName );
    oa << BOOST_SERIALIZATION_NVP ( _vstrIRPathName );
    oa << BOOST_SERIALIZATION_NVP ( stdvImageResolution );
    oa << BOOST_SERIALIZATION_NVP ( stdvRGBKMatrix );
    oa << BOOST_SERIALIZATION_NVP ( stdvRGBDistortionCoeff );
    oa << BOOST_SERIALIZATION_NVP ( stdvRGBRotationVectors );
    oa << BOOST_SERIALIZATION_NVP ( stdvRGBRotationMatrices );
    oa << BOOST_SERIALIZATION_NVP ( stdvRGBTranslationVectors );
    oa << BOOST_SERIALIZATION_NVP ( stdvRGB2DCorners );
    oa << BOOST_SERIALIZATION_NVP ( stdvIRKMatrix );
    oa << BOOST_SERIALIZATION_NVP ( stdvIRDistortionCoeff );
    oa << BOOST_SERIALIZATION_NVP ( stdvIRRotationVectors );
    oa << BOOST_SERIALIZATION_NVP ( stdvIRRotationMatrices );
    oa << BOOST_SERIALIZATION_NVP ( stdvIRTranslationVectors );
    oa << BOOST_SERIALIZATION_NVP ( stdvIR2DCorners );
    oa << BOOST_SERIALIZATION_NVP ( stdv3DCorners );

    return;
}

void CCalibrateKinect::load()
{
    // create and open a character archive for input
    std::ifstream ifs ( "CalibrateKinect.xml" );
    boost::archive::xml_iarchive ia ( ifs );

    //rgb
    std::vector< std::vector< double > > stdvRGBKMatrix;
    std::vector< std::vector< double > > stdvRGBDistortionCoeff;
    std::vector< std::vector< std::vector< double > > > stdvRGBRotationVectors;
    std::vector< std::vector< std::vector< double > > > stdvRGBRotationMatrices;
    std::vector< std::vector< std::vector< double > > > stdvRGBTranslationVectors;
    std::vector< std::vector< std::vector< float  > > > stdvRGB2DCorners;
    //ir
    std::vector< std::vector< double > > stdvIRKMatrix;
    std::vector< std::vector< double > > stdvIRDistortionCoeff;
    std::vector< std::vector< std::vector< double > > > stdvIRRotationVectors;
    std::vector< std::vector< std::vector< double > > > stdvIRRotationMatrices;
    std::vector< std::vector< std::vector< double > > > stdvIRTranslationVectors;
    std::vector< std::vector< std::vector< float  > > > stdvIR2DCorners;
    //both
    std::vector< std::vector< int > >    stdvImageResolution;
    std::vector< std::vector< std::vector< float  > > > stdv3DCorners;

    ia >> BOOST_SERIALIZATION_NVP ( _NUM_CORNERS_X );
    ia >> BOOST_SERIALIZATION_NVP ( _NUM_CORNERS_Y );
    ia >> BOOST_SERIALIZATION_NVP ( _X );
    ia >> BOOST_SERIALIZATION_NVP ( _Y );
    ia >> BOOST_SERIALIZATION_NVP ( _uViews );
    ia >> BOOST_SERIALIZATION_NVP ( _vstrRGBPathName );
    ia >> BOOST_SERIALIZATION_NVP ( _vstrIRPathName );
    ia >> BOOST_SERIALIZATION_NVP ( stdvImageResolution );
    ia >> BOOST_SERIALIZATION_NVP ( stdvRGBKMatrix );
    ia >> BOOST_SERIALIZATION_NVP ( stdvRGBDistortionCoeff );
    ia >> BOOST_SERIALIZATION_NVP ( stdvRGBRotationVectors );
    ia >> BOOST_SERIALIZATION_NVP ( stdvRGBRotationMatrices );
    ia >> BOOST_SERIALIZATION_NVP ( stdvRGBTranslationVectors );
    ia >> BOOST_SERIALIZATION_NVP ( stdvRGB2DCorners );
    ia >> BOOST_SERIALIZATION_NVP ( stdvIRKMatrix );
    ia >> BOOST_SERIALIZATION_NVP ( stdvIRDistortionCoeff );
    ia >> BOOST_SERIALIZATION_NVP ( stdvIRRotationVectors );
    ia >> BOOST_SERIALIZATION_NVP ( stdvIRRotationMatrices );
    ia >> BOOST_SERIALIZATION_NVP ( stdvIRTranslationVectors );
    ia >> BOOST_SERIALIZATION_NVP ( stdvIR2DCorners );
    ia >> BOOST_SERIALIZATION_NVP ( stdv3DCorners );


    //rgb
    stdvRGBKMatrix           >> _mRGBK;
    stdvRGBDistortionCoeff   >> _mRGBDistCoeffs;
    stdvRGBRotationVectors   >> _vmRGBRotationVectors;
    stdvRGBRotationMatrices  >> _vmRGBRotationMatrices;
    stdvRGBTranslationVectors >> _vmRGBTranslationVectors;
    stdvRGB2DCorners         >> _vvRGB2DCorners;
    //ir
    stdvIRKMatrix           >> _mIRK;
    stdvIRDistortionCoeff   >> _mIRDistCoeffs;
    stdvIRRotationVectors   >> _vmIRRotationVectors;
    stdvIRRotationMatrices  >> _vmIRRotationMatrices;
    stdvIRTranslationVectors >> _vmIRTranslationVectors;
    stdvIR2DCorners         >> _vvIR2DCorners;
    //both
    stdvImageResolution   >> _vImageResolution;
    Eigen::Matrix3d mK;
    stdv3DCorners         >> _vv3DCorners;

    cv::invert ( _mRGBK, _mRGBInvK, cv::DECOMP_SVD );
    cv::invert ( _mIRK,  _mIRInvK,  cv::DECOMP_SVD );

    generateMapXY4UndistortRGB();
    generateMapXY4UndistortIR();

    _eimRGBK << _mRGBK;
    _eimRGBInvK << _mRGBInvK;
    _eimIRK  << _mIRK;
    _eimIRInvK << _mIRInvK;
    _eimRelativeRotation << _cvmRelativeRotation;
    _eivRelativeTranslation << _cvmRelativeTranslation;

    return;
}
/*
void CCalibrateKinect::exportKinectIntrinsicsYML()
{
	//cout << " exportKinectIntrinsics() \n"<< flush;
	// create and open a character archive for output
	std::ofstream ofs ( "kinect_intrinsics.xml" );
	boost::archive::xml_oarchive oa ( ofs );

	//convert non-standard variables into std::vectors

	//rgb
	std::vector< std::vector< double > > stdvRGBKMatrix;
	std::vector< std::vector< double > > stdvRGBDistortionCoeff;
	//ir
	std::vector< std::vector< double > > stdvIRKMatrix;
	std::vector< std::vector< double > > stdvIRDistortionCoeff;
	//both
	std::vector< std::vector< int > >    stdvImageResolution;
	std::vector< std::vector< double > > stdvRelativeRotaion;
	std::vector< std::vector< double > > stdvRelativeTranslation;

	//rgb
	stdvRGBKMatrix          << _mRGBK;
	stdvRGBDistortionCoeff  << _mRGBDistCoeffs;
	//ir
	stdvIRKMatrix           << _mIRK;
	stdvIRDistortionCoeff   << _mIRDistCoeffs;
	//both
	stdvImageResolution     << _vImageResolution;
	stdvRelativeRotaion     << _cvmRelativeRotation;
	stdvRelativeTranslation << _cvmRelativeTranslation;


	oa << BOOST_SERIALIZATION_NVP ( stdvImageResolution );
	oa << BOOST_SERIALIZATION_NVP ( stdvRGBKMatrix );
	oa << BOOST_SERIALIZATION_NVP ( stdvRGBDistortionCoeff );
	oa << BOOST_SERIALIZATION_NVP ( stdvIRKMatrix );
	oa << BOOST_SERIALIZATION_NVP ( stdvIRDistortionCoeff );
	oa << BOOST_SERIALIZATION_NVP ( stdvRelativeRotaion );
	oa << BOOST_SERIALIZATION_NVP ( stdvRelativeTranslation );


	return;
}
*/
void CCalibrateKinect::exportKinectIntrinsics()
{
	//cout << " exportKinectIntrinsics() \n"<< flush;
    // create and open a character archive for output
    std::ofstream ofs ( "kinect_intrinsics.xml" );
    boost::archive::xml_oarchive oa ( ofs );

    //convert non-standard variables into std::vectors

    //rgb
    std::vector< std::vector< double > > stdvRGBKMatrix;
    std::vector< std::vector< double > > stdvRGBDistortionCoeff;
    //ir
    std::vector< std::vector< double > > stdvIRKMatrix;
    std::vector< std::vector< double > > stdvIRDistortionCoeff;
    //both
    std::vector< std::vector< int > >    stdvImageResolution;
    std::vector< std::vector< double > > stdvRelativeRotaion;
    std::vector< std::vector< double > > stdvRelativeTranslation;

    //rgb
    stdvRGBKMatrix          << _mRGBK;
    stdvRGBDistortionCoeff  << _mRGBDistCoeffs;
    //ir
    stdvIRKMatrix           << _mIRK;
    stdvIRDistortionCoeff   << _mIRDistCoeffs;
    //both
    stdvImageResolution     << _vImageResolution;
    stdvRelativeRotaion     << _cvmRelativeRotation;
    stdvRelativeTranslation << _cvmRelativeTranslation;


    oa << BOOST_SERIALIZATION_NVP ( stdvImageResolution );
    oa << BOOST_SERIALIZATION_NVP ( stdvRGBKMatrix );
    oa << BOOST_SERIALIZATION_NVP ( stdvRGBDistortionCoeff );
    oa << BOOST_SERIALIZATION_NVP ( stdvIRKMatrix );
    oa << BOOST_SERIALIZATION_NVP ( stdvIRDistortionCoeff );
    oa << BOOST_SERIALIZATION_NVP ( stdvRelativeRotaion );
    oa << BOOST_SERIALIZATION_NVP ( stdvRelativeTranslation );


    return;
}

void CCalibrateKinect::importKinectIntrinsics()
{
    // create and open a character archive for output
#if __linux__
    std::ifstream ifs ( "/space/csxsl/src/opencv-shuda/Data/kinect_intrinsics.xml" );
#else if _WIN32 || _WIN64
	std::ifstream ifs ( "C:\\csxsl\\src\\opencv-shuda\\Data\\kinect_intrinsics.xml" );
#endif
    boost::archive::xml_iarchive ia ( ifs );

    //convert non-standard variables into std::vectors

    //rgb
    std::vector< std::vector< double > > stdvRGBKMatrix;
    std::vector< std::vector< double > > stdvRGBDistortionCoeff;
    //ir
    std::vector< std::vector< double > > stdvIRKMatrix;
    std::vector< std::vector< double > > stdvIRDistortionCoeff;
    //both
    std::vector< std::vector< int > >    stdvImageResolution;
    std::vector< std::vector< double > > stdvRelativeRotaion;
    std::vector< std::vector< double > > stdvRelativeTranslation;


    ia >> BOOST_SERIALIZATION_NVP ( stdvImageResolution );
    ia >> BOOST_SERIALIZATION_NVP ( stdvRGBKMatrix );
    ia >> BOOST_SERIALIZATION_NVP ( stdvRGBDistortionCoeff );
    ia >> BOOST_SERIALIZATION_NVP ( stdvIRKMatrix );
    ia >> BOOST_SERIALIZATION_NVP ( stdvIRDistortionCoeff );
    ia >> BOOST_SERIALIZATION_NVP ( stdvRelativeRotaion );
    ia >> BOOST_SERIALIZATION_NVP ( stdvRelativeTranslation );

    //rgb
    stdvRGBKMatrix          >> _mRGBK;
    stdvRGBDistortionCoeff  >> _mRGBDistCoeffs;
    //ir
    stdvIRKMatrix           >> _mIRK;
    stdvIRDistortionCoeff   >> _mIRDistCoeffs;
    //both
    stdvImageResolution     >> _vImageResolution;
    stdvRelativeRotaion     >> _cvmRelativeRotation;
    stdvRelativeTranslation >> _cvmRelativeTranslation;

    cv::invert ( _mRGBK, _mRGBInvK, cv::DECOMP_SVD );
    cv::invert ( _mIRK,  _mIRInvK,  cv::DECOMP_SVD );

    generateMapXY4UndistortRGB();
    generateMapXY4UndistortIR();

    _eimRGBK << _mRGBK;
    _eimRGBInvK << _mRGBInvK;
    _eimIRK  << _mIRK;
    _eimIRInvK << _mIRInvK;
    _eimRelativeRotation << _cvmRelativeRotation;
    _eivRelativeTranslation << _cvmRelativeTranslation;

    /*
    	PRINT( _mRGBK );
    	PRINT( _mRGBInvK );
    	PRINT( _mRGBDistCoeffs );
    	PRINT( _vImageResolution );
    	PRINT( _mIRK );
    	PRINT( _mIRInvK );
    	PRINT( _mIRDistCoeffs );
    */
    return;
}

void CCalibrateKinect::importKinectIntrinsicsYML()
{
    // create and open a character archive for output
#if __linux__
    cv::FileStorage cFSRead( "/space/csxsl/src/opencv-shuda/Data/kinect_intrinsics.yml", cv::FileStorage::READ );
#else if _WIN32 || _WIN64
	cv::FileStorage cFSRead ( "C:\\csxsl\\src\\opencv-shuda\\Data\\kinect_intrinsics.yml", cv::FileStorage::READ );
#endif

	cFSRead ["mRGBK"] >> _mRGBK;
	cFSRead ["mRGBDistCoeffs"] >> _mRGBDistCoeffs;
	cv::Mat_<int> _cvmImageResolution; std::vector<int> vTemp;
	cFSRead ["cvmImageResolution"] >> _cvmImageResolution;
	_cvmImageResolution >> vTemp;	vTemp >> _vImageResolution;
	_nRows = _vImageResolution(1);
	_nCols = _vImageResolution(0);
	cFSRead ["mIRK"] >> _mIRK;
	cFSRead ["mIRDistCoeffs"] >> _mIRDistCoeffs;
	cFSRead ["cvmRelativeRotation"] >> _cvmRelativeRotation;
	cFSRead ["cvmRelativeTranslation"] >> _cvmRelativeTranslation;

	cFSRead.release();

    cv::invert ( _mRGBK, _mRGBInvK, cv::DECOMP_SVD );
    cv::invert ( _mIRK,  _mIRInvK,  cv::DECOMP_SVD );

    generateMapXY4UndistortRGB();
    generateMapXY4UndistortIR();

    _eimRGBK << _mRGBK;
    _eimRGBInvK << _mRGBInvK;
    _eimIRK  << _mIRK;
    _eimIRInvK << _mIRInvK;
    _eimRelativeRotation << _cvmRelativeRotation;
    _eivRelativeTranslation << _cvmRelativeTranslation;

    /*
    	PRINT( _mRGBK );
    	PRINT( _mRGBInvK );
    	PRINT( _mRGBDistCoeffs );
    	PRINT( _vImageResolution );
    	PRINT( _mIRK );
    	PRINT( _mIRInvK );
    	PRINT( _mIRDistCoeffs );
    */
    return;
}


void CKinectView::setIntrinsics ( unsigned int nScaleViewport_, int nCameraType_, double dNear_, double dFar_ )
{
    // set intrinsics
    double dWidth = _cCK.imageResolution() ( 0 ) * nScaleViewport_;
    double dHeight = _cCK.imageResolution() ( 1 ) * nScaleViewport_;
//    glutReshapeWindow( int ( dWidth ), int ( dHeight ) );

    //PRINT(dWidth);
    //PRINT(dHeight);

    glMatrixMode ( GL_PROJECTION );

    Eigen::Matrix3d mK;
    mK = _cCK.eiMatK ( nCameraType_ );

    //PRINT( mK );
    double u = mK ( 0, 2 );
    double v = mK ( 1, 2 );
    double f = ( mK ( 0, 0 ) + mK ( 1, 1 ) ) / 2.;
    //no need to times nScaleViewport_ factor, because v/f, -(dHeight -v)/f cancel the factor off.
    double dNear = dNear_;
    double dLeft, dRight, dBottom, dTop;
    //Two assumptions:
    //1. assuming the principle point is inside the image
    //2. assuming the x axis pointing right and y axis pointing upwards
    dTop    =              v  / f;
    dBottom = - ( dHeight - v ) / f;
    dLeft   =             -u  / f;
    dRight  = ( dWidth   - u ) / f;

    glLoadIdentity(); //use the same style as page 130, opengl programming guide
    glFrustum ( dLeft * dNear, dRight * dNear, dBottom * dNear, dTop * dNear, dNear, dFar_ );
    glMatrixMode ( GL_VIEWPORT );

    if ( nScaleViewport_ == 2 )
    {
        glViewport ( 0, - ( GLsizei ) dHeight, ( GLsizei ) dWidth * nScaleViewport_, ( GLsizei ) dHeight * nScaleViewport_ );
    }
    else if ( nScaleViewport_ == 1 )
    {
        glViewport ( 0, 0, ( GLsizei ) dWidth, ( GLsizei ) dHeight );
    }

    // set intrinsics end.
    return;
}


GLuint CKinectView::LoadTexture ( const cv::Mat& img )
{
    GLuint uTexture;
    glPixelStorei ( GL_UNPACK_ALIGNMENT, 1 );

    glGenTextures ( 1, &uTexture );
    glBindTexture ( GL_TEXTURE_2D, uTexture );
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ); // cheap scaling when image bigger than texture
    glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ); // cheap scaling when image smalled than texture
    // 2d texture, level of detail 0 (normal), 3 components (red, green, blue), x size from image, y size from image,
    // border 0 (normal), rgb color data, unsigned byte data, and finally the data itself.
    if( 3 == img.channels())
        glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, img.data ); //???????????????????
    else if( 1 == img.channels())
        glTexImage2D ( GL_TEXTURE_2D, 0, GL_INTENSITY, img.cols, img.rows, 0, GL_INTENSITY, GL_UNSIGNED_BYTE, img.data );
        //glTexEnvi ( GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_REPEAT );

    // 2d texture, 3 colors, width, height, RGB in that order, byte data, and the data.
    //gluBuild2DMipmaps ( GL_TEXTURE_2D, GL_RGB, img.cols, img.rows,  GL_RGB, GL_UNSIGNED_BYTE, img.data );
    return uTexture;
}
void CKinectView::renderOnImage ( int nX_, int nY_ )
{
    Eigen::Matrix3d mK = _cCK.eiMatK ( CCalibrateKinect::RGB_CAMERA );

    const double u = mK ( 0, 2 );
    const double v = mK ( 1, 2 );
    const double f = ( mK ( 0, 0 ) + mK ( 1, 1 ) ) / 2.;

    // Draw principle point
    double dPhysicalFocalLength = .015;
    double dY =  v - nY_;
    dY /= f;
    dY *= dPhysicalFocalLength;
    double dX = -u + nX_;
    dX /= f;
    dX *= dPhysicalFocalLength;

    //draw principle point
    glVertex3d ( dX, dY, -dPhysicalFocalLength );
}
void CKinectView::renderCamera ( GLuint uTexture_, int nCameraType_, int nCameraRender_, double dPhysicalFocalLength_ ) const //dPhysicalFocalLength_ = .02 by default
{
    Eigen::Matrix3d mK = _cCK.eiMatK ( nCameraType_ );

    const double u = mK ( 0, 2 );
    const double v = mK ( 1, 2 );
    const double f = ( mK ( 0, 0 ) + mK ( 1, 1 ) ) / 2.;
    const double dW = _cCK.imageResolution() ( 0 );
    const double dH = _cCK.imageResolution() ( 1 );

    // Draw principle point
    double dT =  v;
    dT /= f;
    dT *= dPhysicalFocalLength_;
    double dB =  v - dH;
    dB /= f;
    dB *= dPhysicalFocalLength_;
    double dL = -u;
    dL /= f;
    dL *= dPhysicalFocalLength_;
    double dR = -u + dW;
    dR /= f;
    dR *= dPhysicalFocalLength_;

    glPushAttrib ( GL_CURRENT_BIT );
/*
    //draw principle point
    glColor3d ( 0., 0., 1. );
    glPointSize ( 5 );
    glBegin ( GL_POINTS );
    glVertex3d ( 0, 0, -dPhysicalFocalLength_ );
    glEnd();

    //draw principle axis
    glColor3d ( 0., 0., 1. );
    glLineWidth ( 1 );
    glBegin ( GL_LINES );
    glVertex3d ( 0, 0, 0 );
    glVertex3d ( 0, 0, -dPhysicalFocalLength_ );
    glEnd();

    //draw x axis in camera view
    glColor3d ( 1., 0., 0. ); //x
    glBegin ( GL_LINES );
    glVertex3d ( 0, 0, -dPhysicalFocalLength_ );
    glVertex3d ( dR, 0, -dPhysicalFocalLength_ );
    glEnd();

    //draw y axis in camera view
    glColor3d ( 0., 1., 0. ); //y
    glBegin ( GL_LINES );
    glVertex3d ( 0, 0, -dPhysicalFocalLength_ );
    glVertex3d ( 0, dT, -dPhysicalFocalLength_ );
    glEnd();
*/
    glPopAttrib();

    //draw frame
    if ( ALL_CAMERA == nCameraRender_ )
    {
        glEnable ( GL_TEXTURE_2D );
        glTexEnvf ( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
        glBindTexture ( GL_TEXTURE_2D, uTexture_ );

        //glColor3d(1., 1., 1.); glLineWidth(.5);
        glBegin ( GL_QUADS );
        glTexCoord2f ( 0.0, 0.0 );
        glVertex3d ( dL, dT, -dPhysicalFocalLength_ );
        glTexCoord2f ( 0.0, 1.0 );
        glVertex3d ( dL, dB, -dPhysicalFocalLength_ );
        glTexCoord2f ( 1.0, 1.0 );
        glVertex3d ( dR, dB, -dPhysicalFocalLength_ );
        glTexCoord2f ( 1.0, 0.0 );
        glVertex3d ( dR, dT, -dPhysicalFocalLength_ );
        glEnd();
        glDisable ( GL_TEXTURE_2D );
    }
/*
    //glColor3d(1., 1., 1.); glLineWidth(1.);
    glBegin ( GL_LINES );
    glVertex3d ( 0,   0, 0 );
    glVertex3d ( dL, dT, -dPhysicalFocalLength_ );
    glEnd();
    glBegin ( GL_LINES );
    glVertex3d ( 0,   0, 0 );
    glVertex3d ( dR, dT, -dPhysicalFocalLength_ );
    glEnd();
    glBegin ( GL_LINES );
    glVertex3d ( 0,   0, 0 );
    glVertex3d ( dR, dB, -dPhysicalFocalLength_ );
    glEnd();
    glBegin ( GL_LINES );
    glVertex3d ( 0,   0, 0 );
    glVertex3d ( dL, dB, -dPhysicalFocalLength_ );
    glEnd();
*/
    return;
}

} //namespace videosource
} //namespace extra
} //namespace btl
