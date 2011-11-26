#include "calibratekinect.hpp"
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/lexical_cast.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <map>


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

    importKinectIntrinsics();

    //definition of parameters
    _dThreshouldDepth = 10;

    // allocate memory for later use ( registrate the depth with rgb image
    _pPxDIR   = new unsigned short[ 307200*3 ]; //2D coordinate along with depth for ir image
    _pPxRGB   = new unsigned short[ 307200*2 ]; //2D coordinate in rgb image
    _pIRWorld = new double[ 307200*3 ]; //XYZ w.r.t. IR camera reference system

    // refreshed for every frame
    _pRGBWorld    = new double[ 307200*3 ];//X,Y,Z coordinate of depth w.r.t. RGB camera reference system
    _pRGBWorldRGB = new double[ 307200*3 ];//registered to RGB image of the X,Y,Z coordinate

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

    _fxIR = eiMatIRK() ( 0, 0 );
    _fyIR = eiMatIRK() ( 1, 1 );
    _uIR  = eiMatIRK() ( 0, 2 );
    _vIR  = eiMatIRK() ( 1, 2 );

    _fxRGB = eiMatRGBK() ( 0, 0 );
    _fyRGB = eiMatRGBK() ( 1, 1 );
    _uRGB  = eiMatRGBK() ( 0, 2 );
    _vRGB  = eiMatRGBK() ( 1, 2 );

	_nPatternType = SQUARE;
    //define 3D pattern corners
    definePattern ( _X, _Y, _NUM_CORNERS_X, _NUM_CORNERS_Y, _nPatternType, &_vPatterCorners3D );

    cout << "CCalibrateKinect() done." << endl;
    return;
}
CCalibrateKinect::~CCalibrateKinect()
{
    delete [] _pIRWorld;
    delete [] _pPxDIR;
    delete [] _pPxRGB;
    delete [] _pRGBWorld;
    delete [] _pRGBWorldRGB;
}

Matrix3d CCalibrateKinect::eiMatK ( int nCameraType_ ) const
{
    Matrix3d eimK;

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
    ifstream iFile ( "/space/csxsl/src/opencv-shuda/Data/control.yaml" );
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
    map< string, int >::const_iterator cIt1;

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

	std::map <string, string> mpType;
    parser.GetNextDocument ( doc );
    doc[0] >> mpType;
    string strGridType = mpType["grid type"];
	if ( "circle" == strGridType )
		_nPatternType = CIRCLE;
	else if( "chessboard" == strGridType )
		_nPatternType = SQUARE;
		
    // properties
    std::map < std::string, vector< int > > mpCornerCounts;
    parser.GetNextDocument ( doc );
    doc[0] >> mpCornerCounts;
    //PRINT( mpCornerCounts );
    //process mpCornerCounts
    vector< int > vCornerCounts;
    vCornerCounts = mpCornerCounts["No. of Corners X Y"];
    _NUM_CORNERS_X = vCornerCounts[0];
    _NUM_CORNERS_Y = vCornerCounts[1];

    std::map < string, vector< float > > mpUnitLength;
    parser.GetNextDocument ( doc );
    doc[0] >> mpUnitLength;
    //PRINT( mpUnitLength );
    //process mpUnitLength
    vector< float > vUnitLength;
    vUnitLength = mpUnitLength["Unit length in X Y"];
    _X = vUnitLength[0];
    _Y = vUnitLength[1];

    std::map <string, string> mpProperties;
    parser.GetNextDocument ( doc );
    doc[0] >> mpProperties;
    string _strImageDirectory = mpProperties["image directory"];
    //PRINT( mpProperties );
	
	vector< int > vIdx;
	parser.GetNextDocument ( doc );
	for ( unsigned i = 0; i < doc.size(); i++ )
	{
		doc[i] >> vIdx;
	}

    vector<string> vRGBNames;
	vector<string> vDepthNames;
	vector<string> vUndistRGB;
	vector<string> vUndistDepth;

	for ( vector< int >::const_iterator cit = vIdx.begin(); cit != vIdx.end(); cit ++ )
	{
		string strNum = boost::lexical_cast<string> ( *cit );
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
        _vImageResolution ( 0 ) = _vRGBs[0].cols;
        _vImageResolution ( 1 ) = _vRGBs[0].rows;
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

        //convert rotation vectors to rotation matrices
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
        cout << "serialized to XML \n" ;
    }

    if ( 1 == _nSerializeFromXML )
    {
        load();
        cout << "serialized from XML \n" ;
    }

    exportKinectIntrinsics();
	cout << " intrinsics exported \n" ;
    importKinectIntrinsics();
	cout << " intrinsics imported \n" ;
    return;
}
void CCalibrateKinect::loadImages ( const boost::filesystem::path& cFullPath_, const std::vector< std::string >& vImgNames_, std::vector< cv::Mat >* pvImgs_ ) const
{
    pvImgs_->clear();

    string strPathName  = cFullPath_.string();

    for ( unsigned int i = 0; i < vImgNames_.size(); i++ )
    {
        std::string strRGBFileName = strPathName + vImgNames_[i]; //saved into the folder from which the KinectCalibrationDemo is being run.
        //PRINT( strRGBFileName );
        CHECK ( boost::filesystem::exists ( strRGBFileName ), "Not found: " + strRGBFileName + "\n" );
        Mat img = cv::imread ( strRGBFileName );
        cvtColor ( img, img, CV_BGR2RGB );
        pvImgs_->push_back ( img );
    }

    return;
}
void CCalibrateKinect::exportImages ( const boost::filesystem::path& cFullPath_, const vector< std::string >& vImgNames_, const std::vector< cv::Mat >& vImgs_ ) const
{
    string strPathName  = cFullPath_.string();

    for ( unsigned int n = 0; n < vImgNames_.size(); n++ )
    {
        Mat img = vImgs_[n];
        cvtColor ( img, img, CV_RGB2BGR );
        cv::imwrite ( strPathName + vImgNames_[n], vImgs_[n] );
    }

    return;
}


void CCalibrateKinect::locate2DCorners ( const vector< Mat >& vImages_, const int& nX_, const int& nY_, vector< vector<cv::Point2f> >* pvv2DCorners_, int nPatternType_ ) const //nPatternType_ = SQUARE
{

    CHECK ( !vImages_.empty(), "locate2DCorners(): no images.\n" );

    if ( SQUARE == nPatternType_ )
    {
		std::cout << " locate chessboard corners.\n ";
        pvv2DCorners_->clear();

        cv::Size patternSize ( nX_, nY_ );

        for ( unsigned int i = 0; i < vImages_.size(); i++ )
        {
            const Mat& cvFrame = vImages_[i] ;

            vector<cv::Point2f> vCurrentCorners;//float 2d point is required by the OpenCV API.
            //locate corners roughly
            bool _bChessBoardCornersFoundThisFrame = cv::findChessboardCorners ( cvFrame, patternSize, vCurrentCorners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS );

            CHECK ( _bChessBoardCornersFoundThisFrame, " No corners are found.\n" );
			PRINT( vCurrentCorners.size() );
            //locate corners in sub-pixel level
            Mat cvFrameGrey;
            cv::cvtColor ( cvFrame, cvFrameGrey, CV_BGR2GRAY );
            cv::cornerSubPix ( cvFrameGrey, vCurrentCorners, cv::Size ( 9, 9 ), cv::Size ( -1, -1 ), cv::TermCriteria ( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1 ) );

            //store the corners inpto a vector
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
            const Mat& cvFrame = vImages_[i] ;
			Mat cvmTmp;
			cv::cvtColor ( cvFrame, cvmTmp, CV_RGB2GRAY );

            vector<cv::Point2f> vCurrentCorners;//float 2d point is required by the OpenCV API.
            //locate corners roughly
            bool _bChessBoardCornersFoundThisFrame = cv::findCirclesGrid ( cvmTmp, patternSize, vCurrentCorners, CALIB_CB_ASYMMETRIC_GRID );

			PRINT( vCurrentCorners.size() );

            //CHECK ( _bChessBoardCornersFoundThisFrame, " No corners are found.\n" );

            //store the corners inpto a vector
            pvv2DCorners_->push_back ( vCurrentCorners );
        }
    }

    return;
}

void CCalibrateKinect::definePattern (  const float& fX_, const float& fY_, const int& nX_, const int& nY_,  const int& nPatternType_, vector<cv::Point3f>* pv3DPatternCorners_ ) const
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

void CCalibrateKinect::define3DCorners ( const vector<cv::Point3f>& vPattern_, const unsigned int& nViews_, vector< vector<cv::Point3f> >* pvv3DCorners_  ) const
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
    _mRGBK = Mat_<double> ( 3, 3 );
    _mRGBDistCoeffs = Mat_<double> ( 5, 1 );

    _mRGBK.at<double> ( 0, 0 ) = 1;
    _mRGBK.at<double> ( 1, 1 ) = 1;

    if ( _vRGBs.empty() )
    {
        CError cE;
        cE << CErrorInfo ( " No images have been loaded. \n" );
        throw cE;
    }

    cv::Size cvFrameSize ( _vImageResolution[0], _vImageResolution[1] );

    //calibrate the rgb camera
    double dErrorRGB = cv::calibrateCamera ( _vv3DCorners, _vvRGB2DCorners, cvFrameSize, _mRGBK, _mRGBDistCoeffs, _vmRGBRotationVectors, _vmRGBTranslationVectors );
	cout << "calibrate RGBs.\n";
    double dErrorIR  = cv::calibrateCamera ( _vv3DCorners, _vvIR2DCorners,  cvFrameSize, _mIRK,  _mIRDistCoeffs,  _vmIRRotationVectors,  _vmIRTranslationVectors  );
	cout << "calibrate IRs.\n";

    Mat E, F; // E is essential matrix and F is the fundamental matrix
    double dErrorStereo = cv::stereoCalibrate ( _vv3DCorners, _vvRGB2DCorners, _vvIR2DCorners, _mRGBK, _mRGBDistCoeffs, _mIRK, _mIRDistCoeffs, cvFrameSize, _cvmRelativeRotation, _cvmRelativeTranslation, E, F );

    cv::invert ( _mRGBK, _mRGBInvK, DECOMP_SVD );
    cv::invert ( _mIRK, _mIRInvK, DECOMP_SVD );
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
* @brief convert rotation vectors into rotation matrices using cv::
*
* @param
* @param argv
*/
void CCalibrateKinect::convertRV2RM ( const vector< Mat >& vMat_, vector< Mat >* pvMat_ ) const
{
    pvMat_->clear();

    for ( unsigned int n = 0; n < vMat_.size() ; n++ )
    {
        Mat cvmRGB;
        cv::Rodrigues ( vMat_[n], cvmRGB );
        pvMat_->push_back ( cvmRGB );
    }

    /*
        PRINT( _cvmRelativeRotation );
        PRINT( _cvmRelativeTranslation );
        Mat vRT  = (Mat_<double>(1,3) << 0, 0, 0);
        Mat vRR0 = (Mat_<double>(3,1) << 0, 0, 0);
        Mat mRR, mRGBInvR;
        for(unsigned int i = 0; i < views(); i++)
        {
            vRT += _vmIRTranslationVectors[i] - _vmRGBTranslationVectors[i];
            cv::invert( _vmRGBRotationMatrices[i], mRGBInvR, DECOMP_SVD );
            mRR = _vmIRRotationMatrices[i] * mRGBInvR;
            Mat vRR;
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

void CCalibrateKinect::undistortRGB ( const Mat& cvmRGB_, Mat& Undistorted_ ) const
{
    cv::remap ( cvmRGB_, Undistorted_, _cvmMapXYRGB, _cvmMapY, cv::INTER_NEAREST, cv::BORDER_CONSTANT );
}

void CCalibrateKinect::undistortIR ( const Mat& cvmIR_, Mat& Undistorted_ ) const
{
    cv::remap ( cvmIR_, Undistorted_, _cvmMapXYIR, _cvmMapY, cv::INTER_NEAREST, cv::BORDER_CONSTANT );
}
void CCalibrateKinect::generateMapXY4UndistortRGB()
{
    CHECK ( !_mRGBK.empty(),         "undistortImages(): K matrix cannot be empty.\n" );
    CHECK ( !_mRGBInvK.empty(),      "undistortImages(): inverse of K matrix cannot be empty.\n" );
    CHECK ( !_mRGBDistCoeffs.empty(), "undistortImages(): distortion coefficients cannot be empty.\n" );

    map4UndistortImage<double> ( _vImageResolution, _mRGBK, _mRGBInvK, _mRGBDistCoeffs, &_cvmMapXYRGB );
}
void CCalibrateKinect::generateMapXY4UndistortIR()
{
    CHECK ( !_mIRK.empty(),         "undistortImages(): K matrix cannot be empty.\n" );
    CHECK ( !_mIRInvK.empty(),      "undistortImages(): inverse of K matrix cannot be empty.\n" );
    CHECK ( !_mIRDistCoeffs.empty(), "undistortImages(): distortion coefficients cannot be empty.\n" );

    map4UndistortImage<double> ( _vImageResolution, _mIRK, _mIRInvK, _mIRDistCoeffs, &_cvmMapXYIR );
}
void CCalibrateKinect::undistortImages ( const vector< Mat >& vImages_,  const Mat_<double>& cvmK_, const Mat_<double>& cvmInvK_, const Mat_<double>& cvmDistCoeffs_, vector< Mat >* pvUndistorted_ ) const
{
	cout << "undistortImages() "<< endl << flush;
    CHECK ( !vImages_.empty(),      "undistortImages(): # of undistorted images can not be zero.\n" );
    CHECK ( !cvmK_.empty(),         "undistortImages(): K matrix cannot be empty.\n" );
    CHECK ( !cvmInvK_.empty(),      "undistortImages(): inverse of K matrix cannot be empty.\n" );
    CHECK ( !cvmDistCoeffs_.empty(), "undistortImages(): distortion coefficients cannot be empty.\n" );

    cv::Size cvFrameSize = vImages_[0].size(); //x,y;
    pvUndistorted_->clear();

    for ( unsigned int n = 0; n < vImages_.size(); n++ )
    {
        Mat cvUndistorted;
		cout << "distort: "<< n << "-th image.\n"<< flush;
        undistortImage ( vImages_[n],  cvmK_, cvmInvK_, cvmDistCoeffs_, &cvUndistorted );
        pvUndistorted_->push_back ( cvUndistorted );

        //string strNum = boost::lexical_cast<string> ( n );
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

    //convert non-standard variables into vectors

    //rgb
    vector< vector< double > > stdvRGBKMatrix;
    vector< vector< double > > stdvRGBDistortionCoeff;
    vector< vector< vector< double > > > stdvRGBRotationVectors;
    vector< vector< vector< double > > > stdvRGBRotationMatrices;
    vector< vector< vector< double > > > stdvRGBTranslationVectors;
    vector< vector< vector< float  > > > stdvRGB2DCorners;
    //ir
    vector< vector< double > > stdvIRKMatrix;
    vector< vector< double > > stdvIRDistortionCoeff;
    vector< vector< vector< double > > > stdvIRRotationVectors;
    vector< vector< vector< double > > > stdvIRRotationMatrices;
    vector< vector< vector< double > > > stdvIRTranslationVectors;
    vector< vector< vector< float  > > > stdvIR2DCorners;
    //both
    vector< vector< int > >    stdvImageResolution;
    vector< vector< vector< float  > > > stdv3DCorners;

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
    vector< vector< double > > stdvRGBKMatrix;
    vector< vector< double > > stdvRGBDistortionCoeff;
    vector< vector< vector< double > > > stdvRGBRotationVectors;
    vector< vector< vector< double > > > stdvRGBRotationMatrices;
    vector< vector< vector< double > > > stdvRGBTranslationVectors;
    vector< vector< vector< float  > > > stdvRGB2DCorners;
    //ir
    vector< vector< double > > stdvIRKMatrix;
    vector< vector< double > > stdvIRDistortionCoeff;
    vector< vector< vector< double > > > stdvIRRotationVectors;
    vector< vector< vector< double > > > stdvIRRotationMatrices;
    vector< vector< vector< double > > > stdvIRTranslationVectors;
    vector< vector< vector< float  > > > stdvIR2DCorners;
    //both
    vector< vector< int > >    stdvImageResolution;
    vector< vector< vector< float  > > > stdv3DCorners;

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

    cv::invert ( _mRGBK, _mRGBInvK, DECOMP_SVD );
    cv::invert ( _mIRK,  _mIRInvK,  DECOMP_SVD );

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

void CCalibrateKinect::exportKinectIntrinsics()
{
	//cout << " exportKinectIntrinsics() \n"<< flush;
    // create and open a character archive for output
    std::ofstream ofs ( "kinect_intrinsics.xml" );
    boost::archive::xml_oarchive oa ( ofs );

    //convert non-standard variables into vectors

    //rgb
    vector< vector< double > > stdvRGBKMatrix;
    vector< vector< double > > stdvRGBDistortionCoeff;
    //ir
    vector< vector< double > > stdvIRKMatrix;
    vector< vector< double > > stdvIRDistortionCoeff;
    //both
    vector< vector< int > >    stdvImageResolution;
    vector< vector< double > > stdvRelativeRotaion;
    vector< vector< double > > stdvRelativeTranslation;

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
void CCalibrateKinect::computeFundamental()
{

    return;
}

void CCalibrateKinect::importKinectIntrinsics()
{
    // create and open a character archive for output
    std::ifstream ifs ( "/space/csxsl/src/opencv-shuda/Data/kinect_intrinsics.xml" );
    boost::archive::xml_iarchive ia ( ifs );

    //convert non-standard variables into vectors

    //rgb
    vector< vector< double > > stdvRGBKMatrix;
    vector< vector< double > > stdvRGBDistortionCoeff;
    //ir
    vector< vector< double > > stdvIRKMatrix;
    vector< vector< double > > stdvIRDistortionCoeff;
    //both
    vector< vector< int > >    stdvImageResolution;
    vector< vector< double > > stdvRelativeRotaion;
    vector< vector< double > > stdvRelativeTranslation;


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

    cv::invert ( _mRGBK, _mRGBInvK, DECOMP_SVD );
    cv::invert ( _mIRK,  _mIRInvK,  DECOMP_SVD );

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

void CCalibrateKinect::registration ( const unsigned short* pDepth_ )
{
    double* pM = _pRGBWorldRGB ;

    // initialize the Registered depth as NULLs
    for ( int i = 0; i < 307200; i++ )
    {
        *pM++ = 0;
        *pM++ = 0;
        *pM++ = 0;
    }

    //3D Pt in camera coordinate system.
    //const unsigned short* pDepth = (unsigned short*)cvmDepth_.data;

    //collecting depths
    unsigned short* pMovingPxDIR = _pPxDIR;

    for ( unsigned short r = 0; r < 480; r++ )
        for ( unsigned short c = 0; c < 640; c++ )
        {
            *pMovingPxDIR++ = c;  	    //x
            *pMovingPxDIR++ = r;        //y
            *pMovingPxDIR++ = *pDepth_++;//depth
        }

    //unproject the depth map to IR coordinate
    unprojectIR      ( _pPxDIR, 307200, _pIRWorld );
    //transform from IR coordinate to RGB coordinate
    transformIR2RGB  ( _pIRWorld, 307200, _pRGBWorld );
    //project RGB coordinate to image to register the depth with rgb image
    projectRGB       ( _pRGBWorld, 307200, _pRGBWorldRGB );

    //cout << "registration() end."<< endl;
}

void CCalibrateKinect::unprojectIR ( const unsigned short* pCamera_, const int& nN_, double* pWorld_ )
{
// pCamer format
// 0 x (c) 1 y (r) 2 d
//the pixel coordinate is defined w.r.t. camera reference, which is defined as x-left, y-downward and z-forward. It's
//a right hand system.
//when rendering the point using opengl's camera reference which is defined as x-left, y-upward and z-backward. the
//for example: glVertex3d ( Pt(0), -Pt(1), -Pt(2) );
    for ( int i = 0; i < nN_; i++ )
    {
        if ( * ( pCamera_ + 2 ) > 400 )
        {
            * ( pWorld_ + 2 ) = ( * ( pCamera_ + 2 ) + 5 ) / 1000.; //convert to meter z 5 million meter is added according to experience. as the OpenNI
            //coordinate system is defined w.r.t. the camera plane which is 0.5 centimeters in front of the camera center
            * pWorld_    = ( * pCamera_    - _uIR ) / _fxIR * * ( pWorld_ + 2 ); // + 0.0025;     //x by experience.
            * ( pWorld_ + 1 ) = ( * ( pCamera_ + 1 ) - _vIR ) / _fyIR * * ( pWorld_ + 2 ); // - 0.00499814; //y the value is esimated using CCalibrateKinectExtrinsics::calibDepth(
        }
        else
        {
            * ( pWorld_ + 2 ) = 0;
        }

        pCamera_ += 3;
        pWorld_ += 3;
    }

    return;
}

void CCalibrateKinect::transformIR2RGB ( const double* pIR_, const int& nN_, double* pRGB_ )
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
        if ( abs ( * ( pIR_ + 2 ) ) < 0.0000001 )
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

void CCalibrateKinect::projectRGB ( double* pWorld_, const int& nN_, double* pRGBWorld_ )
{
// pWorld_ is the a 640*480 matrix aranged the same way as depth map
// pRGBWorld_ is another 640*480 matrix aranged the same wey as rgb image.
// this is much faster than the function
// eiv2DPt = mK * vPt; eiv2DPt /= eiv2DPt(2);
    //cout << "projectRGB() starts." << endl;
    unsigned short nX, nY;
    int nIdx;

    for ( int i = 0; i < nN_; i++ )
    {
        if ( abs ( * ( pWorld_ + 2 ) ) > 0.0000001 )
        {
            // get 2D image projection in RGB image of the XYZ in the world
            nX = ( _fxRGB * ( * pWorld_   ) / * ( pWorld_ + 2 ) + _uRGB + 0.5 );
            nY = ( _fyRGB * ( * ( pWorld_ + 1 ) ) / * ( pWorld_ + 2 ) + _vRGB + 0.5 );

            // set 2D rgb XYZ
            if ( nX >= 0 && nX < 640 && nY >= 0 && nY < 480 )
            {
                nIdx = ( nY * 640 + nX ) * 3;
                pRGBWorld_[ nIdx++ ] = *pWorld_;
                pRGBWorld_[ nIdx++ ] = * ( pWorld_ + 1 );
                pRGBWorld_[ nIdx   ] = * ( pWorld_ + 2 );
                //PRINT( nX ); PRINT( nY ); PRINT( pWorld_ );
            }
        }

        /*
        		pT = _sDepth._ppRGBWorld[ nY*640 + nX ];
        		PRINT( *pT++ );
        		PRINT( *pT++ );
        		PRINT( *pT++ );
        */
        pWorld_ += 3;
    }
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
