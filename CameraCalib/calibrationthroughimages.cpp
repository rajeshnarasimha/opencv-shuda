#include "calibrationthroughimages.hpp"
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/lexical_cast.hpp>

namespace shuda
{

void CCalibrationThroughImages::main ( const boost::filesystem::path& cFullPath_ )
{
    //load depth camera intrinsics
    //initializeDepthIntrinsics();

    //load images
    loadImages ( cFullPath_ );
    //std::cout << "image loaded \n" ;

    //find corners
    locate2DCorners();
    //std::cout << "corners located \n" ;
    
    //define 3D corners
    define3DCorners ( _X, _Y );
    //std::cout << "3d corners defined \n" ;

    //calibration
    calibrate();
    //std::cout << "camera calibrated \n" ;

    //convert rotation vectors to rotation matrices
    convertRV2RM();
    //std::cout << "convertRV2RM() executed \n" ;
    
    undistortImages();
    //std::cout << " undistortImages(); \n";
    
    //undistortDepthMaps();

    save();
    //std::cout << "saved \n" ; 

    load();
    std::cout << "loaded \n" ;

    return;
}

void CCalibrationThroughImages::initializeDepthIntrinsics()
{
    // from : http://nicolas.burrus.name/index.php/Research/KinectCalibration#tocLink2
    _eimRelativeRotation <<  9.9984628826577793e-01, 1.2635359098409581e-03, -1.7487233004436643e-02, 
                            -1.4779096108364480e-03, 9.9992385683542895e-01, -1.2251380107679535e-02,
                             1.7470421412464927e-02, 1.2275341476520762e-02,  9.9977202419716948e-01;
    _eivRelativeTranslation << 1.9985242312092553e-02, -7.4423738761617583e-04,-1.0916736334336222e-02;
    _mDepthDistCoeffs = (cv::Mat_<double>(5,1) << -2.6386489753128833e-01, 9.9966832163729757e-01, -7.6275862143610667e-04, 5.0350940090814270e-03, -1.3053628089976321e+00);
    _mDepthK = (cv::Mat_<double>(3,3) << 5.9421434211923247e+02, 0, 3.3930780975300314e+02,
                                         0, 5.9104053696870778e+02, 2.4273913761751615e+02,
                                         0, 0, 1.);
    cv::invert( _mDepthK, _mInvDepthK, DECOMP_SVD );                                    
    return;
}

void CCalibrationThroughImages::loadImages ( const boost::filesystem::path& cFullPath_ )
{
    //check the cFullPathz
    if ( !boost::filesystem::exists ( cFullPath_ ) )
    {
        CError cE; cE << CErrorInfo ( "Not found: " ) << CErrorInfo ( cFullPath_.string() ) << CErrorInfo ( "\n" );
        throw cE;
    }

    if ( !boost::filesystem::is_directory ( cFullPath_ ) )
    {
        CError cE; cE << CErrorInfo ( "Not a directory: " ) << CErrorInfo ( cFullPath_.string() ) << CErrorInfo ( "\n" );
        throw cE;
    }

    _vImages.clear(); 
    _vstrImagePathName.clear();

    string strPathName  = cFullPath_.string();

    for(int i = 0; i < 35; i++ )
    {
        string strNum = boost::lexical_cast<string> ( i );
        std::string strRGBFileName = strPathName + "rgb" + strNum + ".bmp"; //saved into the folder from which the KinectCalibrationDemo is being run.
        std::string strDepFileName = strPathName + "ir" + strNum + ".bmp"; 
        _vImages.push_back( cv::imread( strRGBFileName ) );
        _vDepthMaps.push_back( cv::imread( strDepFileName ) );
        _vstrImagePathName.push_back( strRGBFileName );
        _vstrDepthPathName.push_back( strDepFileName );
        cout << strRGBFileName << " loaded. " << endl << flush;
        cout << strDepFileName << " loaded. " << endl << flush;
    }



/*
    //load all images in a folder
    boost::filesystem::directory_iterator itrEnd;
    for ( boost::filesystem::directory_iterator itrDir ( cFullPath_ );   itrDir != itrEnd;  ++itrDir )
    {
        if ( itrDir->path().extension() == ".jpg" || itrDir->path().extension() == ".bmp" || itrDir->path().extension() == ".png" )
        {
            _vImages.push_back ( cv::imread ( itrDir->path().string(), 1 ) );
            _vstrImagePathName.push_back( itrDir->path().string() );
            //std::cout << itrDir->path().filename() << std::endl;
        }
    }
*/

    _uViews = _vImages.size();

    _vImageResolution << _vImages[0].cols , _vImages[1].rows; // x,y; 
    return;
}

void CCalibrationThroughImages::locate2DCorners()
{
    _vv2DCorners.clear();

    cv::Size boardSize ( _NUM_CORNERS_X, _NUM_CORNERS_Y );

    for ( unsigned int i = 0; i < _vImages.size(); i++ )
    {
        cv::Mat& cvFrame = _vImages[i] ;

        std::vector<cv::Point2f> vCurrentCorners;//float 2d point is required by the OpenCV API.
        /*
        //locate corners roughly
        bool _bChessBoardCornersFoundThisFrame = cv::findChessboardCorners ( cvFrame, boardSize, vCurrentCorners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS );

        if ( !_bChessBoardCornersFoundThisFrame )
        {
            CError cE;
            cE << CErrorInfo ( " No corners are found.\n" );
            throw cE;
        }
        */
        //cv::Size patternSize(4,11);

        bool _bChessBoardCornersFoundThisFrame =
            cv::findCirclesGrid( cvFrame, boardSize, vCurrentCorners, CALIB_CB_CLUSTERING | CALIB_CB_SYMMETRIC_GRID);
        if ( !_bChessBoardCornersFoundThisFrame )
        {
            CError cE;
            string strNum = boost::lexical_cast<string> ( i );
            string strError = strNum + " No circles are found.\n";
            cE << CErrorInfo ( strError.c_str() );
            throw cE;
        }

        //locate corners in sub-pixel level
        cv::Mat cvFrameGrey;
        cv::cvtColor ( cvFrame, cvFrameGrey, CV_BGR2GRAY );
        //cv::cornerSubPix ( cvFrameGrey, vCurrentCorners, cv::Size ( 11, 11 ), cv::Size ( -1, -1 ), cv::TermCriteria ( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1 ) );
        //cv::drawChessboardCorners ( cvFrame, boardSize, vCurrentCorners, _bChessBoardCornersFoundThisFrame );//draw corners

        //store the corners inpto a vector
        _vv2DCorners.push_back ( vCurrentCorners );
    }

    return;
}

void CCalibrationThroughImages::define3DCorners ( const float& fX_, const float& fY_ )
{
    _vv3DCorners.clear();

    std::vector<cv::Point3f> vObjPts;
    for ( int r = 0; r < _NUM_CORNERS_Y; r++ )
        for ( int c = 0; c < _NUM_CORNERS_X; c++ )
        {
            vObjPts.push_back ( cv::Point3f ( r * fY_, c * fX_,  0.0f ) );
        }

    for ( unsigned int i = 0; i < _vv2DCorners.size(); ++i )
    {
        _vv3DCorners.push_back ( vObjPts );
    }

    return;
}

void CCalibrationThroughImages::calibrate ()
{
    _mK = cv::Mat_<double> ( 3, 3 );
    _mDistCoeffs = cv::Mat_<double> ( 5, 1 );

    _mK.at<double> ( 0, 0 ) = 1;
    _mK.at<double> ( 1, 1 ) = 1;

    if ( _vImages.empty() )
    {
        CError cE;
        cE << CErrorInfo ( " No images have been loaded. \n" );
        throw cE;
    }

    cv::Size cvFrameSize ( _vImageResolution[0], _vImageResolution[1] );

    //calibrate the camera
    double dBackProjectError = cv::calibrateCamera ( _vv3DCorners, _vv2DCorners, cvFrameSize, _mK, _mDistCoeffs, _vmRotationVectors, _vmTranslationVectors );
    cv::invert( _mK, _mInvK, DECOMP_SVD );

    //output calibration results
    std::cout << "Camera calibrated." << std::endl;
    PRINT( _mK );
    PRINT( _mDistCoeffs );
    PRINT( dBackProjectError );
    PRINT( _mK.type() );
    PRINT( _mDistCoeffs.type() );
    PRINT( _vmRotationVectors[0].type() );
    PRINT( _vmTranslationVectors[0].type() ); 
}

/**
* @brief convert rotation vectors into rotation matrices using cv::
*
* @param
* @param argv
*/
void CCalibrationThroughImages::convertRV2RM()
{
    _vmRotationMatrices.clear();

    for ( unsigned int n = 0; n < _vmRotationVectors.size() ; n++ )
    {
        cv::Mat cvmR;
        cv::Rodrigues ( _vmRotationVectors[n], cvmR );
        _vmRotationMatrices.push_back ( cvmR );
    }

    return;
}

void CCalibrationThroughImages::undistortImages()
{
    cv::Size cvFrameSize ( _vImageResolution[0], _vImageResolution[1] ); //x,y;
    _vUndistortedImages.clear();
    for ( unsigned int n = 0; n < _uViews; n++ )
    {
        cv::Mat_<float> _mapX, _mapY;
        _mapX = cv::Mat_<float> ( cvFrameSize );
        _mapY = cv::Mat_<float> ( cvFrameSize );

        for ( int y = 0; y < _vImageResolution ( 1 ); ++y )
        {
            for ( int x = 0; x < _vImageResolution ( 0 ); ++x )
            {
                //btl::ImageRGB output ( _frameSize ( 0 ), _frameSize ( 1 ) );
                Eigen::Vector2d undistorted ( x, y );
                Eigen::Vector2d distorted = distortPoint ( undistorted );
                _mapX[y][x] = ( float ) distorted ( 0 );
                _mapY[y][x] = ( float ) distorted ( 1 );
            }
        }
        cv::Mat cvUndistorted;
        cv::remap ( _vImages[n], cvUndistorted, _mapX, _mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT );
        _vUndistortedImages.push_back( cvUndistorted );
        std::string strNum = boost::lexical_cast<string> ( n );
        std::string strRGBUndistortedFileName = "rgbUndistorted" + strNum + ".bmp";
        cv::imwrite ( strRGBUndistortedFileName, cvUndistorted ); 
    }

    return;
}
void CCalibrationThroughImages::undistortDepthMaps()
{
    cv::Size cvFrameSize ( _vImageResolution[0], _vImageResolution[1] ); //x,y;
    _vUndistortedDepthMaps.clear();
    for ( unsigned int n = 0; n < _uViews; n++ )
    {
        cv::Mat_<float> _mapX, _mapY;
        _mapX = cv::Mat_<float> ( cvFrameSize );
        _mapY = cv::Mat_<float> ( cvFrameSize );

        for ( int y = 0; y < _vImageResolution ( 1 ); ++y )
        {
            for ( int x = 0; x < _vImageResolution ( 0 ); ++x )
            {
                //btl::ImageRGB output ( _frameSize ( 0 ), _frameSize ( 1 ) );
                Eigen::Vector2d undistorted ( x, y );
                Eigen::Vector2d distorted = distortDepthPoint ( undistorted );
                _mapX[y][x] = ( float ) distorted ( 0 );
                _mapY[y][x] = ( float ) distorted ( 1 );
            }
        }
        cv::Mat cvUndistorted;
        cv::remap ( _vDepthMaps[n], cvUndistorted, _mapX, _mapY, cv::INTER_LINEAR, cv::BORDER_CONSTANT );
        _vUndistortedDepthMaps.push_back( cvUndistorted );
        std::string strNum = boost::lexical_cast<string> ( n );
        std::string strDepthUndistortedFileName = "depthUndistorted" + strNum + ".bmp";
        cv::imwrite ( strDepthUndistortedFileName, cvUndistorted ); 
    }

    return;
}

void CCalibrationThroughImages::save()
{
    using namespace shuda;

    // create and open a character archive for output
    std::ofstream ofs ( "CalibrationThroughImages.xml" );
    boost::archive::xml_oarchive oa ( ofs );

    //convert non-standard variables into vectors
    std::vector< std::vector< int > >    stdvImageResolution;     
    std::vector< std::vector< double > > stdvKMatrix;         
    std::vector< std::vector< double > > stdvDistortionCoeff;     
    std::vector< std::vector< std::vector< double > > > stdvRotationVectors;     
    std::vector< std::vector< std::vector< double > > > stdvRotationMatrices;
    std::vector< std::vector< std::vector< double > > > stdvTranslationVectors;
    std::vector< std::vector< std::vector< float  > > > stdv2DCorners;
    std::vector< std::vector< std::vector< float  > > > stdv3DCorners;


    stdvImageResolution   << _vImageResolution;
    stdvKMatrix           << _mK;
    stdvDistortionCoeff   << _mDistCoeffs;
    stdvRotationVectors   << _vmRotationVectors;
    stdvRotationMatrices  << _vmRotationMatrices;
    stdvTranslationVectors<< _vmTranslationVectors;
    stdv2DCorners         << _vv2DCorners;
    stdv3DCorners         << _vv3DCorners;
    
    oa << BOOST_SERIALIZATION_NVP ( _NUM_CORNERS_X );
    oa << BOOST_SERIALIZATION_NVP ( _NUM_CORNERS_Y );
    oa << BOOST_SERIALIZATION_NVP ( _X );
    oa << BOOST_SERIALIZATION_NVP ( _Y );
    oa << BOOST_SERIALIZATION_NVP ( _vstrImagePathName );
    oa << BOOST_SERIALIZATION_NVP ( _uViews );
    oa << BOOST_SERIALIZATION_NVP ( stdvImageResolution );
    oa << BOOST_SERIALIZATION_NVP ( stdvKMatrix );
    oa << BOOST_SERIALIZATION_NVP ( stdvDistortionCoeff );
    oa << BOOST_SERIALIZATION_NVP ( stdvRotationVectors );
    oa << BOOST_SERIALIZATION_NVP ( stdvRotationMatrices );
    oa << BOOST_SERIALIZATION_NVP ( stdvTranslationVectors );
    oa << BOOST_SERIALIZATION_NVP ( stdv2DCorners );
    oa << BOOST_SERIALIZATION_NVP ( stdv3DCorners );

    return;
}

void CCalibrationThroughImages::load()
{
    using namespace shuda;

    // create and open a character archive for input
    std::ifstream ifs ( "CalibrationThroughImages.xml" );
    boost::archive::xml_iarchive ia ( ifs );

    std::vector< std::vector< int > > stdvImageResolution;
    std::vector< std::vector< double > > stdvKMatrix;         
    std::vector< std::vector< double > > stdvDistortionCoeff; 
    std::vector< std::vector< std::vector< double > > > stdvRotationVectors;
    std::vector< std::vector< std::vector< double > > > stdvRotationMatrices;
    std::vector< std::vector< std::vector< double > > > stdvTranslationVectors;
    std::vector< std::vector< std::vector< float > > > stdv2DCorners;
    std::vector< std::vector< std::vector< float > > > stdv3DCorners;

    ia >> BOOST_SERIALIZATION_NVP ( _NUM_CORNERS_X );
    ia >> BOOST_SERIALIZATION_NVP ( _NUM_CORNERS_Y );
    ia >> BOOST_SERIALIZATION_NVP ( _X );
    ia >> BOOST_SERIALIZATION_NVP ( _Y );
    ia >> BOOST_SERIALIZATION_NVP ( _vstrImagePathName );
    ia >> BOOST_SERIALIZATION_NVP ( _uViews );
    ia >> BOOST_SERIALIZATION_NVP ( stdvImageResolution );
    ia >> BOOST_SERIALIZATION_NVP ( stdvKMatrix );
    ia >> BOOST_SERIALIZATION_NVP ( stdvDistortionCoeff );
    ia >> BOOST_SERIALIZATION_NVP ( stdvRotationVectors );
    ia >> BOOST_SERIALIZATION_NVP ( stdvRotationMatrices );
    ia >> BOOST_SERIALIZATION_NVP ( stdvTranslationVectors );
    ia >> BOOST_SERIALIZATION_NVP ( stdv2DCorners );
    ia >> BOOST_SERIALIZATION_NVP ( stdv3DCorners );

    //convert vectors into non-standard variables
    stdvImageResolution   >> _vImageResolution;
    stdvKMatrix           >> _mK;
    stdvDistortionCoeff   >> _mDistCoeffs;
    stdvRotationVectors   >> _vmRotationVectors;
    stdvRotationMatrices  >> _vmRotationMatrices;
    stdvTranslationVectors>> _vmTranslationVectors;
    stdv2DCorners         >> _vv2DCorners;
    stdv3DCorners         >> _vv3DCorners;

    return;
}
/*
btl::camera::CameraModelPinhole CCalibrationThroughImages::btlCameraModelPinHoleK() const
{
    if ( _vImages.empty() )
    {
        CError cE;
        cE << CErrorInfo ( " No images have been loaded. \n" );
        throw cE;
    }

    return btl::camera::CameraModelPinhole(_vImageResolution[0],_vImageResolution[1],_mK.at<double>(0,0),_mK.at<double>(1,1),_mK.at<double>(0,2),_mK.at<double>(1,2) );
}*/

}//shuda
