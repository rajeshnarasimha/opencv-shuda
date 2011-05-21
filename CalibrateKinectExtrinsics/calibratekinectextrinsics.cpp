#include "calibratekinectextrinsics.hpp"
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/lexical_cast.hpp>
#include "optimdepth.hpp"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <map>
#include "boost/date_time/posix_time/posix_time.hpp"

using namespace boost::posix_time;
using namespace boost::gregorian;

namespace shuda
{

void CCalibrateKinectExtrinsics::convertDepth ()
{
	for(unsigned int i = 0; i < _vstrImagePathName.size(); i++ )
    { 
		cv::Mat& Depth =_vUndistortedDepthMaps[i];
		cv::Mat_<unsigned short> DepthInt( _vImageResolution(1), _vImageResolution(0) );
		cv::Mat_<unsigned short> FilteredDepthInt;
		PRINT( _dDepthThreshold );
		for ( int y = 0; y < Depth.rows; y++ )
        for ( int x = 0; x < Depth.cols; x++ )
        {
			DepthInt.at<unsigned short> ( y, x ) = rawDepth< unsigned short >( x,   y  , Depth );
        }
		//btl::utility::filterDepth < unsigned short > ( (unsigned short) _dDepthThreshold, DepthInt, &FilteredDepthInt );
		_vFilteredUndistDepthInts.push_back( DepthInt );
    }
	return;
}

void CCalibrateKinectExtrinsics::mainFunc ( const boost::filesystem::path& cFullPath_ )
{
	_strFullPath = cFullPath_.file_string();

	parseControlYAML();

 	//load depth camera intrinsics
    importKinectIntrinsics();
	std::cout << "kinect intrinsics imported. \n";

	//load images
	if( 1 == _nLoadRGB )
	{
	    loadImages ( cFullPath_, _vstrImagePathName, &_vImages );
		_vImageResolution(0) = _vImages[0].cols;
		_vImageResolution(1) = _vImages[0].rows;
		std::cout << "rgb images loaded.\n";
	}
	if( 1 == _nLoadDepth )
	{
		//PRINT( _vstrDepthPathName );
		loadImages ( cFullPath_, _vstrDepthPathName, &_vDepthMaps );
		std::cout << "depth images loaded.\n";
	}
	if( 1 == _nLoadUndistortedRGB )
	{
		loadImages ( cFullPath_, _vstrUndistortedImagePathName, &_vUndistortedImages );
		std::cout << "undistorted rgb images loaded.\n";
	}
	if( 1 == _nLoadUndistortedDepth )
	{
		loadImages ( cFullPath_, _vstrUndistortedDepthPathName, &_vUndistortedDepthMaps );
		std::cout << "undistorted depth images loaded.\n";
	}
	if( 1 == _nUndistortImages )
	{
    	undistortImages( _vImages,   _mRGBK, _mRGBInvK, _mRGBDistCoeffs,  &_vUndistortedImages );
	    undistortImages( _vDepthMaps,_mIRK,  _mIRInvK,  _mIRDistCoeffs,   &_vUndistortedDepthMaps );
		std::cout << "image undistorted.\n";
	}
	convertDepth ();

	if( 1 == _nExportUndistortedRGB )
	{
		exportImages( cFullPath_, _vstrUndistortedImagePathName,  _vUndistortedImages );
		std::cout << " Undistorted rgb image exported. \n";
	}
	if( 1 == _nExportUndistortedDepth )
	{
		exportImages( cFullPath_, _vstrUndistortedDepthPathName,  _vUndistortedDepthMaps );
		std::cout << " Undistorted depth image exported. \n";
	}

    //find corners
	if( 1 == _nCalibrateExtrinsics )
	{
	    locate2DCorners(_vImages, _NUM_CORNERS_X, _NUM_CORNERS_Y, &_vv2DCorners );
    	std::cout << "corners located \n" ;
    	//define 3D corners
    	define3DCorners ( _X, _Y, _NUM_CORNERS_X, _NUM_CORNERS_Y, views(), &_vv3DCorners );
    	std::cout << "3d corners defined \n" ;
	    //calibration
    	calibrateExtrinsics();
    	std::cout << "camera calibrateExtrinsicsd \n" ;
    	//convert rotation vectors to rotation matrices
    	convertRV2RM( _vmRotationVectors, &_vmRotationMatrices );
    	std::cout << "convertRV2RM() executed \n" ;
    }

	if( 1 == _nSerializeToXML )
	{
		save();
    	cout << "serialized to XML \n" ; 
	}

	if( 1 == _nSerializeFromXML )
	{
    	load();
    	cout << "serialized from XML \n" ;
	}
	
	calcAllProjMatrices( RGB_CAMERA,   &_veimRGBProjs );
	calcAllProjMatrices( IR_CAMERA, &_veimDepthProjs );

/*
	for( unsigned int i = 0; i< views(); i++ )
	{
		Mat_<int> filtered;
		filterDepth( _dDepthThreshold, _vUndistortedDepthMaps[i], &filtered);
		_vFilteredDepth.push_back( filtered ); 
	}
*/

	//collecting 3D Points
	collect3DPtAll( &_vveiv3DPts, &_vvp3DColors);

	if( 1 == _nCalibrateDepth )	
	{
		calibDepth(); 
		cout << "depth calibrated";
	}

    return;
}
void CCalibrateKinectExtrinsics::parseControlYAML()
{
 	ifstream iFile ( "control.yaml" );
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
		if( "load rgb images" == (*cIt1).first )
			_nLoadRGB =(*cIt1).second;
		else if( "load depth maps" == (*cIt1).first )
			_nLoadDepth =(*cIt1).second;
		else if( "load undistorted rgb images" == (*cIt1).first )
			_nLoadUndistortedRGB =(*cIt1).second;
		else if( "load undistorted depth maps" == (*cIt1).first )
			_nLoadUndistortedDepth =(*cIt1).second;
		else if("undistort images" == (*cIt1).first )
			_nUndistortImages =(*cIt1).second;
		else if("export undistorted rgb images" == (*cIt1).first )
			_nExportUndistortedRGB =(*cIt1).second;
		else if("export undistorted depth images" == (*cIt1).first )
			_nExportUndistortedDepth =(*cIt1).second;
		else if("calibrate extrinsics" == (*cIt1).first )
			_nCalibrateExtrinsics =(*cIt1).second;
		else if("calibrate depth" == (*cIt1).first )	
			_nCalibrateDepth =(*cIt1).second;
		else if("export to xml" == (*cIt1).first )
			_nSerializeToXML =(*cIt1).second;
		else if("import from xml" == (*cIt1).first )
			_nSerializeFromXML =(*cIt1).second;
	}

	// properties
	std::map < std::string, double > mpProperties1;
 	parser.GetNextDocument ( doc );
	doc[0] >> mpProperties1;
	_dDepthThreshold = mpProperties1["depth filter threshold"];

	// properties
	std::map < std::string, vector< int > > mpCornerCounts;
 	parser.GetNextDocument ( doc );
	doc[0] >> mpCornerCounts;
	//PRINT( mpCornerCounts );
	//process mpCornerCounts
	vector< int > vCornerCounts;
	vCornerCounts = mpCornerCounts["No. of Corners X Y"];
	_NUM_CORNERS_X = vCornerCounts[0]; _NUM_CORNERS_Y = vCornerCounts[1]; 

	std::map < string, vector< float > > mpUnitLength;
 	parser.GetNextDocument ( doc );
	doc[0] >> mpUnitLength;
	//PRINT( mpUnitLength );
	//process mpUnitLength
	vector< float > vUnitLength;
	vUnitLength = mpUnitLength["Unit length in X Y"];
	_X = vUnitLength[0]; _Y = vUnitLength[1]; 

	std::map <string, string> mpProperties;
 	parser.GetNextDocument ( doc );
	doc[0] >> mpProperties;
	string _strImageDirectory = mpProperties["image directory"];
	//PRINT( mpProperties );


	vector<string> vRGBNames;
 	parser.GetNextDocument ( doc );
	for ( unsigned i = 0; i < doc.size(); i++ )
    {
		doc[i] >> vRGBNames;
	}
	//PRINT( vRGBNames );

	vector<string> vDepthNames;
 	parser.GetNextDocument ( doc );
	for ( unsigned i = 0; i < doc.size(); i++ )
    {
		doc[i] >> vDepthNames;
	}
	//PRINT( vDepthNames );

	vector<string> vUndistRGB;
 	parser.GetNextDocument ( doc );
	for ( unsigned i = 0; i < doc.size(); i++ )
    {
		doc[i] >> vUndistRGB;
	}
	//PRINT( vUndistRGB );

	vector<string> vUndistDepth;
 	parser.GetNextDocument ( doc );
	for ( unsigned i = 0; i < doc.size(); i++ )
    {
		doc[i] >> vUndistDepth;
	}
	//PRINT( vUndistDepth );
	CHECK( vRGBNames.size() == vDepthNames.size(), "There must be the same # of rgb and depth, undistorted rgb and depth.");
	CHECK( vRGBNames.size() == vUndistRGB.size(), "There must be the same # of rgb and depth, undistorted rgb and depth.");
	CHECK( vRGBNames.size() == vUndistDepth.size(), "There must be the same # of rgb and depth, undistorted rgb and depth.");

    _uViews = vRGBNames.size();

    for ( unsigned int i = 0; i< vRGBNames.size(); i++ )
	{
		_vstrImagePathName.push_back( _strImageDirectory+vRGBNames[i] ); 
    	_vstrDepthPathName.push_back( _strImageDirectory+vDepthNames[i] ); 
    	_vstrUndistortedImagePathName.push_back( _strImageDirectory+vUndistRGB[i] );     	
		_vstrUndistortedDepthPathName.push_back( _strImageDirectory+vUndistDepth[i] );  
	}
	return;
}
/*
void CCalibrateKinectExtrinsics::loadImages ( const boost::filesystem::path& cFullPath_, const std::vector< std::string >& vImgNames_, std::vector< cv::Mat >* pvImgs_ ) const
{
    pvImgs_->clear(); 

    string strPathName  = cFullPath_.string();

    for(unsigned int i = 0; i < vImgNames_.size(); i++ )
    { 
        std::string strRGBFileName = strPathName + vImgNames_[i]; //saved into the folder from which the KinectCalibrationDemo is being run.
		PRINT( strRGBFileName );
		CHECK( boost::filesystem::exists ( strRGBFileName ), "Not found: " + strRGBFileName + "\n" );
 		Mat img = cv::imread( strRGBFileName );
		cvtColor(img, img, CV_BGR2RGB );
        pvImgs_->push_back( img );
    }

    return;
}

void CCalibrateKinectExtrinsics::exportImages( const boost::filesystem::path& cFullPath_, const vector< std::string >& vImgNames_, const std::vector< cv::Mat >& vImgs_ ) const
{
    string strPathName  = cFullPath_.string();

	for(unsigned int n = 0; n < vImgNames_.size(); n++ )
    {
		Mat img = vImgs_[n];
		cvtColor(img, img, CV_RGB2BGR );
        cv::imwrite ( strPathName + vImgNames_[n], vImgs_[n] ); 
    }
	return;
}
*/

void CCalibrateKinectExtrinsics::calibrateExtrinsics ()
{
	CHECK(!_vImages.empty(),      " No images have been loaded. \n" );
    CHECK(!_vv3DCorners.empty(),  " No 3D corners have been defined. \n" );
	CHECK(!_vv2DCorners.empty(),  " No 2D corners have been detected. \n" );

	_vmRotationVectors.clear();
	_vmTranslationVectors.clear();

    cv::Size cvFrameSize ( _vImageResolution[0], _vImageResolution[1] );

	CvMat* p3DCorner = cvCreateMat( 1, _NUM_CORNERS_Y*_NUM_CORNERS_X, CV_32FC3);
	CvMat* p2DCorner = cvCreateMat( 1, _NUM_CORNERS_Y*_NUM_CORNERS_X, CV_32FC2);

    Mat_<double> K = cvMatRGBK(); Mat_<double> D = cvMatRGBDistort();

    CvMat* pK = cvCreateMat(3,3,  cvMatRGBK().type() ); 
	CvMat* pD = cvCreateMat(cvMatRGBDistort().rows, cvMatRGBDistort().cols, cvMatRGBDistort().type() ); 
    *pK << K;
	*pD << D;
	CvMat* pR = cvCreateMat(3,1, CV_64F);
	CvMat* pT = cvCreateMat(3,1, CV_64F);

	for ( int n = 0; n < views(); n++ )
	{
		float* p3D = (float*) p3DCorner->data.ptr;
		float* p2D = (float*) p2DCorner->data.ptr;

	    for ( int c = 0; c < _NUM_CORNERS_X * _NUM_CORNERS_Y; c++ )
    	{
    		*p3D =  _vv3DCorners[n][c].x; p3D++;
			*p3D =  _vv3DCorners[n][c].y; p3D++;
			*p3D =  _vv3DCorners[n][c].z; p3D++;
			*p2D =  _vv2DCorners[n][c].x; p2D++;
			*p2D =  _vv2DCorners[n][c].y; p2D++;
   		}

		
	    //calibrateExtrinsics the camera
		cvFindExtrinsicCameraParams2(p3DCorner, p2DCorner, pK, pD, pR, pT);


		Mat_<double> R; R << *pR;
		Mat_<double> T; T << *pT;
        
		_vmRotationVectors.push_back( R );
		_vmTranslationVectors.push_back( T.t() );
	}

    //output calibration results
    //PRINT( _vmRotationVectors );
	//PRINT( _vmTranslationVectors );

	return;
}
/*
void CCalibrateKinectExtrinsics::loadControlScript()
{
	using namespace shuda;

    // create and open a character archive for input
    std::ifstream ifs ( "CalibrationThroughImages.xml" );
    boost::archive::xml_iarchive ia ( ifs );

}
*/
void CCalibrateKinectExtrinsics::save()
{
    using namespace shuda;

    // create and open a character archive for output
    std::ofstream ofs ( "CalibrationThroughImages.xml" );
    boost::archive::xml_oarchive oa ( ofs );

    //convert non-standard variables into vectors
    std::vector< std::vector< int > >    stdvImageResolution;     
    std::vector< std::vector< std::vector< double > > > stdvRotationVectors;     
    std::vector< std::vector< std::vector< double > > > stdvRotationMatrices;
    std::vector< std::vector< std::vector< double > > > stdvTranslationVectors;
    std::vector< std::vector< std::vector< float  > > > stdv2DCorners;
    std::vector< std::vector< std::vector< float  > > > stdv3DCorners;


    stdvImageResolution   << _vImageResolution;
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
	oa << BOOST_SERIALIZATION_NVP ( _vstrDepthPathName );
    oa << BOOST_SERIALIZATION_NVP ( _uViews );
    oa << BOOST_SERIALIZATION_NVP ( stdvImageResolution );
    oa << BOOST_SERIALIZATION_NVP ( stdvRotationVectors );
    oa << BOOST_SERIALIZATION_NVP ( stdvRotationMatrices );
    oa << BOOST_SERIALIZATION_NVP ( stdvTranslationVectors );
    oa << BOOST_SERIALIZATION_NVP ( stdv2DCorners );
    oa << BOOST_SERIALIZATION_NVP ( stdv3DCorners );

    return;
}

void CCalibrateKinectExtrinsics::load()
{
    using namespace shuda;

    // create and open a character archive for input
    std::ifstream ifs ( "CalibrationThroughImages.xml" );
    boost::archive::xml_iarchive ia ( ifs );

    std::vector< std::vector< int > > stdvImageResolution;
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
    ia >> BOOST_SERIALIZATION_NVP ( _vstrDepthPathName );
    ia >> BOOST_SERIALIZATION_NVP ( _uViews );
    ia >> BOOST_SERIALIZATION_NVP ( stdvImageResolution );
    ia >> BOOST_SERIALIZATION_NVP ( stdvRotationVectors );
    ia >> BOOST_SERIALIZATION_NVP ( stdvRotationMatrices );
    ia >> BOOST_SERIALIZATION_NVP ( stdvTranslationVectors );
    ia >> BOOST_SERIALIZATION_NVP ( stdv2DCorners );
    ia >> BOOST_SERIALIZATION_NVP ( stdv3DCorners );

    //convert vectors into non-standard variables
    stdvImageResolution   >> _vImageResolution;
    stdvRotationVectors   >> _vmRotationVectors;
    stdvRotationMatrices  >> _vmRotationMatrices;
    stdvTranslationVectors>> _vmTranslationVectors;
    stdv2DCorners         >> _vv2DCorners;
    stdv3DCorners         >> _vv3DCorners;

    return;
}

Vector3d CCalibrateKinectExtrinsics::eiVecT(unsigned int uNthView_, int nCameraType_ ) const
{
	Vector3d eivT;

	if( IR_CAMERA == nCameraType_ )
	{
    	eivT = eiMatRelativeRotation() *  eiVecT(uNthView_) + eiVecRelativeTranslation();
    }
	else if( RGB_CAMERA == nCameraType_ )
	{
		eivT = eiVecT( uNthView_ );
	}
	else
	{
		THROW( "eiVecT(): unrecognized camera type." );
	}
	
	return eivT;
}

Matrix3d CCalibrateKinectExtrinsics::eiMatR(unsigned int uNthView_, int nCameraType_ ) const
{
	Matrix3d eimR;

	if( IR_CAMERA == nCameraType_ )
	{
    	eimR = eiMatRelativeRotation() * eiMatR(uNthView_);
    }
	else if( RGB_CAMERA == nCameraType_ )
	{
		eimR = eiMatR(uNthView_);
	}
	else
	{
		THROW( "eiMatR(): unrecognized camera type." );
	}
	
	return eimR;
}



Matrix< double , 3, 4 > CCalibrateKinectExtrinsics::calcProjMatrix( unsigned int uNthView_, int nCameraType_ ) const
{
	Matrix3d eimR = eiMatR(uNthView_, nCameraType_ );
	Vector3d eivT = eiVecT(uNthView_, nCameraType_ );

    Matrix< double , 3, 4 > mPrj;
	mPrj( 0,0 ) = eimR( 0, 0 ); mPrj( 0, 1 ) = eimR( 0, 1 ); mPrj( 0, 2 ) = eimR( 0, 2 ); mPrj( 0, 3 ) = eivT( 0, 0 );
	mPrj( 1,0 ) = eimR( 1, 0 ); mPrj( 1, 1 ) = eimR( 1, 1 ); mPrj( 1, 2 ) = eimR( 1, 2 ); mPrj( 1, 3 ) = eivT( 1, 0 );
	mPrj( 2,0 ) = eimR( 2, 0 ); mPrj( 2, 1 ) = eimR( 2, 1 ); mPrj( 2, 2 ) = eimR( 2, 2 ); mPrj( 2, 3 ) = eivT( 2, 0 );

	mPrj = eiMatK( nCameraType_ ) * mPrj;

	return mPrj;
}

void CCalibrateKinectExtrinsics::calcAllProjMatrices(int nCameraType_, std::vector< Matrix< double , 3, 4 > >* pveimProjs_ ) const
{
	pveimProjs_->clear();
	for(unsigned int n = 0; n < views(); n++ )
	{
		pveimProjs_->push_back( calcProjMatrix( n, nCameraType_ ) );
	}
	return ;
}

/*
void CCalibrateKinectExtrinsics::calibPhysicalFocalLength()
{

}
*/

void CCalibrateKinectExtrinsics::calibDepth()
{
	vector<Vector4d> veivDifference;

	Matrix3d mK = eiMatK( IR_CAMERA );
	const double u = mK(0,2);
    const double v = mK(1,2);
    const double f = ( mK(0,0) + mK(1,1) )/2.;

	for( unsigned int uNthView = 0; uNthView < views(); uNthView ++ )
	{
		Matrix<double, 3, 4 > mPrj = _veimDepthProjs[ uNthView ];
	
		Matrix3d mR = eiMatR( uNthView, IR_CAMERA );
		Vector3d vT = eiVecT( uNthView, IR_CAMERA );
		Matrix4d eimGLM = setOpenGLModelViewMatrix( mR, vT );
	
	    const Mat& img = undistortedDepth ( uNthView );
	
		Vector4d vOne(0,0,0,0);
		// get samples of 3D corners 
		unsigned int uCounter = 0;
		for(int i = 0; i <_NUM_CORNERS_X * _NUM_CORNERS_Y ; i++)
		{
			Vector4d eiv3DCheckboardCornersPtH( _vv3DCorners[ uNthView ][i].x,  _vv3DCorners[ uNthView ][i].y,  _vv3DCorners[ uNthView ][i].z, 1.0 );
			Vector3d eiv2DPtH = mPrj * eiv3DCheckboardCornersPtH;
			eiv2DPtH /= eiv2DPtH(2);
			Vector4d eivDepthPtH;
			if ( getPtCameraCoordinate<double>(  int( eiv2DPtH(0) + 0.5 ), int( eiv2DPtH(1) + 0.5 ), f, u, v, img, &eivDepthPtH ) )
			{
				// transform from world coordinate to camera coordinate
				eiv3DCheckboardCornersPtH  = eimGLM * eiv3DCheckboardCornersPtH;
				eiv3DCheckboardCornersPtH /= eiv3DCheckboardCornersPtH(3);
				// compute the difference and accumulate it
				eivDepthPtH -= eiv3DCheckboardCornersPtH;
				vOne += eivDepthPtH;
				uCounter++;
			}
		}
		veivDifference.push_back( vOne/uCounter );
	}
	Vector4d vOne(0,0,0,0);
	for(vector< Vector4d >::const_iterator const_it = veivDifference.begin(); const_it != veivDifference.end(); ++ const_it)
	{
		vOne += *const_it;
	}
	vOne /= veivDifference.size();
	PRINT( vOne );
//	COptimDepth* cOD = new COptimDepth;
//	cOD->set( vRealDepth, vRawDepth );
//	cOD->SetMdAlg(shuda::COptim::GRADIENTDESCENDENT);
//	cOD->Go();
//	_mDK =  cOD->GetX().clone();
//	PRINT( _mDK );

	return;
}


void CCalibrateKinectExtrinsics::calibDepthFreeNect()
{
	const unsigned int uNthView = 1;

	Matrix<double, 3, 4 > mPrj = _veimDepthProjs[ uNthView ];

	Matrix3d mR = eiMatR( uNthView, IR_CAMERA );
	Vector3d vT = eiVecT( uNthView, IR_CAMERA );

	// get camera center
    Vector3d eivCamera = - mR.transpose() * vT;
    const Mat& img = undistortedDepth ( uNthView );

	vector<double> vRealDepth;
	vector<int> vRawDepth;
	// get samples of 3D corners 
	for(int i = 0; i <_NUM_CORNERS_X * _NUM_CORNERS_Y ; i++)
	{
		Vector4d eiv3DPtH( _vv3DCorners[ uNthView ][i].x,  _vv3DCorners[ uNthView ][i].y,  _vv3DCorners[ uNthView ][i].z, 1.0 );
		Vector3d eiv3DPt( _vv3DCorners[ uNthView ][i].x,  _vv3DCorners[ uNthView ][i].y,  _vv3DCorners[ uNthView ][i].z );
		Vector3d eiv2DPtH = mPrj * eiv3DPtH;
		eiv2DPtH /= eiv2DPtH(2);
		int nRawDepth = rawDepth <int> ( int( eiv2DPtH(0) + 0.5 ), int( eiv2DPtH(1) + 0.5 ), img );
		double dD = depthInMeters<double> ( eiv2DPtH(0), eiv2DPtH(1), img );
		eiv3DPt -= eivCamera;
		double dDistReal = sqrt(eiv3DPt.dot(eiv3DPt));

		vRealDepth.push_back( dDistReal );
		vRawDepth. push_back( nRawDepth );
	}

	COptimDepth* cOD = new COptimDepth;
	cOD->set( vRealDepth, vRawDepth );
	cOD->SetMdAlg(shuda::COptim::GRADIENTDESCENDENT);

	cOD->Go();

	_mDK =  cOD->GetX().clone();

	PRINT( _mDK );

	return;
}
void CCalibrateKinectExtrinsics::collect3DPtAll(vector< vector< Vector4d > >* pvveiv3DPts_, vector< vector< unsigned char* > >* pvvp3DColors_) const
{
	pvveiv3DPts_->clear();
	pvvp3DColors_->clear();
	for (unsigned int i =0; i< views(); i++ )
	{
		vector< Vector4d > 		 veiv3DPts;
		vector< unsigned char* > vp3DColors;
		collect3DPt(i, &veiv3DPts, &vp3DColors);
		pvveiv3DPts_->push_back(veiv3DPts);
		pvvp3DColors_->push_back(vp3DColors);
		PRINT( i );
		PRINT( veiv3DPts.size() );
		PRINT( vp3DColors.size() );
	}
}

void CCalibrateKinectExtrinsics::collect3DPt(unsigned int uNthView_, vector< Vector4d >* pveiv3DPts_, vector< unsigned char* >* pvp3DColors_) const
{
 	Eigen::Matrix3d mK = eiMatK( IR_CAMERA );

    const double u = mK(0,2);
    const double v = mK(1,2);
    const double f = ( mK(0,0) + mK(1,1) )/2.;

	pveiv3DPts_->clear();
	pveiv3DPts_->reserve( 300000 );
	pvp3DColors_->clear();
	pvp3DColors_->reserve( 300000 );

	//3D Pt in camera coordinate system.
	//const Mat& img = undistortedDepth ( uNthView_ );
	//const Mat_<int> img = _vFilteredDepth[uNthView_];
	const Mat_<unsigned short>& img = _vFilteredUndistDepthInts[uNthView_];
	
	const Mat& color = undistortedImg ( uNthView_ );
	//PRINT( img.size() );
    boost::posix_time::ptime cT0 ( microsec_clock::local_time() );
	//collect the Pt w.r.t. [I|0]
	Vector4d eivPt;
	for ( int y = 0; y < img.rows; y++ )
        for ( int x = 0; x < img.cols; x++ )
        {
			if ( getPtCameraCoordinate2<double, unsigned short>( x, y, f, u, v, img, &eivPt ) )
			{
				pveiv3DPts_->push_back( eivPt );
			}
        }

	boost::posix_time::ptime cT1 ( microsec_clock::local_time() );
    time_duration cTDAll = cT1 - cT0 ;
	PRINT( cTDAll );
	
	cT0 = microsec_clock::local_time() ;
	Eigen::Vector3d vT = eiVecRelativeTranslation();
    Eigen::Matrix3d mR = eiMatRelativeRotation();
	mR.transposeInPlace();

	//Eigen::Matrix4d eimGLM = setOpenGLModelViewMatrix( mR, vT );
    //eimGLM = eimGLM.inverse().eval();

	Eigen::Matrix3d mRGBK = eiMatRGBK(); 
	vector< Vector4d >::iterator it = pveiv3DPts_->begin();
	Vector3d eiv2DPt ;
	for ( ; it != pveiv3DPts_->end(); ++it)
	{
		//eiv3DPt = eimGLM * (*it);
		//eiv2DPt = mRGBK * eiv3DPt.head<3>();
		eiv2DPt = mRGBK * mR * ((*it).head<3>() - vT);
		eiv2DPt /= eiv2DPt(2);
		unsigned char* pColor = getColorPtr<unsigned char>( int(eiv2DPt(0)+0.5), int(eiv2DPt(1)+0.5), color );
		pvp3DColors_->push_back( pColor );
	}
	cT1 = microsec_clock::local_time();
	cTDAll = cT1 - cT0 ;
	PRINT( cTDAll );
	
	/*
	//convert the Pts to world coordinate system
    Eigen::Vector3d vT = eiVecT( uNthView_ ,IR_CAMERA );
    Eigen::Matrix3d mR = eiMatR( uNthView_ ,IR_CAMERA );
    
	//cout << "placeCameraInWorldCoordinate() after setup the RT\n";
    Eigen::Matrix4d eimGLM = setOpenGLModelViewMatrix( mR, vT );
    eimGLM = eimGLM.inverse().eval();
	cT0 = microsec_clock::local_time() ;

	const Matrix< double , 3, 4 >& eimP = prjMatrix(uNthView_, RGB_CAMERA);
	vector< Vector4d >::iterator it = pveiv3DPts_->begin();
	Vector3d eiv2DPt;
	Vector4d eiv3DPt; eiv3DPt(3) = 1.0;
	for ( ; it != pveiv3DPts_->end(); ++it)
	{
		*it = eimGLM * (*it);
		//(*it) /= (*it)(3);
		eiv2DPt = eimP * (*it);
		eiv2DPt /= eiv2DPt(2);

		//This approach is problematic as 
		//eiv3DPt.head<3>() = mR.transpose() * ((*it).head<3>() - vT ) ;
		//eiv2DPt = eimP * eiv3DPt;
		//eiv2DPt /= eiv2DPt(2);
		//eiv2DPt(0) = 640 - eiv2DPt(0); 
		unsigned char* pColor = getColorPtr<unsigned char>( int(eiv2DPt(0)+0.5), int(eiv2DPt(1)+0.5), color );
		pvp3DColors_->push_back( pColor );
	}

	cT1 = microsec_clock::local_time();
	cTDAll = cT1 - cT0 ;
	PRINT( cTDAll );
*/
	return;
}

void CCalibrateKinectExtrinsics::createDepth ( unsigned int uNthView_, const Mat& cvmDepth_, Mat_<int>* pcvmDepthNew_ ) const
{
	pcvmDepthNew_->create( cvmDepth_.size() );
	const int nThreshold = 100;

    for ( int y = 0; y < cvmDepth_.rows; y++ )
        for ( int x = 0; x < cvmDepth_.cols; x++ )
        {
			int c = rawDepth< int >( x,   y  , cvmDepth_ );
			pcvmDepthNew_ ->at<int> (y,x) = c;
        }

	return;
}

void CCalibrateKinectExtrinsics::filterDepth (const double& dThreshould_, const Mat& cvmDepth_, Mat_<int>* pcvmDepthNew_ ) const
{
	pcvmDepthNew_->create( cvmDepth_.size() );

    for ( int y = 0; y < cvmDepth_.rows; y++ )
        for ( int x = 0; x < cvmDepth_.cols; x++ )
        {
			pcvmDepthNew_->at<int> ( y, x ) = 0;
			if( x == 0 || x == cvmDepth_.cols-1 || y == 0 || y == cvmDepth_.rows-1 )
				continue;
			int c = rawDepth< int >( x,   y  , cvmDepth_ );
			int cl= rawDepth< int >( x-1, y  , cvmDepth_ );
			int cr= rawDepth< int >( x+1, y  , cvmDepth_ );
			int cu= rawDepth< int >( x  , y-1, cvmDepth_ );
			int cb= rawDepth< int >( x  , y+1, cvmDepth_ );
			int cul=rawDepth< int >( x-1, y-1, cvmDepth_ );
			int cur=rawDepth< int >( x+1, y-1, cvmDepth_ );
			int cbl=rawDepth< int >( x-1, y+1, cvmDepth_ );
			int cbr=rawDepth< int >( x+1, y+1, cvmDepth_ );

			if( abs( c-cl ) < dThreshould_ && abs( c-cr ) < dThreshould_ && abs( c-cu ) < dThreshould_ && abs( c-cb ) < dThreshould_ &&
				abs( c-cul) < dThreshould_ && abs( c-cur) < dThreshould_ && abs( c-cbl) < dThreshould_ && abs( c-cbr) < dThreshould_ )
				pcvmDepthNew_ ->at<int> (y,x) = c;
        }
	return;
}

}//shuda
