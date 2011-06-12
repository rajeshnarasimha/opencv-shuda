#include "calibratekinectextrinsics.hpp"
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
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
	PRINT( _dDepthThreshold );

	for(unsigned int i = 0; i < _vstrImagePathName.size(); i++ )
    { 
		cv::Mat& Depth =_vUndistortedDepthMaps[i];
		cv::Mat_<unsigned short> DepthInt( _vImageResolution(1), _vImageResolution(0) );
		cv::Mat_<unsigned short> FilteredDepthInt;
		for ( int y = 0; y < Depth.rows; y++ )
        for ( int x = 0; x < Depth.cols; x++ )
        {
			DepthInt.at<unsigned short> ( y, x ) = rawDepth< unsigned short >( x,   y  , Depth );
        }
		btl::utility::filterDepth < unsigned short > ( (unsigned short) _dDepthThreshold, DepthInt, &FilteredDepthInt );
		_vFilteredUndistDepthInts.push_back( FilteredDepthInt );
    }
	return;
}

void CCalibrateKinectExtrinsics::mainFunc ( const boost::filesystem::path& cFullPath_ )
{
	_strFullPath = cFullPath_.file_string();

	parseControlYAML();

 	//load depth camera intrinsics
    //importKinectIntrinsics();
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
	convertDepth (); //convert from rgb format depth to unsigned short format depth
					 //and filter the depth to remove noise

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
    	define3DCorners ( pattern(), views(), &_vv3DCorners );
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

	//buildRegistrationTable();

	//collecting 3D Points
	if( 1 == _nCollect3DPts )
	{
		collect3DPt();
		//collect3DPtAll( &_vveiv3DPts, &_vvp3DColors);
	}

	if( 1 == _nCalibrateDepth )	
	{
		//calibDepth(); 
		cout << "depth calibrated";
	}
	cout << "mainFunc() done." << endl;
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
		else if("collect 3D points" == (*cIt1).first )
			_nCollect3DPts =(*cIt1).second;
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

void CCalibrateKinectExtrinsics::calibDepth()
{
	/*
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
*/
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

void CCalibrateKinectExtrinsics::buildRegistrationTable()
{
	const unsigned short MINDEPTH =   401; 
	const unsigned short MAXDEPTH = 10000; 

	Vector3d vPt; 
	const Eigen::Vector3d& vT = eiVecRelativeTranslation();
    Eigen::Matrix3d mRTrans = eiMatRelativeRotation().transpose();

	Vector2s_type vRGBPxCurr, vRGBPxPrev, mOffset; 
	
	//for ( int y = 0; y < 480; y++ )
	{
    //for ( int x = 0; x < 640; x++ )
	int x = 200; int y = 0;
	{
		unprojectCamera2World< double, double >( x, y, MINDEPTH, eiMatIRK() , &vPt ); 
		projectWorld2Camera<double> ((mRTrans * ( vPt - vT )), eiMatRGBK(), &vRGBPxCurr );
		_mpTable( y, x ).insert ( pair< unsigned short, Vector2s_type >( MINDEPTH , vRGBPxCurr ) );

		for ( unsigned short d = MINDEPTH+1; d < MAXDEPTH; d++ )
        {
			vRGBPxPrev = vRGBPxCurr;
			unprojectCamera2World< double, double >( x, y, d, eiMatIRK() , &vPt ); 
			projectWorld2Camera<double> ((mRTrans * ( vPt - vT )), eiMatRGBK(), &vRGBPxCurr );
			mOffset = vRGBPxCurr - vRGBPxPrev;
			if( 0 != mOffset(0) || 0 != mOffset(1) )
			{
				_mpTable( y, x ).insert ( pair< unsigned short, Vector2s_type >( d , vRGBPxCurr ) );
			}
        }
	}
	PRINT( y );
	}

	PRINT( _mpTable( 0, 200 ) );

	//exportTable();
	cout << "exportTable() done."<< endl;

	//importTable();
	cout << "importTable() done."<< endl;
	PRINT( _mpTable( 0, 200 ) );

	boost::posix_time::ptime cT0 ( microsec_clock::local_time() );
	boost::posix_time::ptime cT1 ( microsec_clock::local_time() );

	cT0 = microsec_clock::local_time() ;
	unprojectCamera2World< double, double >( 200, 0, 3370, eiMatIRK() , &vPt ); 
	map_type::const_iterator it_mp = _mpTable( 0,200 ).upper_bound( short( 3370 ) );
	it_mp--;
	unprojectCamera2World< double, double >( 200, 0, 570, eiMatIRK() , &vPt ); 
	it_mp = _mpTable( 0,200 ).upper_bound( short( 570 ) );
	it_mp--;
	unprojectCamera2World< double, double >( 200, 0, 7370, eiMatIRK() , &vPt ); 
	it_mp = _mpTable( 0,200 ).upper_bound( short( 7370 ) );
	it_mp--;

	unprojectCamera2World< double, double >( 200, 0, 3370, eiMatIRK() , &vPt ); 
	it_mp = _mpTable( 0,200 ).upper_bound( short( 3370 ) );
	it_mp--;
	unprojectCamera2World< double, double >( 200, 0, 570, eiMatIRK() , &vPt ); 
	it_mp = _mpTable( 0,200 ).upper_bound( short( 570 ) );
	it_mp--;
	unprojectCamera2World< double, double >( 200, 0, 7370, eiMatIRK() , &vPt ); 
	it_mp = _mpTable( 0,200 ).upper_bound( short( 7370 ) );
	it_mp--;


	cT1 = microsec_clock::local_time() ;
	time_duration cTDAll = cT1 - cT0 ;
	
	PRINT( (it_mp)->first );
	PRINT( (it_mp)->second );
	PRINT( cTDAll );


	cT0 = microsec_clock::local_time() ;

	unprojectCamera2World< double, double >( 200,0, 3370, eiMatIRK() , &vPt ); 
	projectWorld2Camera<double> ((mRTrans * ( vPt - vT )), eiMatRGBK(), &vRGBPxCurr );

	unprojectCamera2World< double, double >( 200,0, 570, eiMatIRK() , &vPt ); 
	projectWorld2Camera<double> ((mRTrans * ( vPt - vT )), eiMatRGBK(), &vRGBPxCurr );

	unprojectCamera2World< double, double >( 200,0, 7370, eiMatIRK() , &vPt ); 
	projectWorld2Camera<double> ((mRTrans * ( vPt - vT )), eiMatRGBK(), &vRGBPxCurr );

	unprojectCamera2World< double, double >( 200,0, 3370, eiMatIRK() , &vPt ); 
	projectWorld2Camera<double> ((mRTrans * ( vPt - vT )), eiMatRGBK(), &vRGBPxCurr );

	unprojectCamera2World< double, double >( 200,0, 570, eiMatIRK() , &vPt ); 
	projectWorld2Camera<double> ((mRTrans * ( vPt - vT )), eiMatRGBK(), &vRGBPxCurr );

	unprojectCamera2World< double, double >( 200,0, 7370, eiMatIRK() , &vPt ); 
	projectWorld2Camera<double> ((mRTrans * ( vPt - vT )), eiMatRGBK(), &vRGBPxCurr );


	cT1 = microsec_clock::local_time() ;
	cTDAll = cT1 - cT0 ;

	PRINT( vRGBPxCurr );
	PRINT( cTDAll );

	cout << "map building done." << endl;

}

void CCalibrateKinectExtrinsics::exportTable()
{
    // create and open a character archive for output
    std::ofstream ofs ( "/space/csxsl/src/opencv-shuda/Data/table.xml" );
    boost::archive::xml_oarchive oa ( ofs );

	typedef vector< short > stdv_type;
	typedef map< unsigned short, stdv_type > mapstdv_type;
	typedef vector< mapstdv_type > tablestdv_type;
	tablestdv_type stvTable;
	vector< short > stdvPixel;
	map_type::const_iterator it_mp;

	//convert non-standard table to standard
	for( int r = 0; r <_mpTable.rows(); r++ )
	for( int c = 0; c <_mpTable.cols(); c++ )
	{
		mapstdv_type mpstd;
    	for( it_mp = _mpTable(r, c).begin(); it_mp != _mpTable(r, c).end(); it_mp++ )
		{
			stdvPixel << it_mp->second; 
			mpstd.insert( pair<  unsigned short, vector< short > >( it_mp->first, stdvPixel ) );
		}
		stvTable.push_back( mpstd );
	}

	oa << BOOST_SERIALIZATION_NVP ( stvTable );

    return;
}

void CCalibrateKinectExtrinsics::importTable()
{
	typedef vector< short > stdv_type;
	typedef map< unsigned short, stdv_type > mapstdv_type;
	typedef vector< mapstdv_type > tablestdv_type;
	tablestdv_type stvTable;

    // create and open a character archive for output
    std::ifstream ifs ( "/space/csxsl/src/opencv-shuda/Data/table.xml" );
	boost::archive::xml_iarchive ia ( ifs );

	ia >> BOOST_SERIALIZATION_NVP ( stvTable );

	Vector2s_type vPixel;
	mapstdv_type::const_iterator it_mp;

	//convert standard table to non-standard
	int n=0; int r,c;
	tablestdv_type::const_iterator it_table = stvTable.begin();
	for( ;it_table != stvTable.end(); it_table++ )
	{
		r = n/640; c = n%640; n++; 
		_mpTable(r,c).clear();
    	for( it_mp = it_table->begin(); it_mp != it_table->end(); it_mp++ )
		{
			it_mp->second >> vPixel; 
			_mpTable(r,c).insert( pair< unsigned short, Vector2s_type >( it_mp->first, vPixel ) );
		}
	}

    return;
}

void CCalibrateKinectExtrinsics::collect3DPtAll(vector< vector< Vector3d > >* pvveiv3DPts_, vector< vector< unsigned char* > >* pvvp3DColors_) const
{
	pvveiv3DPts_->clear();
	pvvp3DColors_->clear();
	for (unsigned int i =0; i< views(); i++ )
	{
		vector< Vector3d > 		 veiv3DPts;
		vector< unsigned char* > vp3DColors;
		collect3DPt(i, &veiv3DPts, &vp3DColors);
		pvveiv3DPts_->push_back(veiv3DPts);
		pvvp3DColors_->push_back(vp3DColors);
		PRINT( i );
		PRINT( veiv3DPts.size() );
		PRINT( vp3DColors.size() );
	}
}

void CCalibrateKinectExtrinsics::collect3DPt(unsigned int uNthView_, vector< Vector3d >* pveiv3DPts_, vector< unsigned char* >* pvp3DColors_) const
{

	pveiv3DPts_->clear();
	pvp3DColors_->clear();
	pveiv3DPts_->resize( 307200 );
	pvp3DColors_->resize( 307200 );

	//3D Pt in camera coordinate system.
	//const Mat& img = undistortedDepth ( uNthView_ );
	//const Mat_<int> img = _vFilteredDepth[uNthView_];
	const Mat_<unsigned short>& cvmDepth = _vFilteredUndistDepthInts[uNthView_];
	const Mat&                  cvmRGB   = undistortedImg ( uNthView_ );

	//collect the Pt w.r.t. [I|0]
	Eigen::Matrix< short, 2, 1> eiv2DPt;
	const Eigen::Vector3d& vT = eiVecRelativeTranslation();
    Eigen::Matrix3d mRTrans = eiMatRelativeRotation().transpose();
	boost::posix_time::ptime cT0 ( microsec_clock::local_time() );

	map_type::const_iterator it_mp;

	vector< Vector3d >::iterator       it_Pt = pveiv3DPts_ ->begin();
	vector< unsigned char* >::iterator it_Cl = pvp3DColors_->begin();
	for ( int y = 0; y < cvmDepth.rows; y++ )
        for ( int x = 0; x < cvmDepth.cols; x++ )
        {
			unprojectCamera2World< double, double >( x, y, cvmDepth.at<unsigned short>(y,x), eiMatIRK() , &(*it_Pt) ); 
			//projectWorld2Camera<double> ((mRTrans * ( (*it_Pt) - vT )), eiMatRGBK(), &eiv2DPt );
			//(*it_Cl) = getColorPtr< unsigned char >( eiv2DPt(0), eiv2DPt(1), cvmRGB );

			it_mp = _mpTable( y,x ).upper_bound( cvmDepth.at<unsigned short>(y,x) );
			it_mp--;
			(*it_Cl) = getColorPtr< unsigned char >( it_mp->second(0), it_mp->second(1), cvmRGB );
			it_Pt++; it_Cl++;
        }

	boost::posix_time::ptime cT1 ( microsec_clock::local_time() );
    time_duration cTDAll = cT1 - cT0 ;
	PRINT( cTDAll );
	
	return;
}


void CCalibrateKinectExtrinsics::collect3DPt() 
{
// register the depth w.r.t. rgb camera
// the useful output are _vppRGBWorld a vector of registered XYZ coordinates w.r.t. rgb camera in the same order with
// rgb images. 
	boost::posix_time::ptime cT0, cT1;

	for(unsigned int uNthView_ = 0; uNthView_ < views(); uNthView_++ )
	{
		//timer on
		cT0 =  microsec_clock::local_time(); 

		double*  pRGBWorldRGB = new double[ 307200*3 ];//in order of RGB image. X,Y,Z coordinate of depth w.r.t. RGB camera reference system

		_vpRGBWorld.push_back( pRGBWorldRGB );
	
		const unsigned short* pDepth = (const unsigned short*)_vFilteredUndistDepthInts[uNthView_].data;

		registration( pDepth );

		const double* pRegistered = registeredDepth();	

		double*  pM = pRGBWorldRGB ;		// initialize the Registered depth as 0
		for (int i=0; i< 307200;i++ )
		{
			*pM++ = *pRegistered++;
			*pM++ = *pRegistered++;
			*pM++ = *pRegistered++;
		}	
		//timer off
		cT1 = microsec_clock::local_time();
	    time_duration cTDAll = cT1 - cT0 ;
		PRINT( cTDAll );
	}

	cout<< "color collection done." << endl;
}
}//shuda
