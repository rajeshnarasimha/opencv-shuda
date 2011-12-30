#include "opencv2/opencv.hpp"
#include <time.h>
#include "Converters.hpp"
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

using namespace btl::utility;

#define SMALL 1e-20

int main(int, char** argv)
{
    //read from .xml using boost //////////////////////////////////////////

#if __linux__
    std::ifstream ifs ( "/space/csxsl/src/opencv-shuda/Data/kinect_intrinsics.xml" );
#else if _WIN32 || _WIN64
	std::ifstream ifs ( "C:\\csxsl\\src\\opencv-shuda\\Data\\kinect_intrinsics.xml" );
#endif
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


    cv::Mat_<double> _mRGBK; 
    cv::Mat_<double> _mRGBDistCoeffs;

    cv::Mat_<double> _mIRK; 
    cv::Mat_<double> _mIRDistCoeffs;

    cv::Mat_<double> _cvmRelativeRotation;
    cv::Mat_<double> _cvmRelativeTranslation;   
	cv::Mat_<int>	 _cvmImageResolution;
 
    //rgb
    stdvRGBKMatrix          >> _mRGBK;
    stdvRGBDistortionCoeff  >> _mRGBDistCoeffs;
    //ir
    stdvIRKMatrix           >> _mIRK;
    stdvIRDistortionCoeff   >> _mIRDistCoeffs;
    //both
    stdvImageResolution     >> _cvmImageResolution;
    stdvRelativeRotaion     >> _cvmRelativeRotation;
    stdvRelativeTranslation >> _cvmRelativeTranslation;



    //write to .yml using cv::FileStorage  ///////////////////////////////////
	cv::FileStorage cFS("C:\\csxsl\\src\\opencv-shuda\\Data\\kinect_intrinsics.yml", cv::FileStorage::WRITE); // change "test.yml" you get yml format 

	cFS << "mRGBK" << _mRGBK;
	cFS << "mRGBDistCoeffs" << _mRGBDistCoeffs;
	cFS << "cvmImageResolution" << _cvmImageResolution;
	cFS << "mIRK" << _mIRK;
	cFS << "mIRDistCoeffs" << _mIRDistCoeffs;
	cFS << "cvmRelativeRotation" << _cvmRelativeRotation;
	cFS << "cvmRelativeTranslation" << _cvmRelativeTranslation;

	cFS.release();

	//read from .yml using cv::FileStorage  ///////////////////////////////////
	FileStorage cFSRead("C:\\csxsl\\src\\opencv-shuda\\Data\\kinect_intrinsics.yml", FileStorage::READ);

	cv::Mat_<double> _mRGBK2; 
	cv::Mat_<double> _mRGBDistCoeffs2;

	cv::Mat_<double> _mIRK2; 
	cv::Mat_<double> _mIRDistCoeffs2;

	cv::Mat_<double> _cvmRelativeRotation2;
	cv::Mat_<double> _cvmRelativeTranslation2;   
	cv::Mat_<int>	 _cvmImageResolution2;

	cFSRead ["mRGBK"] >> _mRGBK2;
	cFSRead ["mRGBDistCoeffs"] >> _mRGBDistCoeffs2;
	cFSRead ["cvmImageResolution"] >> _cvmImageResolution2;
	cFSRead ["mIRK"] >> _mIRK2;
	cFSRead ["mIRDistCoeffs"] >> _mIRDistCoeffs2;
	cFSRead ["cvmRelativeRotation"] >> _cvmRelativeRotation2;
	cFSRead ["cvmRelativeTranslation"] >> _cvmRelativeTranslation2;

	double dDif = cv::norm( _mRGBK - _mRGBK2, cv::NORM_L1 );
	int nFail = 0;
	if( dDif > SMALL )
	{
		PRINT( _mRGBK );
		PRINT( _mRGBK2 );
		nFail++;
	}

	dDif = cv::norm( _mRGBDistCoeffs - _mRGBDistCoeffs2, cv::NORM_L1 );
	if( dDif > SMALL )
	{
		PRINT( _mRGBDistCoeffs );
		PRINT( _mRGBDistCoeffs2 );
		nFail++;
	}

	dDif = cv::norm( _cvmImageResolution - _cvmImageResolution2, cv::NORM_L1 );
	if( dDif > SMALL )
	{
		PRINT( _cvmImageResolution );
		PRINT( _cvmImageResolution2 );
		nFail++;
	}

	dDif = cv::norm( _mIRK - _mIRK2, cv::NORM_L1 );
	if( dDif > SMALL )
	{
		PRINT( _mIRK );
		PRINT( _mIRK2 );
		nFail++;
	}

	dDif = cv::norm( _mIRDistCoeffs - _mIRDistCoeffs2, cv::NORM_L1 );
	if( dDif > SMALL )
	{
		PRINT( _mIRDistCoeffs );
		PRINT( _mIRDistCoeffs2 );
		nFail++;
	}

	dDif = cv::norm( _cvmRelativeRotation - _cvmRelativeRotation2, cv::NORM_L1 );
	if( dDif > SMALL )
	{
		PRINT( _cvmRelativeRotation );
		PRINT( _cvmRelativeRotation2 );
		nFail++;
	}

	dDif = cv::norm( _cvmRelativeTranslation - _cvmRelativeTranslation2, cv::NORM_L1 );
	if( dDif > SMALL )
	{
		PRINT( _cvmRelativeTranslation );
		PRINT( _cvmRelativeTranslation2 );
		nFail++;
	}

	if( !nFail )
		std::cout << "Test success.\n";

	return 0;
}
