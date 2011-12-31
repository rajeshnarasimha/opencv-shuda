#include "../Converters.hpp"
using namespace btl::utility;
#include <vector>

int main()
{

	{
		std::cout << "test0: matNormL1 ( std::vector<> ) " << std::endl;
		std::vector< int > vTest1,vTest2;
		for(int i=0; i<3; i++ )
		{
			vTest1.push_back(i);
			vTest2.push_back(i);
		}
		int nDif = matNormL1<int>( vTest1,vTest2 );
		PRINT( nDif );

		std::cout << "test0: matNormL1 ( cv::Mat_<> ) " << std::endl;
		cv::Mat_< int > cvmTest1( 3,1, CV_32S ),cvmTest2( 3,1, CV_32S );
		for(int i=0; i<3; i++ )
		{
			cvmTest1.at<int>(i,0) = i;
			cvmTest2.at<int>(i,0) = i;
		}

		nDif = matNormL1<int>( cvmTest1,cvmTest2 );
		PRINT( nDif );
	}

	{
		std::cout << "test1: cv::Mat_<> << vector<> " << std::endl;
		std::vector< int > vTest;
		for(int i=0; i<3; i++ )
		{
			vTest.push_back(i);
		}
		PRINT( vTest );
		cv::Mat_< int > cvmTest;
		cvmTest << vTest;
		PRINT( cvmTest );
		std::cout << "test2:  vector<> >> cv::Mat_<> " << std::endl;
		for(int i=0; i<3; i++ )
		{
			vTest.push_back(i);
		}
		PRINT( vTest );
		vTest >> cvmTest;
		PRINT( cvmTest );
	}

	{
		std::cout << "test3: cv::Mat_<> >> vector<> " << std::endl;
		cv::Mat_< int > cvmTest( 3,1, CV_32S );
		for(int i=0; i<3; i++ )
		{
			cvmTest.at<int>(i,0) = i;
		}
		std::vector< int > vTest;
		cvmTest >> vTest;

		PRINT( cvmTest );
		PRINT( vTest );


		std::cout << "test4: vector<> << cv::Mat_<> " << std::endl;
		for(int i=0; i<3; i++ )
		{
			cvmTest.at<int>(i,0) = i+1;
		}
		vTest << cvmTest;

		PRINT( cvmTest );
		PRINT( vTest );
	}


	return 0;
}

/*
// test pointer
#include <btl/Utility/Converters.hpp>
#include <stdio.h>

void assign( double* pPointer_, int nN_ )
{
	for( int i=0;i<nN_;i++)
	{
		*pPointer_++ = i;
	}
}

int main()
{
	double* pXYZ = new double[ 10 ];
	double* pMoving = pXYZ;

	assign( pXYZ, 10 );
	for( int i=0;i<10;i++)
	{
		PRINT( *pMoving++ );
	}
	pMoving = pXYZ;
	memset( pMoving, 0, 10*sizeof(double)); 
	for( int i=0;i<10;i++)
	{
		PRINT( *pMoving++ );
	}
	pMoving = new double[ 10 ];
	memcpy( pMoving, pXYZ, 10*sizeof(double) );
	double* pM = pMoving;
	for( int i=0;i<10;i++)
	{
		PRINT( *pM++ );
	}

	delete [] pXYZ;
	delete [] pMoving;
}
*/
/*
// test the function absoluteOrientation()
#include <btl/Utility/Converters.hpp>
using namespace std;
using namespace cv;
using namespace btl::utility;

int main()
{
	//constructing the testsuite
	
	Eigen::Vector3d eivT;
	eivT << 45, 
			-78, 
			98;
	Eigen::Matrix3d eimR;
	eimR << 0.36, 0.48, -0.8,
			-0.8, 0.6,  0,
			0.48, 0.64, 0.6;
	Eigen::MatrixXd eimX(3,4), eimY(3,4), Noise(3,4);

	eimX << 0.272132, 0.538001, 0.755920, 0.582317,
            0.728957, 0.089360, 0.507490, 0.100513,
            0.578818, 0.779569, 0.136677, 0.785203;
	
	Noise << -0.23, -0.01, 0.03, -0.06,
			  0.07, -0.09,-0.037, -0.08,
			  0.009, 0.09, -0.056, 0.012;
	double dS = 1.0;
	for( int i=0; i< eimX.cols(); i++ )
	{
		Eigen::Vector3d eivPt;
		eimY.col(i) = dS*eimR*eimX.col(i) + eivT;
	}
	//eimY += Noise;
    Eigen::Matrix3d eimR2;
	Eigen::Vector3d eivT2;
	double dS2;
 	double dError = absoluteOrientation< double >( eimX, eimY, false, &eimR2, &eivT2, &dS2 );
	PRINT( dError );
	PRINT( eimR2 );
	PRINT( eimR );
	PRINT( eivT2 );
	PRINT( eivT );
	PRINT( dS2 );
	PRINT( dS );

	return 0;
}
*/

/*
// test optim.hpp and optim.cpp
#include <btl/Utility/Converters.hpp>
#include "optim.hpp"
int main()
{
	cout << "start ";
    shuda::COptim o;
		try{
	o.SetMdAlg(shuda::COptim::CONJUGATE);
	o.Go();
	o.SetMdAlg(shuda::COptim::DIRECTIONSETS);
	o.Go();
	o.SetMdAlg(shuda::COptim::GRADIENTDESCENDENT);
	o.Go();

	}
	catch ( CError& e )
    {
        if ( string const* mi = boost::get_error_info< CErrorInfo > ( e ) )
        {
            std::cerr << "Error Info: " << *mi << std::endl;
        }
    }

	return 0;
}
*/
/*
// test the function to convert kinect raw depth to meter
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <string>
#include <btl/Utility/Converters.hpp>

using namespace std;
using namespace cv;
using namespace btl::utility;

int main()
{
    cv::Mat cvmDepth = cv::imread ( "depthUndistorted0.bmp" );
	Mat_<double> mDK= ( cv::Mat_<double>(3, 1) << 1.16600151976717, 2842.513906334149, 0.1205634721557855 );
	PRINT( mDK );
	double dDepth = depthInMeters<double> ( 545, 156, cvmDepth, mDK, 1 );
    PRINT( dDepth );
	Mat_<char> cvmColor = getColor< char > ( 545, 156, cvmDepth );
	//cvmColor /= 255.;
    PRINT( cvmColor );
    return 0;
}
*/
/*
//test << converting from cv::Mat_<double> to CvMat
//test << converting from CvMat to cv::Mat_<double>
#include <btl/Utility/Converters.hpp>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <string>
using namespace std;
using namespace cv;
using namespace btl::utility;
int main()
{
	try{

	Mat_<double> cppK; cppK = ( cv::Mat_<double>(3,3) << 1,2,3,4,5,6,7,8,9 );
	PRINT( cppK );

  	CvMat* pcK = cvCreateMat( cppK.rows, cppK.cols, CV_64F );

	*pcK << cppK;
	Mat_<double> cppK2;
	cppK2 << *pcK;

    //assignPtr( &cppK, pcK );
	//assignPtr( pcK, &cppK );

	PRINT( cppK2 );

	}
	catch ( CError& e )
    {
        if ( string const* mi = boost::get_error_info< CErrorInfo > ( e ) )
        {
            std::cerr << "Error Info: " << *mi << std::endl;
        }
    }
	return 0;
}
*/

