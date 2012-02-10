#define INFO

#include "../Converters.hpp"
#include "../CVUtil.hpp"
#include "../EigenUtil.hpp"
#include "TestCuda.h"
#include <vector>
using namespace btl::utility;
#include <opencv2/gpu/gpu.hpp>

void testCVUtilOperators()
{
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
}
void testMatNormL1()
{
	std::cout << "test0.0: matNormL1 ( std::vector<> ) " << std::endl;
	std::vector< int > vTest1,vTest2;
	for(int i=0; i<3; i++ )
	{
		vTest1.push_back(i);
		vTest2.push_back(i);
	}
	int nDif = matNormL1<int>( vTest1,vTest2 );
	PRINT( nDif );

	std::cout << "test0.1: matNormL1 ( cv::Mat_<> ) " << std::endl;
	cv::Mat_< int > cvmTest1( 3,1, CV_32S ),cvmTest2( 3,1, CV_32S );
	for(int i=0; i<3; i++ )
	{
		cvmTest1.at<int>(i,0) = i;
		cvmTest2.at<int>(i,0) = i;
	}

	nDif = matNormL1<int>( cvmTest1,cvmTest2 );
	PRINT( nDif );

	std::cout << "test0.2: matNormL1 ( Eigen::Matrix<> ) " << std::endl;
	Eigen::Matrix< double, 3,1 > eimTest1, eimTest2;
	eimTest1 << 0.1,1,1; eimTest2 << 0.2,1.1,1;

	double dDif = matNormL1<double,3,1>( eimTest1,eimTest2 );
	PRINT( dDif );
}
void testConvert2DisparityDomain()
{
	std::cout << "test: CVUtil::convert2DisparityDomain ( ) " << std::endl;
	cv::Mat_<float> cvDepth( 10, 10, CV_32FC1);
	cv::Mat_<float> cvResult;
	cv::Mat cvDisparity;

	for(unsigned int r = 0; r < cvDepth.rows; r++ )
		for(unsigned int c = 0; c < cvDepth.cols; c++ )
		{
			cvDepth.at<float>( r,c) = r* 43 + c;   
		}

		PRINT( cvDepth );
		float fMin, fMax;
		btl::utility::convert2DisparityDomain<float> ( cvDepth, &cvDisparity, &fMax, &fMin) ;
		PRINT(fMin);
		PRINT(fMax);
		PRINT( cvDisparity );
		btl::utility::convert2DepthDomain<float> ( cvDisparity, &cvResult, CV_32FC1 ); // convert back
		PRINT( cvResult );
		double dDiff = btl::utility::matNormL1<float>(cvDepth,cvResult);
		PRINT( dDiff );
}
/*
void testClearMat()
{
	PRINTSTR("test: CVUtil::clearMat()");

	cv::Mat_<float> cvmFloat = cv::Mat::ones(2,3,CV_32FC1);
	PRINT( cvmFloat );
	cvmFloat.setTo(0);
	//btl::utility::clearMat<float>(0,&cvmFloat);
	PRINT( cvmFloat );

	cv::Mat_<int> cvmInt = cv::Mat::ones(2,3,CV_16SC1);
	PRINT( cvmInt );
	cvmInt.setTo(0);
	//btl::utility::clearMat<int>(0,&cvmInt);
	PRINT( cvmInt );

	cv::Mat cvmAny = cv::Mat::ones(2,3,CV_16SC1);
	PRINT( cvmAny );
	btl::utility::clearMat<short>(0,&cvmAny);
	PRINT( cvmAny );
}
*/
void testDownSampling()
{
	std::cout << "test: CVUtil::downSampling() " << std::endl;
	//cv::Mat cvmData = cv::Mat::ones(10,20,CV_32FC1);
	//cvmData *= 3;
	cv::Mat cvmData = (cv::Mat_<float>(4,4) << 11,12,13,14, 21,22,23,24, 31,32,33,34, 41,42,43,44);
	cv::Mat cvmDataHalf(cvmData.rows/2,cvmData.cols/2,cvmData.type());
	PRINT(cvmData.rows);
	//cv:: pyrDown(cvmData, cvmDataHalf);
	btl::utility::downSampling<float>(cvmData,&cvmDataHalf);

	PRINT(cvmData.size());
	PRINT(cvmData);
	PRINT(cvmDataHalf.size());
	PRINT(cvmDataHalf);
}
void cvUtilColor()
{
	PRINTSTR("try: btl::utility::aColors[]");
	for(int i=0; i<8; i++)
	{
		cv::Mat_<unsigned char> cvColor(3,1);
		PRINT(cvColor)
		cvColor.data = __aColors[i];
		PRINT(cvColor);
	}
}
void testCVUtil()
{
	testMatNormL1();
	testCVUtilOperators();
	testConvert2DisparityDomain();
	testDownSampling();
	cvUtilColor();
}
void testException()
{
	PRINTSTR( "testException()" );
	try
	{
		BTL_ASSERT(false,"Test error message.");
	}
	catch (std::runtime_error& e)
	{
		PRINTSTR( e.what() );
	}
}
///////////////////////////////////////
//try
void tryCppOperator()
{
	std::cout << "try: >> / <<" << std::endl;
	int nL = 1; 
	PRINT(nL);
	PRINT(nL<<1);
	PRINT(nL<<2);
	PRINT(nL<<3);
}
void tryCppLongDouble()
{
	PRINTSTR("try long double and double type the effective digits");
	PRINT( std::numeric_limits<long double>::digits10 );
	PRINT( std::numeric_limits<double>::digits10);
}
void tryStdVectorResize()
{
	PRINTSTR("try std::vector::resize() whether it allocate memory");
	std::vector<int> vInt;
	vInt.resize(3);
	PRINT(vInt);
	vInt[2]=10;
	PRINT(vInt);

	PRINTSTR("try std::vector< <> >::resize() whether it allocate memory");
	std::vector< std::vector< int > > vvIdx;
	vvIdx.resize(3);
	vvIdx[2].push_back(1);
	PRINT(vvIdx);
}
void tryStdVectorConstructor()
{
	PRINTSTR("try std::vector< <> >::vector()");
	std::vector< int > vInt(5,1);
	PRINT( vInt );
}
	enum tp_flag { NO_MERGE, MERGE_WITH_LEFT, MERGE_WITH_RIGHT, MERGE_WITH_BOTH };

void tryStdVectorEnum()
{
	PRINTSTR("try std::vector< enum >");
	std::vector<tp_flag > vMergeFlags(2,NO_MERGE);
	vMergeFlags[1] = MERGE_WITH_BOTH;
	PRINT(vMergeFlags);
}
void tryStdVector()
{
	tryStdVectorResize();
	tryStdVectorConstructor();
	tryStdVectorEnum();
}
void tryCppSizeof()
{
	PRINTSTR("try sizeof() operator");
	PRINT(sizeof(long double));
	PRINT(sizeof(double));
	PRINT(sizeof(short));
}
void tryCppTypeDef()
{
	PRINTSTR("try cpp keyword typedef");
	{
		typedef int tp_int;
		tp_int n;
		n = 1;
		PRINT( n );
		{
			tp_int m;
			m = 2;
			PRINT( m );
		}
	}
	//tp_int o; compile error
}

void tryCpp()
{
	tryCppOperator();
	tryCppLongDouble();
	tryCppSizeof();
	tryCppTypeDef();
	tryStdVector();

}
//try CV
void tryCVPryDown()
{
	std::cout << "try: cv::pyrDown ( ) " << std::endl;
	cv::Mat cvmOrigin = cv::imread("C:\\csxsl\\src\\opencv-shuda\\ShudaLib\\test_data\\example.jpg"); //( "..\\test_data\\example.jpg" );
	CHECK(NULL != cvmOrigin.data, "test: CVUtil::downSampling ( ): test image load wrong");
	cv::Mat cvmHalf((cvmOrigin.rows+1)/2,(cvmOrigin.cols+1)/2,cvmOrigin.type());
	cv:: pyrDown(cvmOrigin, cvmHalf);

	cv::imwrite("C:\\csxsl\\src\\opencv-shuda\\ShudaLib\\test_data\\example_half.jpg",cvmHalf);
}
void tryCVMat()
{
	PRINTSTR("try: cv::Mat( const cv::Mat& )");
	cv::Mat cvmData = cv::Mat::ones( 10, 5, CV_32FC1);
	PRINT(cvmData);
	cv::Mat cvmInit(cvmData);
	PRINT(cvmInit);
	// cvmData and cvmInit shares the same memory allocation
	*(float*)cvmInit.data = 10;
	PRINT(cvmInit);
	PRINT(cvmData);
}
void tryCVOperator()
{
	PRINTSTR("try: cv::Mat::operator =");
	cv::Mat cvmData = cv::Mat::ones( 10, 5, CV_32FC1);
	cv::Mat cvmResult = cv::Mat::zeros(cvmData.size(),cvmData.type());
	PRINT( cvmResult );
	cvmResult = cvmData;
	PRINT( cvmResult );
	// cvmData and cvmResult shares the same memory allocation
	*(float*)cvmResult.data = 10;
	PRINT(cvmResult);
	PRINT(cvmData);
	// use clone
	cvmData = cv::Mat::ones( 10, 5, CV_32FC1);
	cvmResult = cvmData.clone();
	*(float*)cvmResult.data = 10;
	PRINT(cvmResult);
	PRINT(cvmData);
	//
	cvmData = cv::Mat::ones( 10, 5, CV_32FC1);
	std::vector<cv::Mat> vcvmMats;
	vcvmMats.push_back(cvmData);
	*(float*)vcvmMats[0].data = 10;
	PRINT(cvmResult);
	PRINT(vcvmMats[0]);
}
void tryCVMatSetTo()
{
	PRINTSTR("try: cv::Mat::setTo()");
	cv::Mat cvmMat= cv::Mat::zeros(4,4,CV_32FC1);
	PRINT(cvmMat);
	cvmMat.setTo(10);
	PRINT(cvmMat);
}
void tryCVGPU()
{
	try
	{
		PRINTSTR("try cv::gpu");
		cv::Mat src_host = cv::imread("C:\\csxsl\\src\\opencv-shuda\\ShudaLib\\test_data\\example.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		cv::gpu::GpuMat dst, src;
		src.upload(src_host);

		cv::gpu::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);
		
		cv::Mat result_host;
		dst.download(result_host);
		cv::imshow("Result", result_host);
		cv::waitKey();
	}
	catch(const cv::Exception& ex)
	{
		std::cout << "Error: " << ex.what() << std::endl;
	}
}

template <bool, typename T1, typename T2> struct Select { typedef T1 type; };
template <typename T1, typename T2> struct Select<false, T1, T2> { typedef T2 type; };
void tryCVTypeSelect()
{
	PRINTSTR( "template <bool, typename T1, typename T2> struct Select { typedef T1 type; };" );
	Select<2,float,int>::type f=0.1;
	PRINT(f);
	Select<false,float,int>::type i=0.1;
	PRINT(i);
}
void tryCV()
{
	PRINTSTR("try opencv.")
	//tryCVPryDown();
	//tryCVMat();
	//tryCVOperator();
	//tryCVMatSetTo();
	tryCVTypeSelect();
	tryCVGPU();
}

void tryEigen()
{
	PRINTSTR("try: Eigen::Vector3d::data()")
	Eigen::Vector3d eivVec;
	double* pData = eivVec.data();
	*pData++ = 1;
	*pData++ = 2;
	*pData = 3;
	PRINT(eivVec);

}
int main()
{
	try
	{
		//testException();
		//testCVUtil();
		cudaTestTry();
		//try Cpp
		//tryCpp();
		//try CV
		//tryCV();
		//try Eigen
		//tryEigen();
	}
	catch ( std::runtime_error e )
	{
		PRINTSTR( e.what() );
	}

	return 0;
}

