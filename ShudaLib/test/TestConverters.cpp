#include "../Converters.hpp"
#include "../CVUtil.hpp"
#include "../EigenUtil.hpp"

using namespace btl::utility;
#include <vector>
void testConvert2DisparityDomain()
{
	std::cout << "test: CVUtil::convert2DisparityDomain ( ) " << std::endl;
	cv::Mat_<float> cvDepth( 10, 10, CV_32FC1);
	cv::Mat cvResult( 10, 10, CV_32FC1);
	cv::Mat cvDisparity;

	for(unsigned int r = 0; r < cvDepth.rows; r++ )
		for(unsigned int c = 0; c < cvDepth.cols; c++ )
		{
			cvDepth.at<float>( r,c) = r* 43 + c;   
		}

		PRINT( cvDepth );
		btl::utility::convert2DisparityDomain<float> ( cvDepth, &cvDisparity) ;
		PRINT( cvDisparity );
		btl::utility::convert2DepthDomain<float> ( cvDisparity, &(cv::Mat_<float>)cvResult ); // convert back
		PRINT( cvResult );
		double dDiff = btl::utility::matNormL1<float>(cvDepth,cvResult);
		PRINT( dDiff );
}
void testCVUtil()
{
	testConvert2DisparityDomain();

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


	{
		PRINTSTR("test: CVUtil::clearMat()");

		cv::Mat_<float> cvmFloat = cv::Mat::ones(2,3,CV_32FC1);
		PRINT( cvmFloat );
		btl::utility::clearMat<float>(0,&cvmFloat);
		PRINT( cvmFloat );

		cv::Mat_<int> cvmInt = cv::Mat::ones(2,3,CV_16SC1);
		PRINT( cvmInt );
		btl::utility::clearMat<int>(0,&cvmInt);
		PRINT( cvmInt );

		cv::Mat cvmAny = cv::Mat::ones(2,3,CV_16SC1);
		PRINT( cvmAny );
		btl::utility::clearMat<int>(0,&cvmAny);
		PRINT( cvmAny );
	}

}

int main()
{

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
	try
	{
		testCVUtil();
    	std::cout << "test: cv::pyrDown ( ) " << std::endl;
	    cv::Mat cvmOrigin = cv::imread("C:\\csxsl\\src\\opencv-shuda\\ShudaLib\\test_data\\example.jpg"); //( "..\\test_data\\example.jpg" );
    	CHECK(NULL != cvmOrigin.data, "test: CVUtil::downSampling ( ): test image load wrong");
	    cv::Mat cvmHalf((cvmOrigin.rows+1)/2,(cvmOrigin.cols+1)/2,cvmOrigin.type());
    	cv:: pyrDown(cvmOrigin, cvmHalf);
	
	    cv::imwrite("C:\\csxsl\\src\\opencv-shuda\\ShudaLib\\test_data\\example_half.jpg",cvmHalf);
	}
	catch ( CError& e )
	{
		if ( std::string const* mi = boost::get_error_info< CErrorInfo > ( e ) )
		{
			std::cerr << "Error Info: " << *mi << std::endl;
		}
	}
	

	std::cout << "try: >> / <<" << std::endl;
	int nL = 1; 
	PRINT(nL);
	PRINT(nL<<1);
	PRINT(nL<<2);
	PRINT(nL<<3);
	return 0;
}

