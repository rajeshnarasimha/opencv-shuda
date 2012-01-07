#include "../Converters.hpp"
#include "../CVUtil.hpp"
#include "../EigenUtil.hpp"

using namespace btl::utility;
#include <vector>

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


	return 0;
}
