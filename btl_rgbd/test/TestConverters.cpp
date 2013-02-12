#define INFO

#include "../Converters.hpp"
#include "../CVUtil.hpp"
#include "../EigenUtil.hpp"
#include "TestCuda.h"
#include <vector>
#include <list>
using namespace btl::utility;
#include <opencv2/gpu/gpu.hpp>
#include <gl/freeglut.h>
#include "../Camera.h"
#include <limits>
#include "../Optim.hpp"
#include "../cuda/pcl/internal.h"
#include "Teapot.h"
#include "TryCpp.h"



void testSCamera()
{
	btl::image::SCamera sRGB("XtionRGB.yml");//import precomuputed parameters from yml file
	//sRGB.importYML();
	PRINT(sRGB._fFx);
	PRINT(sRGB._fFy);
	PRINT(sRGB._u);
	PRINT(sRGB._v);
	PRINT(sRGB._sHeight);
	PRINT(sRGB._sWidth);
	PRINT(sRGB._cvmDistCoeffs);

	btl::image::SCamera sIR("XtionIR.yml");
	//sIR.importYML();
	PRINT(sIR._fFx);
	PRINT(sIR._fFy);
	PRINT(sIR._u);
	PRINT(sIR._v);
	PRINT(sIR._sHeight);
	PRINT(sIR._sWidth);
	PRINT(sIR._cvmDistCoeffs);
}
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

	for(int r = 0; r < cvDepth.rows; r++ )
	for(int c = 0; c < cvDepth.cols; c++ )
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
void testCOptim(){
	PRINTSTR("test btl::utility::COptim");
	btl::utility::COptim cOptim;

	cOptim.setMethod( btl::utility::COptim::GRADIENTDESCENDENT );
	cOptim.Go();
}
void testSetSE3(){
	cv::Mat _cvmX = (cv::Mat_<double>(1,6) << 0,0.1,.03,0,1,0);
	cv::Mat cvmSE3(4,4,CV_64FC1); cvmSE3= cv::Mat::eye(4, 4, CV_64F);

	cv::Mat cvmR,cvmMinusTR;
	cv::Rodrigues(_cvmX.colRange(0,3),cvmR);
	PRINT(cvmR);
	cvmMinusTR = -_cvmX.colRange(3,6)*cvmR;
	PRINT(cvmMinusTR);
	//set SE3
	double* pSE3 = (double*) cvmSE3.data;
	const double* pR   = ( const double*) cvmR.data;
	const double* pMTR = ( const double*) cvmMinusTR.data;
	for (int r =0; r<4; r++){
		for (int c =0; c<3; c++){
			//assign pR
			if (r <3 ){
				*pSE3++ = *pR++;
			}// if r < 3
			else {
				*pSE3++ = *pMTR++;
			}// r==3
		}//for each col
		*pSE3++;
	}//for each row of SE3
	PRINT(cvmSE3);
}
void test(){
	testSetSE3();
	//testCOptim();
	//testException();
	//testCVUtil();
	//testSCamera();
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
	cv::Mat cvmTest(2,2,CV_16SC2);
	cvmTest.setTo(std::numeric_limits<short>::max());
	PRINT(cvmTest);
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
	Select<2,float,int>::type f=0.1f;
	PRINT(f);
	Select<false,float,int>::type i=0.1f;
	PRINT(i);
}
void tryCVMatPtr2UserAllocatedData(){
	PRINTSTR("try constructor for matrix headers pointing to user-allocated data.");
	double _adModelViewGL[16];
	{
		cv::Mat cvmTemp(4,4,CV_64FC1,(void*)_adModelViewGL);
		cv::setIdentity(cvmTemp);
		PRINT(cvmTemp);
		_adModelViewGL[3]=10;
	}
	for (int i=0;i<16;i++){
		PRINT(_adModelViewGL[i]);
	}
}
void tryCVFloodfill(){
	PRINTSTR("try cv::floodFill()");
	cv::Mat cvmImg = 
	(cv::Mat_<float>(5,5) << 
			  1,2,3,4,5,
			  1,2,6,4,5,
			  1,2,6,4,5,
			  1,6,6,6,5,
			  1,2,3,4,5 );
	PRINT(cvmImg);
	cv::floodFill(cvmImg,cv::Point(4,0), 10, NULL, 0.5,0.5 );
	PRINT(cvmImg);
}
void tryCV()
{
	PRINTSTR("try opencv.");
	//tryCVFloodfill();
	//tryCVMatPtr2UserAllocatedData();
	//tryCVPryDown();
	tryCVMat();
	//tryCVOperator();
	//tryCVMatSetTo();
	//tryCVTypeSelect();
	//tryCVGPU();
}
void tryDataOrderEigenMaxtrix(){
	PRINTSTR("try: Eigen::Matrix3d::data() order");
	{
		Eigen::Matrix<float, 3, 3, Eigen::RowMajor> eimRowMajor; 
		eimRowMajor << 1,2,3,
			4,5,6,
			7,8,9;
		PRINT(eimRowMajor);
		float* pData = eimRowMajor.data();
		for(int i=0; i<9; i++){
			PRINT(*pData++);
		}
		Eigen::Vector3f eivT(1,2,3);
		PRINT(eimRowMajor*eivT);
	}
	{
		Eigen::Matrix<float, 3, 3> eimDefault; 
		eimDefault << 1,2,3,
			4,5,6,
			7,8,9;
		PRINT(eimDefault);
		float* pData = eimDefault.data();
		for(int i=0; i<9; i++){
			PRINT(*pData++);
		}
		Eigen::Vector3f eivT(1,2,3);
		PRINT(eimDefault*eivT);
	}
}
void tryEigenData(){
	PRINTSTR("try: Eigen::Vector3d::data() writable")
	Eigen::Vector3d eivVec;
	double* pData = eivVec.data();
	*pData++ = 1;
	*pData++ = 2;
	*pData = 3;
	PRINT(eivVec);

}
void tryEigenExponentialMap(){
	PRINTSTR("try: Eigen exponential map");
	Eigen::Matrix3d eimSkew;
	btl::utility::setSkew<double>(1.e-10,0.05,0.07,&eimSkew);
	PRINT(eimSkew);
	Eigen::Matrix3d eimR;
	btl::utility::setRotMatrixUsingExponentialMap(1.e-30,0.0,0.0,&eimR);
	PRINT(eimR);
	PRINT(eimR*eimR.transpose());
}
void tryEigenRowMajorAssignment(){
	PRINTSTR("try: Eigen::Matrix RowMajor Assigning");
	{
		Eigen::Matrix3d eimdColMajor;
		eimdColMajor << 1,2,3,
			4,5,6,
			7,8,9;

		PRINT(eimdColMajor);
		Eigen::Matrix3f eimfColMajor = eimdColMajor.cast<float>();
		//Eigen::Matrix<float,3,3,Eigen::RowMajor> eimfRowMajor = eimdColMajor.cast<Eigen::Matrix<float,3,3,Eigen::RowMajor>>();
		pcl::device::Mat33&  devRwRef = pcl::device::device_cast<pcl::device::Mat33> (eimfColMajor);
		PRINT(devRwRef.data[0].x);
		PRINT(devRwRef.data[0].y);
		PRINT(devRwRef.data[0].z);
		PRINT(devRwRef.data[1].x);
		PRINT(devRwRef.data[1].y);
		PRINT(devRwRef.data[1].z);
		PRINT(devRwRef.data[2].x);
		PRINT(devRwRef.data[2].y);
		PRINT(devRwRef.data[2].z);
	}
	{
		Eigen::Matrix<float,3,3,Eigen::RowMajor> eimRowMajor;
		eimRowMajor << 1,2,3,
			4,5,6,
			7,8,9;
		PRINT(eimRowMajor);
		pcl::device::Mat33&  devRwRef = pcl::device::device_cast<pcl::device::Mat33> (eimRowMajor);
		PRINT(devRwRef.data[0].x);
		PRINT(devRwRef.data[0].y);
		PRINT(devRwRef.data[0].z);
		PRINT(devRwRef.data[1].x);
		PRINT(devRwRef.data[1].y);
		PRINT(devRwRef.data[1].z);
		PRINT(devRwRef.data[2].x);
		PRINT(devRwRef.data[2].y);
		PRINT(devRwRef.data[2].z);
	}
}
void tryEigenAffine()
{
	Eigen::Matrix3f m; m = Eigen::AngleAxisf(30, -Eigen::Vector3f::UnitY())* Eigen::AngleAxisf(30, Eigen::Vector3f::UnitX());
	Eigen::Vector3f v(1,2,3);
	Eigen::Vector3f s(2,2,2);

	Eigen::Affine3f eimTry; eimTry.setIdentity();
	eimTry.translate(v);
	eimTry.scale(s);
	eimTry.rotate(m);
	eimTry.translate(-v);
	PRINT(eimTry.matrix()); 

	// equivalent to the following code
	Eigen::Affine3f eimTranslate;eimTranslate.setIdentity();
	eimTranslate.translate(-v);
	Eigen::Affine3f eimRotation; eimRotation.setIdentity();
	eimRotation.rotate(m);
	Eigen::Affine3f eimScale; eimScale.setIdentity();
	eimScale.scale(s);
	Eigen::Affine3f eimTranslateBack; eimTranslateBack.setIdentity();
	eimTranslateBack.translate(v);

	Eigen::Affine3f eimTotal = eimTranslateBack*eimScale*eimRotation*eimTranslate; //
	PRINT(eimTotal.matrix());

	//eular angle
	//Eigen::Matrix3f r; r = Eigen::AngleAxisf(30, -Eigen::Vector3f::UnitY())* Eigen::AngleAxisf(30, Eigen::Vector3f::UnitX());
}
void tryEigen()
{
	transformTeapot();
	//tryEigenAffine();
	//tryEigenRowMajorAssignment();
	//tryEigenExponentialMap();
	//tryDataOrderEigenMaxtrix();
	//tryEigenData();
}
void testConverter(){
	using namespace btl::utility;
	cv::Mat cvmVec;
	cvmVec.create(3,1,CV_64FC1);
	cvmVec.at<double>(0,0) = 1.0;
	cvmVec.at<double>(1,0) = 2.0;
	cvmVec.at<double>(2,0) = 3.0;
	PRINT(cvmVec);
	Eigen::VectorXd eivVec;
	Eigen::Vector3d eivVec2;
	eivVec << cvmVec;
	eivVec2 = eivVec;
	PRINT(eivVec2);

	cv::Mat cvmMat;
	cvmMat.create(2,2,CV_64FC1);
	cvmMat.at<double>(0,0) = 1.0;
	cvmMat.at<double>(1,0) = 2.0;
	cvmMat.at<double>(0,1) = 3.0;
	cvmMat.at<double>(1,1) = 4.0;
	PRINT( cvmMat );
	Eigen::MatrixXd eimMat;
	eimMat << cvmMat;
	Eigen::Matrix2d eimMat2;
	eimMat2 = eimMat;
	PRINT( eimMat2 );
}

void createScriptYml(){
	using namespace btl::utility;

	cv::FileStorage cFSWrite( "KinectLiveview.yml", cv::FileStorage::WRITE );
	ushort uResolution = 0;
	cFSWrite << "uResolution" << uResolution;
	ushort uPyrHeight = 3;
	cFSWrite << "uPyrHeight" << uPyrHeight;
	bool bUseNIRegistration = true;
	cFSWrite << "bUseNIRegistration" << bUseNIRegistration;
	ushort uCubicGridResolution = 512;
	cFSWrite << "uCubicGridResolution" << uCubicGridResolution;
	float fVolumeSize = 3.f;
	cFSWrite << "fVolumeSize" << fVolumeSize;
	int nMode = 3; //1 kinect; 2 recorder; 3 player
	cFSWrite << "nMode" << nMode;
	std::string oniFileName("20121121-153156.oni"); // the openni file 
	cFSWrite << "oniFile" << oniFileName;
	bool bRepeat = false;// repeatedly play the sequence 
	cFSWrite << "bRepeat" << bRepeat;
	int nRecordingTimeInSecond = 30;
	cFSWrite << "nRecordingTimeInSecond" << nRecordingTimeInSecond;

	cFSWrite.release();
}


int main()
{
	try
	{
		createScriptYml();
		//test();
		//cudaTestTry();
		//tryCpp();
		//tryCV();
		//tryEigen();
		//testConverter();
	}
	catch ( std::runtime_error e )
	{
		PRINTSTR( e.what() );
	}

	return 0;
}

