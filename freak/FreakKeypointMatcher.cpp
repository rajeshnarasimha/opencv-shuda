#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

#include "Freak.h"
#include "Surf.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

bool sort_pred ( const DMatch& m1_, const DMatch& m2_ )
{
    return m1_.distance < m2_.distance;
}

void help()
{
    cout << "\nThis program demonstrates using SURF_GPU features detector, descriptor extractor and BruteForceMatcher_GPU" << endl;
    cout << "\nUsage:\n\tmatcher_simple_gpu <image1> <image2>" << endl;
}

int main(int argc, char* argv[])
{
   /* if (argc != 3)
    {
        help();
        return -1;
    }*/
	cv::Mat cvmGray1 = imread("rgb0.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat cvmGray2 = imread("rgb1.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	
	cv::gpu::GpuMat cvgmGray1(cvmGray1), cvgmGray2(cvmGray2);
	cv::gpu::GpuMat cvgmKeyPoint1, cvgmKeyPoint2;
	cv::gpu::GpuMat cvgmDescriptor1, cvgmDescriptor2;

	btl::image::CSurf surf(4000,4,2,false,0.1f,true);
	// detecting keypoints & computing descriptors
	btl::image::FREAK *pFreak = new btl::image::FREAK();

	vector<DMatch> vMatches;
	BruteForceMatcher_GPU<HammingLUT> matcher;  

	double t = (double)getTickCount();
	
	surf(cvgmGray1, cv::gpu::GpuMat(), cvgmKeyPoint1);
	cvgmDescriptor1.create( cvgmKeyPoint1.cols,512/8, CV_8U ); 	cvgmDescriptor1.setTo(0); //allocate memory
	unsigned int uT1 = pFreak->gpuCompute( cvgmGray1, surf.getImgInt(), cvgmKeyPoint1, &cvgmDescriptor1 );

	surf(cvgmGray2, cv::gpu::GpuMat(), cvgmKeyPoint2);
	cvgmDescriptor2.create( cvgmKeyPoint2.cols,512/8, CV_8U ); 	cvgmDescriptor2.setTo(255); //allocate memory
	unsigned int uT2 = pFreak->gpuCompute( cvgmGray2, surf.getImgInt(), cvgmKeyPoint2, &cvgmDescriptor2 );
	
	matcher.match(cvgmDescriptor1, cvgmDescriptor2, vMatches);  

	t = ((double)getTickCount() - t)/getTickFrequency();

	std::cout << "whole time [s]: " << t << std::endl;	
    sort (vMatches.begin(), vMatches.end(), sort_pred);
    vector<DMatch> closest;

    int nSize = (int)vMatches.size();//>300?300:matches.size();
    cout << "matched point pairs: " << nSize << endl;
	for( int i=0;i < 30;i++) {
        closest.push_back( vMatches[i] );
        cout << vMatches[i].distance << " ";
    }
    // drawing the results
    Mat cvmImgMatches;

	vector<KeyPoint> vKeypoints1; 
	vector<KeyPoint> vKeypoints2; 

	pFreak->downloadKeypoints(cvgmKeyPoint1,vKeypoints1);
	pFreak->downloadKeypoints(cvgmKeyPoint2,vKeypoints2);

	cout << "FOUND " << vKeypoints1.size() << " keypoints on first image" << endl;
	cout << "FOUND " << vKeypoints2.size() << " keypoints on second image" << endl;

    cv::drawMatches( cvmGray1, vKeypoints1, cvmGray2, vKeypoints2, closest, cvmImgMatches);
    
    namedWindow("matches", 0);
    imshow("matches", cvmImgMatches);
    waitKey(0);

    return 0;
}
