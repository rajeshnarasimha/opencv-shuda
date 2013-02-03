#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <opencv2/legacy/legacy.hpp>

using namespace std;
using namespace cv;
using namespace cv::gpu;

bool sort_pred ( const DMatch& m1_, const DMatch& m2_ )
{
	return m1_.distance < m2_.distance;
}

int main(int argc, char* argv[])
{
	cv::Mat cvmGray1 = imread("rgb0.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat cvmGray2 = imread("rgb1.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	SURF surf(100,4,2,false,true);
	// detecting keypoints & computing descriptors
	FREAK* pFreak = new FREAK(); 

	vector<KeyPoint> vKeypoints1; 
	vector<KeyPoint> vKeypoints2; 

	Mat cvmDescriptor1;
	Mat cvmDescriptor2;

	vector<DMatch> vMatches;
	BruteForceMatcher<HammingLUT> matcher;  


	
	surf(cvmGray1, cv::Mat(), vKeypoints1);
	pFreak->compute( cvmGray1, vKeypoints1, cvmDescriptor1 );

	surf(cvmGray2, cv::Mat(), vKeypoints2);
	double t = (double)getTickCount();

	pFreak->compute( cvmGray2, vKeypoints2, cvmDescriptor2 );
	t = ((double)getTickCount() - t)/getTickFrequency();

	matcher.match(cvmDescriptor1, cvmDescriptor2, vMatches);  

	

	std::cout << "whole time [s]: " << t << std::endl;	
    sort (vMatches.begin(), vMatches.end(), sort_pred);
    vector<DMatch> closest;

    int nSize = (int)vMatches.size();//>300?300:matches.size();
    cout << "matched point pairs: " << nSize << endl;
	for( int i=0;i < 100;i++) {
        closest.push_back( vMatches[i] );
        cout << vMatches[i].distance << " ";
    }
    // drawing the results
    Mat cvmImgMatches;

	cout << "FOUND " << vKeypoints1.size() << " keypoints on first image" << endl;
	cout << "FOUND " << vKeypoints2.size() << " keypoints on second image" << endl;

    cv::drawMatches( cvmGray1, vKeypoints1, cvmGray2, vKeypoints2, closest, cvmImgMatches);
    
    namedWindow("matches", 0);
    imshow("matches", cvmImgMatches);
    waitKey(0);

    return 0;
}
