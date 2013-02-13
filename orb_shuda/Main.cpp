#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "Fast.h"
#include "Orb.h"                                    

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
    /*if (argc != 3)
    {
        help();
        return -1;
    }*/
	cv::Mat cvImg1 = imread(/*argv[1]*/"rgb0.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat cvImg2 = imread(/*argv[2]*/"rgb1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    GpuMat img1(cvImg1);
    GpuMat img2(cvImg2);
    if (img1.empty() || img2.empty()) {
        cout << "Can't read one of the images" << endl;
        return -1;
    }

	btl::image::COrb cOrb(200,2.0,4,10,1,2,0,31);

    // detecting keypoints & computing descriptors
    GpuMat keypoints1GPU, keypoints2GPU;
    GpuMat descriptors1GPU, descriptors2GPU;

    cOrb(img1, GpuMat(), &keypoints1GPU, &descriptors1GPU); 
	cOrb(img2, GpuMat(), &keypoints2GPU, &descriptors2GPU);

    cout << "FOUND " << keypoints1GPU.cols << " keypoints on first image" << endl;
    cout << "FOUND " << keypoints2GPU.cols << " keypoints on second image" << endl;

    // downloading results
    vector<KeyPoint> keypoints1, keypoints2;
    vector<float> descriptors1, descriptors2;
    vector<DMatch> matches;
	
	cOrb.downloadKeyPoints(keypoints1GPU, &keypoints1);
	cOrb.downloadKeyPoints(keypoints2GPU, &keypoints2);

	BruteForceMatcher_GPU<HammingLUT> matcher;  
	matcher.match(descriptors1GPU, descriptors2GPU, matches);  
		
    sort (matches.begin(), matches.end(), sort_pred);
    vector<DMatch> closest;

    int nSize = matches.size();//>300?300:matches.size();
    cout << "matched point pairs: " << nSize << endl;
	for( int i=0;i < 30;i++)
    {
        closest.push_back( matches[i] );
        //cout << matches[i].distance << endl;
    }
    // drawing the results
    Mat img_matches;
    cv::drawMatches( cvImg1, keypoints1, cvImg2, keypoints2, closest, img_matches);
    
    namedWindow("matches", 0);
    imshow("matches", img_matches);
    waitKey(0);

    return 0;
}
