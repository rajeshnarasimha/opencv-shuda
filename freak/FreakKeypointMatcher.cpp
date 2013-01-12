#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

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
    if (argc != 3)
    {
        help();
        return -1;
    }
	cv::Mat cvmGray1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat cvmGray2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

	vector<KeyPoint> vKeypoints1; cv::Mat cvmDescriptors1;
	vector<KeyPoint> vKeypoints2; cv::Mat cvmDescriptors2;

	cv::SURF surf(2000,4);
	surf.detect(cvmGray1, vKeypoints1);
	surf.detect(cvmGray2, vKeypoints2);

	// detecting keypoints & computing descriptors
	cv::FREAK *pFreak = new cv::FREAK();
	pFreak->compute( cvmGray1, vKeypoints1, cvmDescriptors1 );
	pFreak->compute( cvmGray2, vKeypoints2, cvmDescriptors2 );


    cout << "FOUND " << vKeypoints1.size() << " keypoints on first image" << endl;
    cout << "FOUND " << vKeypoints2.size() << " keypoints on second image" << endl;

    // upload results
	cv::gpu::GpuMat cvgmDescriptors1(cvmDescriptors1);
	cv::gpu::GpuMat cvgmDescriptors2(cvmDescriptors2);

    vector<DMatch> vMatches;
	BruteForceMatcher_GPU<HammingLUT> matcher;  
	double t = (double)getTickCount();
	matcher.match(cvgmDescriptors1, cvgmDescriptors2, vMatches);  
	t = ((double)getTickCount() - t)/getTickFrequency();
	std::cout << "match time [s]: " << t << std::endl;	
    sort (vMatches.begin(), vMatches.end(), sort_pred);
    vector<DMatch> closest;

    int nSize = vMatches.size();//>300?300:matches.size();
    cout << "matched point pairs: " << nSize << endl;
	for( int i=0;i < 100;i++) {
        closest.push_back( vMatches[i] );
        //cout << matches[i].distance << endl;
    }
    // drawing the results
    Mat cvmImgMatches;
    cv::drawMatches( cvmGray1, vKeypoints1, cvmGray2, vKeypoints2, closest, cvmImgMatches);
    
    namedWindow("matches", 0);
    imshow("matches", cvmImgMatches);
    waitKey(0);

    return 0;
}
