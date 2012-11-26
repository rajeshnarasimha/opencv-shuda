#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
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
	cv::Mat cvImg1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat cvImg2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat cvT1(cvImg1.rows,cvImg1.cols,CV_32F);
    cv::Mat cvT2(cvImg1.rows,cvImg1.cols,CV_32F);

	float* p1 = (float*) cvT1.data;
	float* p2 = (float*) cvT2.data;
	unsigned char* pt1 = cvImg1.data;
	unsigned char* pt2 = cvImg2.data;
	for (int i=0; i<cvImg1.rows*cvImg1.cols; i++)
	{
		unsigned char c = *pt1; pt1++;
		*p1++ = float (c); 
		c = *pt2++; 
		*p2++ = float (c);
	}
	
	GpuMat img1(cvT1);
    GpuMat img2(cvT2);
    if (img1.empty() || img2.empty())
    {
        cout << "Can't read one of the images" << endl;
        return -1;
    }

	cv::gpu::BroxOpticalFlow cBOF(80,100,0.5,5,20,10);
	GpuMat u,v;
	cBOF(img1,img2,u,v);

	cv::Mat xy[2]; //X,Y
	u.download(xy[0]);
	v.download(xy[1]);
	cv::Mat magnitude, angle;
	cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);

	//translate magnitude to range [0;1]
	double mag_max;
	cv::minMaxLoc(magnitude, 0, &mag_max);
	magnitude.convertTo(magnitude, -1, 1.0/mag_max);

	//build hsv image
	cv::Mat _hsv[3], hsv;
	_hsv[0] = angle;
	_hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
	_hsv[2] = magnitude;
	cv::merge(_hsv, 3, hsv);

	//convert to BGR and show
	Mat bgr;//CV_32FC3 matrix
	cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
	cv::imwrite("flow.jpg",bgr);
	cv::namedWindow("optical flow", 0);
	cv::imshow("optical flow", bgr);
	cv::waitKey(0);

    return 0;
}
