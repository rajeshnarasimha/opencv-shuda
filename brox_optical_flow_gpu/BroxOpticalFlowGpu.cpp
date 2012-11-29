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

	GpuMat img1(cvImg1),cvgmImg1(cvImg1.size(),cv::DataType<float>::type);
	GpuMat img2(cvImg2),cvgmImg2(cvImg2.size(),cv::DataType<float>::type);
	img1.convertTo(cvgmImg1,cv::DataType<float>::type);
	img2.convertTo(cvgmImg2,cv::DataType<float>::type);

    if (img1.empty() || img2.empty())
    {
        cout << "Can't read one of the images" << endl;
        return -1;
    }

	cv::gpu::BroxOpticalFlow cBOF(80,100,0.95,5,20,10);
	GpuMat u,v;
	cBOF(cvgmImg1,cvgmImg2,u,v);

	cv::gpu::GpuMat cvgmMag, cvgmAngle;
	cv::gpu::cartToPolar(u,v,cvgmMag,cvgmAngle,true);

	//translate magnitude to range [0;1]
	double mag_max,mag_min;
	cv::gpu::minMaxLoc(cvgmMag, 0, &mag_max);
	cvgmMag.convertTo(cvgmMag,-1,1.0/mag_max);

	//build hsv image
	cv::gpu::GpuMat cvgmHSV, cvgmOnes(cvgmAngle.size(),CV_32F); cvgmOnes.setTo(1.f);
	std::vector<cv::gpu::GpuMat> vcvgmHSV;
	vcvgmHSV.push_back(cvgmAngle);
	vcvgmHSV.push_back(cvgmOnes);
	vcvgmHSV.push_back(cvgmMag);
	cv::gpu::merge(vcvgmHSV,cvgmHSV);

	//convert to BGR and show
	cv::gpu::GpuMat cvgmBGR;
	cv::gpu::cvtColor(cvgmHSV,cvgmBGR,cv::COLOR_HSV2BGR);//cvgmBGR is CV_32FC3 matrix
	cv::namedWindow("optical flow", 0);
	cvgmBGR.convertTo(cvgmBGR,CV_8UC3,255);
	Mat cvmBGR;
	cvgmBGR.download(cvmBGR);
	
	cv::imwrite("optical.png",cvmBGR);
	cv::imshow("optical flow", cvmBGR);
	cv::waitKey(0);

    return 0;
}
