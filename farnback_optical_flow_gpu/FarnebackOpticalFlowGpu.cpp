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
int video() {

	cv::Mat cvmColorFrame;
	cv::Mat cvmBGR;

	cv::gpu::GpuMat cvgmColorFrame;
	cv::gpu::GpuMat cvgmGrayFramePrev;
	cv::gpu::GpuMat cvgmGrayFrameCurr;


	cv::VideoCapture cap("VBranches.avi");
	if ( !cap.isOpened() ) return -1;
	cap >> cvmColorFrame; cvgmColorFrame.upload(cvmColorFrame);
	cv::gpu::cvtColor(cvgmColorFrame,cvgmGrayFramePrev,CV_RGB2GRAY);


	cv::gpu::FarnebackOpticalFlow cFOF;
	cv::gpu::GpuMat u,v;

	cv::gpu::GpuMat cvgmMag, cvgmAngle;
	cv::gpu::GpuMat cvgmHSV, cvgmOnes(cvgmGrayFramePrev.size(),CV_32F);
	std::vector<cv::gpu::GpuMat> vcvgmHSV;
	cv::gpu::GpuMat cvgmBGR;

	cv::namedWindow("optical flow", 0);

	for ( ;; ){
		double t = (double)cv::getTickCount();
		int nKey = cv::waitKey(1);
		if ( nKey == 'q') break;

		cap >> cvmColorFrame; 

		if (cvmColorFrame.empty()) {
			cap.set(CV_CAP_PROP_POS_AVI_RATIO,0);//replay at the end of the video
			cap >> cvmColorFrame; 
		}
		cvgmColorFrame.upload(cvmColorFrame);
		cv::gpu::cvtColor(cvgmColorFrame,cvgmGrayFrameCurr,CV_RGB2GRAY);
		//

		cFOF(cvgmGrayFramePrev,cvgmGrayFrameCurr,u,v);


		cv::gpu::cartToPolar(u,v,cvgmMag,cvgmAngle,true);

		//translate magnitude to range [0;1]
		double mag_max,mag_min;
		cv::gpu::minMaxLoc(cvgmMag, 0, &mag_max);
		cvgmMag.convertTo(cvgmMag,-1,1.0/mag_max);

		//build hsv image
		cvgmOnes.setTo(1.f);
		vcvgmHSV.push_back(cvgmAngle);
		vcvgmHSV.push_back(cvgmOnes);
		vcvgmHSV.push_back(cvgmMag);
		cv::gpu::merge(vcvgmHSV,cvgmHSV);

		//convert to BGR and show

		cv::gpu::cvtColor(cvgmHSV,cvgmBGR,cv::COLOR_HSV2BGR);//cvgmBGR is CV_32FC3 matrix

		cvgmBGR.convertTo(cvgmBGR,CV_8UC3,255);
		cvgmBGR.download(cvmBGR);

		cv::imshow("optical flow", cvmBGR);

		//
		cvgmGrayFrameCurr.copyTo(cvgmGrayFramePrev);
		vcvgmHSV.clear();

		t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
		std::cout << "frame time [s]: " << t*1000 << " ms" << std::endl;	
	}


	return 0;

}

int image(int argc, char* argv[]){
	if (argc != 3)
    {
        help();
        return -1;
    }
	cv::Mat cvmImg1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat cvmImg2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

	GpuMat cvgmImg1(cvmImg1);
	GpuMat cvgmImg2(cvmImg2);

    if (cvgmImg1.empty() || cvgmImg2.empty())
    {
        cout << "Can't read one of the images" << endl;
        return -1;
    }

	cv::gpu::FarnebackOpticalFlow cFOF;
	GpuMat u,v;
	cFOF(cvgmImg1,cvgmImg2,u,v);

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

int main(int argc, char* argv[])
{
	return video();
	//return image(argc, argv);
}





