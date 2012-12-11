/**
* @file main.cpp
* @brief this code shows how to load video from a default webcam, do some simple image processing and save it into an
* 'avi' file
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.
* @date 2011-03-03
*/

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

namespace btl{ namespace device{ namespace semidense{
	//for debug
	void cudaCalcMaxContrast(const cv::gpu::GpuMat& cvgmImage_, const unsigned char ucContrastThreshold_, cv::gpu::GpuMat* pcvgmContrast_);
	//for debug
	void cudaCalcMinDiameterContrast(const cv::gpu::GpuMat& cvgmImage_, cv::gpu::GpuMat* pcvgmContrast_);
	unsigned int cudaCalcSaliency(const cv::gpu::GpuMat& cvgmImage_, const unsigned char ucContrastThreshold_, const float& fSaliencyThreshold_, cv::gpu::GpuMat* pcvgmSaliency_, cv::gpu::GpuMat* pcvgmKeyPointLocations_);
	unsigned int cudaNonMaxSupression(const cv::gpu::GpuMat& cvgmKeyPointLocation_, const unsigned int uMaxSalientPoints_, const cv::gpu::GpuMat& cvgmSaliency_, short2* ps2devLocations_, float* pfdevResponse_);
}//semidense
}//device
}//btl

enum
{
	LOCATION_ROW = 0,
	RESPONSE_ROW,
	ROWS_COUNT
};
int main ( int argc, char** argv )
{
    //opencv cpp style
    cv::VideoCapture cap ( 1 ); // 0: open the default camera
								// 1: open the integrated webcam

    if ( !cap.isOpened() ) // check if we succeeded
    {
        return -1;
    }

    cv::Mat cvmColorFrame;
	cap >> cvmColorFrame; // get a new frame from camera
	cv::Mat cvmSaliency;

	cv::gpu::GpuMat cvgmFrame(cvmColorFrame);
	cv::gpu::GpuMat cvgmBuffer(cvmColorFrame);
	cv::gpu::GpuMat cvgmBufferC1(cvmColorFrame.size(),CV_8UC1);
	cv::gpu::GpuMat cvgmSaliency(cvmColorFrame.size(),CV_32FC1);
	//Gaussian filter
	float fSigma = 1.f; // page3: r=3/6 and sigma = 1.f/2.f respectively
	unsigned int uRadius = 3; // 
	unsigned int uSize = 2*uRadius + 1;
	//contrast threshold
	unsigned char ucContrastThresold = 5; // 255 * 0.02 = 5.1
	cv::Ptr<cv::gpu::FilterEngine_GPU> _pBlurFilter = cv::gpu::createGaussianFilter_GPU(CV_8UC3, cv::Size(uSize, uSize), fSigma, fSigma, cv::BORDER_REFLECT_101);
	//saliency threshold
	float fSaliencyThreshold = 0.25;
	
	//# of Max key points
	unsigned int uMaxKeyPoints = 5000;
	//key point locations
	cv::gpu::GpuMat cvgmKeyPointLocation(1, uMaxKeyPoints, CV_16SC2);
	unsigned int uTotalSalientPoints = 0;
	//opencv key points
	cv::gpu::GpuMat cvgmKeyPoints(3, uMaxKeyPoints, CV_32FC1) ;
	unsigned int uFinalSalientPoints = 0;
    cv::namedWindow ( "Tracker", 1 );
    for ( ;; ){
		//load a new frame
        cap >> cvmColorFrame; // get a new frame from camera
		cvgmFrame.upload(cvmColorFrame); // get a new frame from camera
		//processing the frame
		//gaussian filter
		_pBlurFilter->apply(cvgmFrame, cvgmBuffer, cv::Rect(0, 0, cvgmFrame.cols, cvgmFrame.rows));
		//detect key points
		//1.compute the saliency score
		uTotalSalientPoints = btl::device::semidense::cudaCalcSaliency(cvgmBuffer, ucContrastThresold, fSaliencyThreshold, &cvgmSaliency, &cvgmKeyPointLocation);
		uTotalSalientPoints = std::min( uTotalSalientPoints, uMaxKeyPoints );
		//2.do a non-max suppression
		uFinalSalientPoints = btl::device::semidense::cudaNonMaxSupression(cvgmKeyPointLocation, uTotalSalientPoints, cvgmSaliency, cvgmKeyPoints.ptr<short2>(LOCATION_ROW), cvgmKeyPoints.ptr<float>(RESPONSE_ROW));
		uFinalSalientPoints = std::min( uFinalSalientPoints, uTotalSalientPoints );
		//3.initialize particles

		cvgmSaliency.convertTo(cvgmBufferC1,CV_8UC1,255);
		//display the frame
		cvgmBufferC1.download(cvmColorFrame);
        imshow ( "Tracker", cvmColorFrame );
		//interactions
        if ( cv::waitKey ( 30 ) >= 0 ){
            break;
        }
    }

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
