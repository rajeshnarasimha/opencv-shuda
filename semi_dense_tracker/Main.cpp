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

#include "SemiDenseTracker.h"
#include "SemiDenseTrackerOrb.h"
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

float _aLight[1024];

void initLight(){
	boost::mt19937 rng; // I don't seed it on purpouse (it's not relevant)
	boost::normal_distribution<> nd(0.0, 1.0);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);

	for (int i=0; i < 1024; ++i){
		_aLight[i] = float(abs(var_nor()));
	}
}

#include <math.h>
//#define  WEB_CAM
int main ( int argc, char** argv )
{
	initLight();
    //opencv cpp style
#ifdef WEB_CAM
	cv::VideoCapture cap ( 1 ); // 0: open the default camera
								// 1: open the integrated webcam
#else
	cv::VideoCapture cap("VTreeTrunk.avi"); //("VSelf.avi");//("VFootball.mkv");//("VCars.avi"); //("VFootball.mkv");//("VRotatePersp.avi");//( "VRectLight.avi" );
	//("VCars.avi"); //("VRotateOrtho.avi"); //("VBranches.avi"); //("VZoomIn.avi");//("VHand.avi"); 
	//("VPerson.avi");//("VHall.avi");// ("VMouth.avi");// // ("VZoomOut.avi");// 
	//  //
#endif

    if ( !cap.isOpened() ) return -1;
	cv::Mat cvmColorFrame;
	cv::Mat cvmColorFrameSmall;

	btl::image::semidense::CSemiDenseTrackerOrb cSDTOrb;
	btl::image::semidense::CSemiDenseTracker cSDTFast;
	cap >> cvmColorFrame; 
	const float fScale = .8f;
	cv::resize(cvmColorFrame,cvmColorFrameSmall,cv::Size(0,0),fScale ,fScale );	cvmColorFrame.release(); cvmColorFrameSmall.copyTo(cvmColorFrame);
	cv::Mat cvmTotalFrame;
	cvmTotalFrame.create(cvmColorFrame.rows*2,cvmColorFrame.cols*2,CV_8UC3);
	cv::Mat cvmROI0(cvmTotalFrame, cv::Rect(	   0,					 0,			 cvmColorFrame.cols, cvmColorFrame.rows));
	cv::Mat cvmROI1(cvmTotalFrame, cv::Rect(       0,			 cvmColorFrame.rows, cvmColorFrame.cols, cvmColorFrame.rows));
	cv::Mat cvmROI2(cvmTotalFrame, cv::Rect(cvmColorFrame.cols,  cvmColorFrame.rows, cvmColorFrame.cols, cvmColorFrame.rows));
	cv::Mat cvmROI3(cvmTotalFrame, cv::Rect(cvmColorFrame.cols,          0,          cvmColorFrame.cols, cvmColorFrame.rows));


	cvmColorFrame.copyTo(cvmROI0); cvmColorFrame.copyTo(cvmROI1); cvmColorFrame.copyTo(cvmROI2); cvmColorFrame.copyTo(cvmROI3);
	bool bIsInitSuccessful;
	bIsInitSuccessful = cSDTFast.initialize( cvmROI2 );
	bIsInitSuccessful = cSDTOrb.initialize( cvmROI1 );

	cv::gpu::GpuMat cvgmColorFrame;

	while(!bIsInitSuccessful){
		cap >> cvmColorFrame; 
		cv::imwrite("tmp.png",cvmColorFrame);
		cv::resize(cvmColorFrame,cvmColorFrameSmall,cv::Size(0,0),fScale ,fScale );	cvmColorFrame.release(); cvmColorFrameSmall.copyTo(cvmColorFrame);
		cv::imwrite("tmp1.png",cvmColorFrame);
		cvmColorFrame.copyTo(cvmROI0); cvmColorFrame.copyTo(cvmROI1); cvmColorFrame.copyTo(cvmROI2); cvmColorFrame.copyTo(cvmROI3);
		bIsInitSuccessful = cSDTOrb.initialize( cvmROI1 );
		bIsInitSuccessful = cSDTFast.initialize( cvmROI2 );
	}

    cv::namedWindow ( "Tracker", 1 );
	bool bStart = false;
	unsigned int uIdx = 0;
    for ( ;;uIdx++ ){
		double t = (double)cv::getTickCount();
		if ( cv::waitKey ( 'a' ) >= 0 ) bStart = true;
		imshow ( "Tracker", cvmTotalFrame );
		if(!bStart) continue;
		//load a new frame
		cap >> cvmColorFrame; 

		if (cvmColorFrame.empty()) {
			cap.set(CV_CAP_PROP_POS_AVI_RATIO,0);//replay at the end of the video
			cap >> cvmColorFrame; 
			cv::resize(cvmColorFrame,cvmColorFrameSmall,cv::Size(0,0),fScale ,fScale ); cvmColorFrame.release(); cvmColorFrameSmall.copyTo(cvmColorFrame);
			cvmColorFrame.copyTo(cvmROI0); cvmColorFrame.copyTo(cvmROI1); cvmColorFrame.copyTo(cvmROI2); cvmColorFrame.copyTo(cvmROI3);
			cSDTOrb.initialize( cvmROI1 );
			cSDTFast.initialize( cvmROI2 );
			cap >> cvmColorFrame; 
			cv::resize(cvmColorFrame,cvmColorFrameSmall,cv::Size(0,0),fScale ,fScale ); cvmColorFrame.release(); cvmColorFrameSmall.copyTo(cvmColorFrame);
			cvmColorFrame.copyTo(cvmROI0); cvmColorFrame.copyTo(cvmROI1); cvmColorFrame.copyTo(cvmROI2); cvmColorFrame.copyTo(cvmROI3);
		}else{
			cvgmColorFrame.upload(cvmColorFrame);
			cvgmColorFrame.convertTo(cvgmColorFrame,CV_8UC3,pow(0.9f,_aLight[uIdx%1024]));
			cvgmColorFrame.download(cvmColorFrame);
			cv::resize(cvmColorFrame,cvmColorFrameSmall,cv::Size(0,0),fScale ,fScale ); cvmColorFrame.release(); cvmColorFrameSmall.copyTo(cvmColorFrame);
			cvmColorFrame.copyTo(cvmROI0); cvmColorFrame.copyTo(cvmROI1); cvmColorFrame.copyTo(cvmROI2); cvmColorFrame.copyTo(cvmROI3);// get a new frame from camera
		}

		cSDTOrb.track( cvmROI1 );
		cSDTFast.track( cvmROI2 );
		cSDTFast.displayCandidates( cvmROI3 );
		t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
		std::cout << "frame time [s]: " << t << " ms" << std::endl;	


		//interactions
        if ( cv::waitKey ( 30 ) >= 0 ){
            break;
        }
    }

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
