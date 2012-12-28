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

//#define  WEB_CAM
int main ( int argc, char** argv )
{
    //opencv cpp style
#ifdef WEB_CAM
	cv::VideoCapture cap ( 1 ); // 0: open the default camera
								// 1: open the integrated webcam
#else
	cv::VideoCapture cap ("VZoomIn.avi");// ("VZoomOut.avi");//("VMouth.avi");//  ("VTreeTrunk.avi"); //("VPerson.avi");//( "VRectLight.avi" );//
	//  ("VRotateOrtho.avi"); //("VBranches.avi"); //("VHand.avi"); //("VCars.avi");//("VRotatePersp.avi");// ("VHall.avi");//
#endif

    if ( !cap.isOpened() ) return -1;
	cv::Mat cvmColorFrame;


	btl::image::semidense::CSemiDenseTrackerOrb cSDT;
	btl::image::semidense::CSemiDenseTracker cSDTFast;
	cap >> cvmColorFrame; 

	cv::Mat cvmTotalFrame;
	cvmTotalFrame.create(cvmColorFrame.rows*2,cvmColorFrame.cols*2,CV_8UC3);
	cv::Mat cvmROI0(cvmTotalFrame, cv::Rect(0,  0,					 cvmColorFrame.cols,cvmColorFrame.rows));
	cv::Mat cvmROI1(cvmTotalFrame, cv::Rect(0,  cvmColorFrame.rows,  cvmColorFrame.cols,cvmColorFrame.rows));
	cv::Mat cvmROI2(cvmTotalFrame, cv::Rect(cvmColorFrame.cols,  cvmColorFrame.rows,cvmColorFrame.cols,cvmColorFrame.rows));
	cv::Mat cvmROI3(cvmTotalFrame, cv::Rect(cvmColorFrame.cols,   0,                cvmColorFrame.cols,cvmColorFrame.rows));

	cvmColorFrame.copyTo(cvmROI0); cvmColorFrame.copyTo(cvmROI1); cvmColorFrame.copyTo(cvmROI2); cvmColorFrame.copyTo(cvmROI3);
	cSDT.initialize( cvmROI1 );
	cSDTFast.initialize( cvmROI2 );
  
    cv::namedWindow ( "Tracker", 1 );
	bool bStart=false;
    for ( ;; ){
		if ( cv::waitKey ( 'a' ) >= 0 ) bStart=true;
		imshow ( "Tracker", cvmTotalFrame );
		if(!bStart) continue;
		//load a new frame
        cap >> cvmColorFrame; 
		
		if (cvmColorFrame.empty()) {
			cap.set(CV_CAP_PROP_POS_AVI_RATIO,0);//replay at the end of the video
			cap >> cvmColorFrame; cvmColorFrame.copyTo(cvmROI0); cvmColorFrame.copyTo(cvmROI1); cvmColorFrame.copyTo(cvmROI2); cvmColorFrame.copyTo(cvmROI3);
			cSDT.initialize( cvmROI1 );
			cSDTFast.initialize( cvmROI2 );
		}else{
			cvmColorFrame.copyTo(cvmROI0); cvmColorFrame.copyTo(cvmROI1); cvmColorFrame.copyTo(cvmROI2); cvmColorFrame.copyTo(cvmROI3);// get a new frame from camera
		}
		cv::putText(cvmROI0, "Input", cv::Point(10, 15), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(1.,0.,0.) );
		//cSDT.trackTest(cvmColorFrame);
		cSDT.track( cvmROI1 );
		cSDTFast.track( cvmROI2 );
		cSDT.displayCandidates( cvmROI3 );
	
		
		//interactions
        if ( cv::waitKey ( 30 ) >= 0 ){
            break;
        }
    }

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
