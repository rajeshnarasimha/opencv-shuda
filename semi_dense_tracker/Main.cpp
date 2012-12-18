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

//#define  WEB_CAM
int main ( int argc, char** argv )
{
    //opencv cpp style
#ifdef WEB_CAM
	cv::VideoCapture cap ( 1 ); // 0: open the default camera
								// 1: open the integrated webcam
#else
	cv::VideoCapture cap ("VHall.avi"); //( "VTreeTrunk.avi" ); 
#endif

    if ( !cap.isOpened() ) return -1;
	cv::Mat cvmColorFrame;
	btl::image::semidense::CSemiDenseTracker cSDT;
	cap >> cvmColorFrame;
	//cSDT.init( cvmColorFrame );
	cSDT.initialize( cvmColorFrame );
  
    cv::namedWindow ( "Tracker", 1 );
    for ( ;; ){
		//load a new frame
        cap >> cvmColorFrame; // get a new frame from camera
		if (cvmColorFrame.empty()) {
			cap.set(CV_CAP_PROP_POS_AVI_RATIO,0);//replay at the end of the video
			cap >> cvmColorFrame;
			cSDT.initialize( cvmColorFrame );
		}
		
		//cSDT.trackTest(cvmColorFrame);
		cSDT.track(cvmColorFrame);
	
		imshow ( "Tracker", cvmColorFrame );
		//interactions
        if ( cv::waitKey ( 30 ) >= 0 ){
            break;
        }
    }

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
