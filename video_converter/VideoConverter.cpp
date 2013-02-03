/**
* @file main.cpp
* @brief this code shows how to load video from a default webcam, do some simple image processing and save it into an
* 'avi' file
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.
* @date 2011-03-03
*/

#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <iostream>

int main ( int argc, char** argv )
{
    //opencv cpp style
    //cv::VideoCapture cap ( 1 ); // 0: open the default camera
								// 1: open the integrated webcam
	cv::VideoCapture cap("test.mkv");
    if ( !cap.isOpened() ) // check if we succeeded
    {
        return -1;
    }

    cv::Mat edges;
    cv::Mat frame;
    cv::Mat cvColorFrame;

    cap >> frame; // get a new frame from camera
    cv::VideoWriter cvSav ( std::string ( "sav.avi" ), CV_FOURCC ( 'M', 'J', 'P', 'G' ), 30, frame.size(), true );

    if ( !cvSav.isOpened() )
    {
        std::cout << "video file not opened!" << std::endl;
    }

    cv::namedWindow ( "edges", 1 );

    for ( ;; )
    {
        cap >> frame; // get a new frame from camera
		if (frame.empty()) break;
        //cv::cvtColor ( frame, edges, CV_BGR2GRAY );
        //cv::GaussianBlur ( frame, edges, cv::Size ( 7, 7 ), 1.5, 1.5 );
        //cv::Canny ( edges, edges, 0, 30, 3 );
        imshow ( "edges", frame );

        if ( cv::waitKey ( 30 ) >= 0 ) {
            break;
        }
        //cvtColor(edges, cvColorFrame, CV_GRAY2RGB);
		//cvSav.write(cvColorFrame);
		
        cvSav << frame;
    }

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
