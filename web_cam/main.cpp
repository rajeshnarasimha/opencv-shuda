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
    //opencv c style
    /*cvNamedWindow( "Example2", CV_WINDOW_AUTOSIZE );
    CvCapture* pCapture;
    if( argc == 1 )
    {
        pCapture = cvCreateCameraCapture(0);
    }
    else
    {
        pCapture =  cvCreateFileCapture( argv[1] );
    }
    assert( pCapture != NULL );
        // cvCapture contains all of the info about the AVI file.
        // cvCapture is initialized to the beginning of the AVI
    IplImage* pFrame;
    while ( true )
    {
        pFrame = cvQueryFrame ( pCapture );
            // grabs the next video frame into memory
            // pFrame use the memory allocated to pCapture
        if ( !pFrame ) break;
        cvShowImage( "Example2", pFrame);
        char c = cvWaitKey( 33 );
            // wait for 33ms
        if( c == 27 )break; //press "ESC" to break the loop
    }
    cvReleaseCapture( &pCapture );
    cvDestroyWindow( "Example2" );*/

    //opencv cpp style
    cv::VideoCapture cap ( 1 ); // 0: open the default camera
								// 1: open the integrated webcam

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
