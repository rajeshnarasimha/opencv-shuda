//#define CV_SSE2 1
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <iostream>
/*
#include <btl/extra/VideoSource/calibratekinect.hpp>

using namespace btl;
using namespace extra;
using namespace videosource;

int main( int argc, char** argv )
{
	cv::Mat img = cv::imread( argv[1] );
	cv::Mat result;

	CCalibrateKinect _cCalibKinect;
	_cCalibKinect.importKinectIntrinsics();

	_cCalibKinect.undistortRGB( img, result );
//	cv::remap ( img, result, mapX, mapY, cv::INTER_NEAREST, cv::BORDER_CONSTANT );

    cv::namedWindow ( "myWindow", 1 );
    while ( true )
    {
	    cv::imshow ( "myWindow", result );

        if ( cv::waitKey ( 30 ) >= 0 )
        {
    	    break;
        }
    }
}
*/
#include <opencv/highgui.h>
#include <iostream>

int main( int argc, char** argv ) 
{ 
//    std::cout<< "here1\n";
    IplImage* img = cvLoadImage( argv[1] ); 
//    std::cout<< "here2\n";
    cvNamedWindow( "Example1", CV_WINDOW_AUTOSIZE );
    cvShowImage( "Example1", img );
    cvWaitKey(0); 
    cvReleaseImage( &img ); 
    cvDestroyWindow( "Example1" );
}

