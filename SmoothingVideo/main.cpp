#include <opencv/cv.h>
#include <opencv/highgui.h>

int _nSliderPosition = 0;
CvCapture* _pCapture = NULL;

void example2_4( IplImage* pImage_ )
{

    // create a window to show our input image
    cvShowImage( "Example4-in",pImage_ );

    // create an image to hold the smoothed output
    IplImage* pOut = cvCreateImage( cvGetSize( pImage_ ), IPL_DEPTH_8U, 3 );
        
    // do the smoothing
    cvSmooth( pImage_, pOut, CV_GAUSSIAN, 11,11 );

    // show the smoothed image in the output window
    cvShowImage( "Example4-out", pOut);

    // be tidy
    cvReleaseImage( &pOut );

}

/**
* @brief
* a callback that will perform the relocation.
*
* @param nPos_
* position of frames.
*
*/
void onTrackbarSlide(int nPos_)
{
    cvSetCaptureProperty( _pCapture, CV_CAP_PROP_POS_FRAMES, nPos_ );
}

int main( int argc, char** argv )
{
    // create some windows to show input and output images in.
    cvNamedWindow( "Example4-in" );
    cvNamedWindow( "Example4-out" );

    _pCapture = cvCreateFileCapture( argv[1] ); 
        // cvCapture contains all of the info about the AVI file.
        // cvCapture is initialized to the beginning of the AVI
    int nFrames = (int) cvGetCaptureProperty( _pCapture, CV_CAP_PROP_FRAME_COUNT );
    IplImage* pFrame;
    while ( true )
    {
        pFrame = cvQueryFrame ( _pCapture );
            // grabs the next video frame into memory
            // pFrame use the memory allocated to pCapture
        if ( !pFrame ) break;
        char c = cvWaitKey( 33 );
            // wait for 33ms
        if( c == 27 )break; //press "ESC" to break the loop

        example2_4(pFrame);    
    }

    cvReleaseCapture( &_pCapture );
    
    // wait for the user to hit a key, then clean up the windows
    cvWaitKey( 0 );
    cvDestroyWindow( "Example4-in" );
    cvDestroyWindow( "Example4-out" );

    return 0;
}
