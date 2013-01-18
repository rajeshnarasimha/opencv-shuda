#include <opencv/cv.h>
#include <opencv/highgui.h>

int _nSliderPosition = 0;
CvCapture* _pCapture = NULL;


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
    cvNamedWindow( "Example3", CV_WINDOW_AUTOSIZE );
    _pCapture = cvCreateFileCapture( argv[1] ); 
        // cvCapture contains all of the info about the AVI file.
        // cvCapture is initialized to the beginning of the AVI
    int nFrames = (int) cvGetCaptureProperty( _pCapture, CV_CAP_PROP_FRAME_COUNT );
    if( nFrames != 0)
    {
        cvCreateTrackbar( "Position", "Example3", &_nSliderPosition, nFrames, onTrackbarSlide );
    }
    IplImage* pFrame;
    while ( true )
    {
        pFrame = cvQueryFrame ( _pCapture );
            // grabs the next video frame into memory
            // pFrame use the memory allocated to pCapture
        if ( !pFrame ) break;
        cvShowImage( "Example3", pFrame);
        char c = cvWaitKey( 33 );
            // wait for 33ms
        if( c == 27 )break; //press "ESC" to break the loop
    }
    cvReleaseCapture( &_pCapture );
    cvDestroyWindow( "Example3" );

    return 0;
}
