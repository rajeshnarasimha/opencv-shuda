// open an avi file 
// ./Exe filename.avi

#include <opencv/highgui.h>


int main( int argc, char** argv )
{
    cvNamedWindow( "Example2", CV_WINDOW_AUTOSIZE );
    CvCapture* pCapture = cvCreateFileCapture( argv[1] ); 
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
    cvDestroyWindow( "Example2" );
}
