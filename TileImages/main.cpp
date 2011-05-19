/**
* @file main.cpp
* @brief tile 13 x 26 images/0_0.jpg 0_1.jpg ... into a single image panorama.jpg
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.0
* @date 2011-01-21
*/

#include <opencv/highgui.h>
#include <iostream>
#include <iomanip> // for setw()
#include <string>
#include <boost/lexical_cast.hpp>

#define ROWS 13
#define COLS 26

using boost::lexical_cast;
using boost::bad_lexical_cast;
using namespace std;

int main ( int argc, char** argv )
{
    IplImage* pImgLarge = cvCreateImage ( cvSize ( 512 * COLS, 512 * ROWS ), 8, 3 );

    for ( int r = 0; r < ROWS; r++ )
    {
        for ( int c = 0; c < COLS; c++ )
        {
            string sFileName ;
            string sC = lexical_cast<string> ( c );
            string sR = lexical_cast<string> ( r );
            sFileName = "images/" + sC + "_" + sR + ".jpg";
            cout << sFileName << endl;
            IplImage* pImg = cvLoadImage ( sFileName.c_str() );
            
            cvSetImageROI ( pImgLarge, cvRect ( 0 + c*512, 0 + r*512, 512, 512 ) );
            cvCopy( pImg, pImgLarge );
            cvResetImageROI( pImgLarge );
            cvReleaseImage ( &pImg );
        }
    }
    
    // for details please refer to Learning OpenCV Page 42 - 45
    cout << " nChannels = " << setw ( 10 ) << pImgLarge->nChannels  <<  " depth         = " << setw ( 10 ) << pImgLarge->depth        << endl\
         << " nSize     = " << setw ( 10 ) << pImgLarge->nSize      <<  " alaphaChannel = " << setw ( 10 ) << pImgLarge->alphaChannel << endl\
         << " colorModel= " << setw ( 10 ) << pImgLarge->colorModel <<  " dataOrder     = " << setw ( 10 ) << pImgLarge->dataOrder    << endl\
         << " origin    = " << setw ( 10 ) << pImgLarge->origin     <<  " align         = " << setw ( 10 ) << pImgLarge->align        << endl\
         << " imageSize = " << setw ( 10 ) << pImgLarge->imageSize  <<  " widthStep     = " << setw ( 10 ) << pImgLarge->widthStep    << endl\
         << " width     = " << setw ( 10 ) << pImgLarge->width      <<  " height        = " << setw ( 10 ) << pImgLarge->height       << endl;

    cvNamedWindow( "Example1", CV_WINDOW_AUTOSIZE );
    cvShowImage( "Example1", pImgLarge );
    cvWaitKey(0);
    cvSaveImage( "panorama.jpg", pImgLarge );
    cvReleaseImage( &pImgLarge );
    cvDestroyWindow( "Example1" );

    return 0;
}
