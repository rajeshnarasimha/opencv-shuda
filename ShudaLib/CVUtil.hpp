#ifndef BTL_CV_UTILITY_HELPER
#define BTL_CV_UTILITY_HELPER

//opencv-based helpers

#include "OtherUtil.hpp"

#define CV_SSE2 1
#include <opencv/highgui.h>
#include <opencv/cv.h>

namespace btl
{
namespace utility
{

//used by freenect depth images
template <class T>
T rawDepthToMetersLinear ( int nRawDepth_, const cv::Mat_< T >& mPara_ = cv::Mat_< T >() )
{
    double k1 = -0.002788688001059727;
    double k2 = 3.330949940125644;

    if ( !mPara_.empty() )
    {
        k1 = mPara_.template at< T > ( 0, 0 );
        k2 = mPara_.template at< T > ( 1, 0 );
    }

    if ( nRawDepth_ < 2047 )
    {
        T tDepth = T ( 1.0 / ( T ( nRawDepth_ ) * k1 + k2 ) );
        tDepth = tDepth > 0 ? tDepth : 0;
        return tDepth;
    }

    return 0;
}
//used by freenect depth images
template <class T>
T rawDepthToMetersTanh ( int nRawDepth_, const cv::Mat_< T >& mPara_ = cv::Mat_< T >() )
{
    double k1 = 1.1863;
    double k2 = 2842.5;
    double k3 = 0.1236;

    if ( !mPara_.empty() )
    {
        k1 = mPara_.template at< T > ( 0, 0 );
        k2 = mPara_.template at< T > ( 1, 0 );
        k3 = mPara_.template at< T > ( 2, 0 );
    }

    //PRINT( nRawDepth_ );

    double depth = nRawDepth_;

    if ( depth < 5047 )
    {
        depth = k3 * tan ( depth / k2 + k1 );
        //PRINT( depth );
    }
    else
    {
        depth = 0;
    }

    return T ( depth );
}

//used by freenect depth images
template< class T >
T rawDepth ( int nX_, int nY_, const cv::Mat& cvmDepth_ )
{
    unsigned char* pDepth = ( unsigned char* ) cvmDepth_.data;
    pDepth += ( nY_ * cvmDepth_.cols + nX_ ) * 3;
    int nR = * ( pDepth );
    int nG = * ( pDepth + 1 );
    int nB = * ( pDepth + 2 );
    T nRawDepth = nR * 256 + nG;
    /*
    	PRINT( nR );
    	PRINT( nG );
    	PRINT( nB );
    	PRINT( nRawDepth );
    */
    return nRawDepth;
}

template< class T >
cv::Mat_< T > getColor ( int nX_, int nY_, const cv::Mat& cvmImg_ )
{
    unsigned char* pDepth = cvmImg_.data;
    pDepth += ( nY_ * cvmImg_.cols + nX_ ) * 3;
    T nR = * ( pDepth );
    T nG = * ( pDepth + 1 );
    T nB = * ( pDepth + 2 );

    cv::Mat_< T > rgb = ( cv::Mat_< T > ( 3, 1 ) << nR, nG, nB );

    /*
    	PRINT( nR );
    	PRINT( nG );
    	PRINT( nB );
    	PRINT( rgb );
    */
    return rgb;
}

template< class T >
T* getColorPtr ( const short& nX_, const short& nY_, const cv::Mat& cvmImg_ )
{
    if ( nX_ < 0 || nX_ >= ( short ) cvmImg_.cols || nY_ < 0 || nY_ >= ( short ) cvmImg_.rows )
    {
        return ( T* ) NULL;
    }

    unsigned char* pDepth = cvmImg_.data  + ( nY_ * cvmImg_.cols + nX_ ) * 3;
    return ( T* ) pDepth;
}



//used by freenect depth images
template < class T >
T depthInMeters ( int nX_, int nY_, const cv::Mat& cvmDepth_, const cv::Mat_< T >& mPara_ = cv::Mat_< T >(), const int nMethodType_ = 0 )
{
    int nRawDepth = rawDepth <int> ( nX_, nY_, cvmDepth_ );
    T tDepth;

    switch ( nMethodType_ )
    {
    case 0:
        tDepth = T ( rawDepthToMetersLinear< T > ( nRawDepth, mPara_ ) );
        break;
    case 1:
        tDepth = T ( rawDepthToMetersTanh< T > ( nRawDepth, mPara_ ) );
        break;
    default:
        tDepth = T ( rawDepthToMetersLinear< T > ( nRawDepth, mPara_ ) );
    }

    return tDepth;
}

template < class T >
Eigen::Matrix< T, 2, 1 > distortPoint ( const Eigen::Matrix< T, 2, 1 >& eivUndistorted_, const cv::Mat_< T >& cvmK_, const cv::Mat_< T >& cvmInvK_, const cv::Mat_< T >& cvmDistCoeffs_ )
{
    double xu = eivUndistorted_ ( 0 );
    double yu = eivUndistorted_ ( 1 );
    double xun = cvmInvK_ ( 0, 0 ) * xu + cvmInvK_ ( 0, 1 ) * yu + cvmInvK_ ( 0, 2 );
    double yun = cvmInvK_ ( 1, 0 ) * xu + cvmInvK_ ( 1, 1 ) * yu + cvmInvK_ ( 1, 2 );
    double x2 = xun * xun;
    double y2 = yun * yun;
    double xy = xun * yun;
    double r2 = x2 + y2;
    double r4 = r2 * r2;
    double r6 = r4 * r2;
    double k1 = cvmDistCoeffs_ ( 0 );
    double k2 = cvmDistCoeffs_ ( 1 );
    double k3 = cvmDistCoeffs_ ( 2 );
    double k4 = cvmDistCoeffs_ ( 3 );
    double k5 = cvmDistCoeffs_ ( 4 );
    double dRadialDistortion ( 1.0 + k1 * r2 + k2 * r4 + k5 * r6 );
    double dTangentialDistortionX = ( 2 * k3 * xy ) + ( k4 * ( r2 + 2 * x2 ) );
    double dTangentialDistortionY = ( k3 * ( r2 + 2 * y2 ) ) + ( 2 * k4 * xy );
    double xdn = ( xun * dRadialDistortion ) + dTangentialDistortionX;
    double ydn = ( yun * dRadialDistortion ) + dTangentialDistortionY;
    double xd = cvmK_ ( 0, 0 ) * xdn + cvmK_ ( 0, 1 ) * ydn + cvmK_ ( 0, 2 );
    double yd = cvmK_ ( 1, 0 ) * xdn + cvmK_ ( 1, 1 ) * ydn + cvmK_ ( 1, 2 );
    Eigen::Vector2d distorted ( xd, yd );
    return distorted;
}

template < class T >
void map4UndistortImage ( const Eigen::Vector2i& eivImageSize_, const cv::Mat_< T >& cvmK_, const cv::Mat_< T >& cvmInvK_, const cv::Mat_< T >& cvmDistCoeffs_, cv::Mat* pMapXY )
{
    pMapXY->create ( eivImageSize_ ( 1 ), eivImageSize_ ( 0 ), CV_16SC2 );
    short* pData = ( short* ) pMapXY->data;
    cv::Mat_<short> mapX, mapY;
//    mapX = cv::Mat_<float> ( cvmImage_.size() );
//    mapY = cv::Mat_<float> ( cvmImage_.size() );
    int nIdx = 0;

    for ( int y = 0; y < eivImageSize_ ( 1 ); ++y )
    {
        for ( int x = 0; x < eivImageSize_ ( 0 ); ++x )
        {
            Eigen::Matrix< T, 2, 1> undistorted ( x, y );
            Eigen::Matrix< T, 2, 1> distorted = distortPoint< T > ( undistorted, cvmK_, cvmInvK_, cvmDistCoeffs_ );
            pData [nIdx  ] = short ( distorted ( 0 ) + 0.5 );
            pData [nIdx+1] = short ( distorted ( 1 ) + 0.5 );
            nIdx += 2;
            //mapX[y][x] = ( float ) distorted ( 0 );
            //mapY[y][x] = ( float ) distorted ( 1 );
        }
    }

    return;
}

template < class T >
void undistortImage ( const cv::Mat& cvmImage_,  const cv::Mat_< T >& cvmK_, const cv::Mat_< T >& cvmInvK_, const cv::Mat_< T >& cvmDistCoeffs_, cv::Mat* pUndistorted_ )
{
    //CHECK( cvmImage_.size() == pUndistorted_->size(), "the size of all images must be the same. \n" );
    cv::Mat mapXY ( cvmImage_.size(), CV_16SC2 );
    short* pData = ( short* ) mapXY.data;
//    cv::Mat_<float> mapX, mapY;
//    mapX = cv::Mat_<float> ( cvmImage_.size() );
//    mapY = cv::Mat_<float> ( cvmImage_.size() );
    int nIdx = 0;

    for ( int y = 0; y < cvmImage_.rows; ++y )
    {
        for ( int x = 0; x < cvmImage_.cols; ++x )
        {
            Eigen::Matrix< T, 2, 1> undistorted ( x, y );
            Eigen::Matrix< T, 2, 1> distorted = distortPoint< T > ( undistorted, cvmK_, cvmInvK_, cvmDistCoeffs_ );
            pData [nIdx  ] = short ( distorted ( 0 ) + 0.5 );
            pData [nIdx+1] = short ( distorted ( 1 ) + 0.5 );
            nIdx += 2;
            //mapX[y][x] = ( float ) distorted ( 0 );
            //mapY[y][x] = ( float ) distorted ( 1 );
        }
    }

//    cout << " undistortImage() " << endl << flush;
    cv::remap ( cvmImage_, *pUndistorted_, mapXY, cv::Mat(), cv::INTER_NEAREST, cv::BORDER_CONSTANT );
//	cout << " after undistortImage() " << endl << flush;
    return;
}


template< class T >
T FindShiTomasiScoreAtPoint ( cv::Mat& img_, const int& nHalfBoxSize_ , const int& nX_, const int& nY_ )
{
	T dXX = 0;
	T dYY = 0;
	T dXY = 0;

	int nStartX = nX_ - nHalfBoxSize_;
	int nEndX   = nX_ + nHalfBoxSize_;
	int nStartY = nY_ - nHalfBoxSize_;
	int nEndY   = nY_ + nHalfBoxSize_;

	for ( int r = nStartY; r <= nEndY; r++ )
		for ( int c = nStartX; c <= nEndX; c++ )
		{
			T dx = img_.at< unsigned char > ( r, c + 1 ) - img_.at< unsigned char > ( r, c - 1 );
			T dy = img_.at< unsigned char > ( r + 1, c ) - img_.at< unsigned char > ( r - 1, c );
			dXX += dx * dx;
			dYY += dy * dy;
			dXY += dx * dy;
		}

		int nPixels = ( 2 * nHalfBoxSize_ + 1 ) * ( 2 * nHalfBoxSize_ + 1 );
		dXX = dXX / ( 2.0 * nPixels );
		dYY = dYY / ( 2.0 * nPixels );
		dXY = dXY / ( 2.0 * nPixels );
		// Find and return smaller eigenvalue:
		return 0.5 * ( dXX + dYY - sqrt ( ( dXX + dYY ) * ( dXX + dYY ) - 4 * ( dXX * dYY - dXY * dXY ) ) );
};

template< class T1, class T2>
void convert2DisparityDomain(const cv::Mat_<T1>& cvDepth_, cv::Mat_<T2>* pcvDisparity_)
{
	const T1* pInputDepth = (T1*)cvDepth_.data;
	T2* pOutputDisparity = (T2*)pcvDisparity_->data;
	for ( unsigned int y = 0; y < cvDepth_.rows; y++ )
	{
		for ( unsigned int x = 0; x < cvDepth_.cols; x++ )
		{
			*pOutputDisparity++ = 1./(*pInputDepth++);
		}
	}
	return;
}

template< class T1, class T2>
void convert2DepthDomain(const cv::Mat_<T1>& cvDepth_, cv::Mat_<T2>* pcvDisparity_)
{
	const T1* pInputDepth = (T1*)cvDepth_.data;
	T2* pOutputDisparity = (T2*)pcvDisparity_->data;
	for ( unsigned int y = 0; y < cvDepth_.rows; y++ )
	{
		for ( unsigned int x = 0; x < cvDepth_.cols; x++ )
		{
			*pOutputDisparity++ = (T2)(1./(*pInputDepth++)+.5);
		}
	}
	return;
}

template< class T >
void bilateralFiltering( const cv::Mat_<T>& cvmSrc_, double dSigmaSpace_, double dSigmaRange_, cv::Mat_<T>* pcvmDst_)
{
	unsigned int uSize = (unsigned int)(dSigmaSpace_+.5)*2;
	cv::Mat_<T> cmSpaceKernel(uSize,uSize);


	return;
}


template< class T >
void filterDepth ( const double& dThreshould_, const cv::Mat_ < T >& cvmDepth_, cv::Mat_< T >* pcvmDepthNew_ )
{
	//PRINT( dThreshould_ );
	pcvmDepthNew_->create ( cvmDepth_.size() );

	for ( int y = 0; y < cvmDepth_.rows; y++ )
		for ( int x = 0; x < cvmDepth_.cols; x++ )
		{
			pcvmDepthNew_->template at< T > ( y, x ) = 0;

			if ( x == 0 || x == cvmDepth_.cols - 1 || y == 0 || y == cvmDepth_.rows - 1 )
			{
				continue;
			}

			T c = cvmDepth_.template at< T > ( y, x   );
			T cl = cvmDepth_.template at< T > ( y, x - 1 );

			if ( fabs ( double( c - cl ) ) < dThreshould_ )
			{
				//PRINT( fabs( c-cl ) );
				T cr = cvmDepth_.template at< T > ( y, x + 1 );

				if ( fabs ( double( c - cr ) )< dThreshould_ )
				{
					T cu = cvmDepth_.template at< T > ( y - 1, x );

					if ( fabs ( double( c - cu ) ) < dThreshould_ )
					{
						T cb = cvmDepth_.template at< T > ( y + 1, x );

						if ( fabs ( double( c - cb ) ) < dThreshould_ )
						{
							T cul = cvmDepth_.template at< T > ( y - 1, x - 1 );

							if ( fabs ( double( c - cul ) ) < dThreshould_ )
							{
								T cur = cvmDepth_.template at< T > ( y - 1, x + 1 );

								if ( fabs ( double( c - cur ) ) < dThreshould_ )
								{
									T cbl = cvmDepth_.template at< T > ( y + 1, x - 1 );

									if ( fabs ( double( c - cbl ) ) < dThreshould_ )
									{
										T cbr = cvmDepth_.template at< T > ( y + 1, x + 1 );

										if ( fabs ( double( c - cbr ) ) < dThreshould_ )
										{
											pcvmDepthNew_ ->template at< T > ( y, x ) = c;
											//PRINT( y );
											//PRINT( x );
										}
									}
								}
							}
						}
					}
				}
			}
		}

		return;
}

template< class T>
T matNormL1 ( const cv::Mat_< T >& cvMat1_, const cv::Mat_< T >& cvMat2_ )
{
	return (T) cv::norm( cvMat1_ - cvMat2_, cv::NORM_L1 );
}

template< class T >
void gaussianKernel( double dSigmaSpace, unsigned int& uSize_, cv::Mat_<T>* pcvmKernel_ )
{

}


}//utility
}//btl
#endif