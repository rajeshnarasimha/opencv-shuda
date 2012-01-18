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
#define BTL_NUM_COLOR 15
	static unsigned char aColors[BTL_NUM_COLOR][3]=
	{
		{255, 0, 0, },//red
		{0, 255, 0,}, //green
		{0, 0, 255,}, //blue
		{255, 255, 255,}, //white
		{0, 0, 0,}, //black
		{255, 255, 0,}, //yellow
		{0, 255, 255,}, //cyan
		{255, 0, 255,}, //magenta
		{255,182,193,}, //light pink
		{139,131,134,},  //lavender blush
		{155,48,255,},	//purple
		{135,206,235,},  //sky blue
		{46,139,87,},   //sea green
		{255,215,0,},   //gold
		{255,165,0,},    //orange
	};
// set the matrix into a certain value
	/*
template< class T >
void clearMat(const T& tValue_, cv::Mat* pcvmMat_)
{
	typedef T Tp;
	cv::Mat& cvmMat_ = *pcvmMat_;
	BTL_ERROR(cvmMat_.channels()>1, "CVUtil::clearMat() only available for 1-channel cv::Mat" );
	BTL_ERROR(!cvmMat_.data, "CVUtil::clearMat() input cv::Mat is empty.");
	cv::MatIterator_<Tp> it;
	it = cvmMat_.begin<Tp>();

	for( ; it != cvmMat_.end<Tp>(); ++it )
	{
		*it = tValue_;
	}
	return;
}
*/
//calculate the L1 norm of two matrices, ( the sum of abs differences between corresponding elements of matrices )
template< class T>
T matNormL1 ( const cv::Mat& cvMat1_, const cv::Mat& cvMat2_ )
{
	return (T) cv::norm( cvMat1_ - cvMat2_, cv::NORM_L1 );
}
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

template< class T>
void convert2DisparityDomain(const cv::Mat_<T>& cvDepth_, cv::Mat* pcvDisparity_, T* ptMax_=NULL, T* ptMin_=NULL)
{
	BTL_ERROR(cvDepth_.channels()>1, "CVUtil::convert2DisparityDomain() only available for 1-channel cv::Mat" );
	BTL_ERROR(!cvDepth_.data, "CVUtil::convert2DisparityDomain() input cvDepth_ is empty.");

	cv::Mat& cvDisparity_ = *pcvDisparity_;
	cvDisparity_.create(cvDepth_.rows, cvDepth_.cols, CV_32FC1);
	const T* pInputDepth = (T*)cvDepth_.data;

	if( ptMax_ && ptMax_)
	{
		*ptMax_ = -BTL_MAX;
		*ptMin_ =  BTL_MAX;
	}

	for(cv::MatIterator_<float> it = cvDisparity_.begin<float>(); it != cvDisparity_.end<float>(); ++it, pInputDepth++ )
	{
			double dDepth = *pInputDepth;
			if( dDepth>SMALL )
			{
				*it = 1./dDepth;
				if( ptMax_ && ptMax_)
				{
					*ptMax_ = *it> *ptMax_?*it:*ptMax_;
					*ptMin_ = *it< *ptMin_?*it:*ptMin_;
				}
			}
			else
				*it = 0.;

	}
	return;
}

template< class T>
void convert2DepthDomain(const cv::Mat& cvDisparity_, cv::Mat* pcvDepth_, int nType_ ) // nType must be identical with T
{
	BTL_ERROR(cvDisparity_.channels()>1, "CVUtil::convert2DepthDomain() only available for 1-channel cvDisparity_" );
	BTL_ERROR(!cvDisparity_.data, "CVUtil::convert2DepthDomain() input cvDisparity_ is empty.");
	BTL_ERROR(cvDisparity_.type() != CV_32FC1, "CVUtil::convert2DepthDomain() input cvDisparity_ must be CV_32FC1 type.");

	cv::Mat& cvDepth_ = *pcvDepth_;
	cvDepth_.create(cvDisparity_.size(),nType_);
	//btl::utility::clearMat<T>(0,&cvDepth_);
	T* pDepth = (T*) cvDepth_.data;

	for(cv::MatConstIterator_<float> cit = cvDisparity_.begin<float>(); cit != cvDisparity_.end<float>(); ++cit, pDepth++ )
	{
		float fDepth = *cit;
		if( fDepth > SMALL )
			if( CV_16UC1 == nType_ ) *pDepth = (unsigned short)(1./fDepth + .5 );
			else *pDepth = 1.f/fDepth;
		else
			*pDepth = 0.;
	}
	
	return;
}

template< class T>
void bilateralFilterInDisparity(cv::Mat* pcvDepth_, double dSigmaDisparity_, double dSigmaSpace_ )
{
	BTL_ASSERT(pcvDepth_->channels()==1,"CVUtil::bilateralFilterInDisparity(): the input must be 1 channel depth map")
	cv::Mat& cvDepth_ = *pcvDepth_;
	cv::Mat cvDisparity, cvFilteredDisparity;

	btl::utility::convert2DisparityDomain< T >( cvDepth_, &cvDisparity );
	cv::bilateralFilter(cvDisparity, cvFilteredDisparity,0, dSigmaDisparity_, dSigmaSpace_); // filter size has to be an odd number.
	btl::utility::convert2DepthDomain< T >( cvFilteredDisparity,&cvDepth_, cvDepth_.type() );

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
	}//forfor

	return;
}

template< class T>
void gaussianC1FilterInDisparity(cv::Mat* pcvDepth_, double dSigmaDisparity_, double dSigmaSpace_ )
{
	BTL_ASSERT(pcvDepth_->channels()==1,"CVUtil::bilateralFilterInDisparity(): the input must be 1 channel depth map");
	BTL_ASSERT(pcvDepth_->type()==CV_32FC1,"CVUtil::bilateralFilterInDisparity(): the input must be CV_32FC1");

	cv::Mat& cvDepth_ = *pcvDepth_;
	cv::Mat cvDisparity, cvGaussianFiltered, cvC1Filtered;

	btl::utility::convert2DisparityDomain< T >( cvDepth_, &cvDisparity );
	cv::GaussianBlur(cvDisparity, cvGaussianFiltered, cv::Size(0,0), dSigmaSpace_, dSigmaSpace_); // filter size has to be an odd number.
	btl::utility::filterDepth <T> ( dSigmaDisparity_, ( cv::Mat_<T>)cvGaussianFiltered, ( cv::Mat_<T>*)&cvC1Filtered );
	btl::utility::convert2DepthDomain< T >( cvC1Filtered, &cvDepth_, cvDepth_.type() );

	return;
}


template< class T >
void downSampling( const cv::Mat& cvmOrigin_, cv::Mat* pcvmHalf_)
{
	BTL_ASSERT(1==cvmOrigin_.channels(),"downSampling() only down samples 1 channel cv::Mat.");
	cv::Mat& cvmHalf_ = *pcvmHalf_;
	cvmHalf_.create(cvmOrigin_.rows/2,cvmOrigin_.cols/2,cvmOrigin_.type());
		
	//btl::utility::clearMat<T>(0,&cvmHalf_);
	cvmHalf_.setTo(0);

	const T* pIn = (const T*)cvmOrigin_.data;
	T* pOut= (T*)cvmHalf_.data;
	int nIdx;
	for(int r = 0; r < cvmOrigin_.rows; r+=2)
	for(int c = 0; c < cvmOrigin_.cols; c+=2)
	{
		nIdx = r*cvmOrigin_.cols + c;
		*pOut++ = pIn[nIdx];
	}

	return;
}




}//utility
}//btl
#endif