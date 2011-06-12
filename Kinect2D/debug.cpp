//display kinect depth in real-time
#include <iostream>
#include <string>
#include <vector>
#include <btl/Utility/Converters.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <algorithm>
#include <utility>
#include "keyframe.hpp"
//camera calibration from a sequence of images

using namespace btl; //for "<<" operator
using namespace utility;

//using namespace extra;
//using namespace videosource;

using namespace Eigen;
using namespace cv;

//class CKinectView;
//class KeyPoint;

SKeyFrame _sCurrentKF;
SKeyFrame _sPreviousKF;

Eigen::Matrix3d _mRAccu; //Accumulated Rotation
Eigen::Vector3d _vTAccu; //Accumulated Translation	
Eigen::Matrix3d _mRx;

bool _bContinuous = true;
bool _bPrevStatus = true;


int main ( int argc, char** argv )
{
    try
    {
		_sPreviousKF.loadfXML("0");
		_sCurrentKF .loadfXML("1");
		_sPreviousKF.detectCorners();
		_bPrevStatus = _sCurrentKF.detectOpticFlowAndRT( _sPreviousKF );

		vector< cv::Point2f >::const_iterator cit_Curr = _sCurrentKF._vCorners.begin();
		for( vector< cv::Point2f >::const_iterator cit = _sPreviousKF._vCorners.begin(); cit != _sPreviousKF._vCorners.end(); cit++, cit_Curr++ )
		{
			cv::circle( _sCurrentKF._cvmRGB, cv::Point( cit->x, cit->y), 1, cv::Scalar( 0, 0, 255, 0), 1, 4 );
			//cv::circle( _sCurrentKF._cvmRGB, cv::Point( cit_Curr->x, cit_Curr->y), 1, cv::Scalar( 0, 255, 0, 0), 1, 4 );
			cv::line( _sCurrentKF._cvmRGB, cv::Point( cit->x, cit->y), cv::Point( cit_Curr->x, cit_Curr->y ), cv::Scalar( 255, 255, 0, 0), 1 );
		}

        cv::namedWindow ( "myWindow", 1 );
        while ( true )
        {
            cv::imshow ( "myWindow", _sCurrentKF._cvmRGB );

            if ( cv::waitKey ( 30 ) >= 0 )
            {
                break;
            }
        }


    }
    catch ( CError& e )
    {
        if ( string const* mi = boost::get_error_info< CErrorInfo > ( e ) )
        {
            std::cerr << "Error Info: " << *mi << std::endl;
        }
    }

    return 0;
}


