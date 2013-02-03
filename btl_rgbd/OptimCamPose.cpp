#define INFO
#include "Converters.hpp"
#include <math.h>
#include <vector>

#include <algorithm>
#include "Optim.hpp"
#include "OptimCamPose.h"

bool btl::utility::COptimCamPose::isOK(){
	PRINTSTR( "isOK()" );
	//set row vector
	_cvmX.create(1,6);
	_cvmX << 0,0,0,0,0,0;
	//_cvmX.at<double>(0,0) = 0;//algha
	//_cvmX.at<double>(0,1) = 0;//beta
	//_cvmX.at<double>(0,2) = 0;//gama
	//_cvmX.at<double>(0,3) = 0;//x
	//_cvmX.at<double>(0,4) = 0;//y
	//_cvmX.at<double>(0,5) = 0;//z

	_cvmDelta.create(_cvmX.size());
	_cvmDelta.setTo( 0.0001 );

	_vCosts.clear();
	_vcvmXs.clear();
	_vcvmGs.clear();
	//set data

	return true;
}
cv::Mat btl::utility::COptimCamPose::setSE3( const cv::Mat& cvmR_, const cv::Mat& cvmT_ ){
	cv::Mat cvmSE3(4,4,CV_64FC1); 
	cvmSE3 = cv::Mat::eye(4,4,CV_64FC1);
	//set SE3
	cv::Mat cvmR = cvmR_.t();
	double* pSE3 = (double*) cvmSE3.data;
	const double* pR = ( const double*) cvmR.data;
	const double* pT = ( const double*) cvmT_.data;
	for (int r =0; r<4; r++){
		for (int c =0; c<3; c++){
			//assign pR
			if (r <3 ){
				*pSE3++ = *pR++;
			}// if r < 3
			else {
				*pSE3++ = *pT++;
			}// r==3
		}//for each col
		*pSE3++;
	}//for each row of SE3
	return cvmSE3;
}

void btl::utility::COptimCamPose::getRT(Eigen::Matrix3d* peimR_, Eigen::Vector3d* peivT_){
	//read R and T from _cvmX
	cv::Mat_<double> cvmR,cvmT;
	cv::Rodrigues(_cvmX.colRange(0,3),cvmR);
	*peimR_ << cvmR;
	cvmT = _cvmX.colRange(3,6).t(); 
	*peivT_ << cvmT;
	PRINT(cvmR);
}
double btl::utility::COptimCamPose::Func( const cv::Mat_<double>& cvmX_ )
{
	//read R and T from _cvmX
	cv::Mat cvmR,cvmT;
	cv::Rodrigues(cvmX_.colRange(0,3),cvmR);
	cvmT = _cvmX.colRange(3,6); 

	cv::Mat cvmSE3 = setSE3(cvmR,cvmT);//transform ref->world
	double dE = 0;
	for (int c=0; c<_cvmPlaneCur.cols; c++) {
		cv::Mat_<double> cvmE = _cvmPlaneRef.col(c) - cvmSE3 * _cvmPlaneCur.col(c);//SE3*q_cur
		
		dE += _cvmPlaneWeight(0,c)*(10*( fabs(cvmE.at<double>(0,0)) + fabs(cvmE.at<double>(1,0)) + fabs(cvmE.at<double>(2,0)) ) + fabs(cvmE.at<double>(3,0)));
	}

	return dE;
}

void btl::utility::COptimCamPose::dFunc(const cv::Mat_<double>& cvmX_, cv::Mat_<double>& cvmG_ )
{
	CHECK( 1 == cvmX_.rows, "Optim::Func() X must be a row vector.\n" ); 
	CHECK( 1 == cvmG_.rows, "Optim::Func() G must be a row vector.\n" ); 

	cv::Mat_<double> cvmX0 = cvmX_;

	if (cvmX0.rows == 0)
	{
		cvmG_.release();
		return;
	}	

	CHECK( cvmX_.size() == _cvmDelta.size(),  "m_vDelta is set incorrectly." );

	// calculate the gradient using finite difference
	// G(X) = F(X+dX) - F(X-dX)
	for(int i=0; i<cvmX0.cols; i++){ //X0 is column vector
		cvmX0.at<double>(0,i) = cvmX_.at<double>(0,i) + _cvmDelta.at<double>(0,i);
		cvmG_.at<double>(0,i)  = Func(cvmX0);
		cvmX0.at<double>(0,i) = cvmX_.at<double>(0,i) - _cvmDelta.at<double>(0,i);
		cvmG_.at<double>(0,i) -= Func(cvmX0);
		cvmX0.at<double>(0,i) = cvmX_.at<double>(0,i);
	}
}
