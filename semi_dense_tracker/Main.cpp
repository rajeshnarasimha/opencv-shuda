/**
* @file main.cpp
* @brief this code shows how to load video from a default webcam, do some simple image processing and save it into an
* 'avi' file
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.
* @date 2011-03-03
*/

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include <boost/shared_ptr.hpp>

#include "SemiDenseTracker.h"
#include "SemiDenseTrackerOrb.h"
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>


float _aLight[1024];

void initLight(){
	boost::mt19937 rng; // I don't seed it on purpouse (it's not relevant)
	boost::normal_distribution<> nd(0.0, 1.0);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);

	for (int i=0; i < 1024; ++i){
		_aLight[i] = float(abs(var_nor()));
	}
}

#include <math.h>

boost::shared_ptr<cv::gpu::GpuMat> _acvgmShrPtrPyrBWs[4]; 
void initPyramid(int r, int c){
	for (int n=0; n<4; n++){
		_acvgmShrPtrPyrBWs[n].reset(new cv::gpu::GpuMat(r,c,CV_8UC1));
		r /= 2; c /= 2;
	}
	return;
}
void buildPyramid(const cv::gpu::GpuMat& cvgmGray_){
	cvgmGray_.copyTo(*_acvgmShrPtrPyrBWs[0]);
	for (int n=0; n< 3; n++){
		cv::gpu::resize(*_acvgmShrPtrPyrBWs[n],*_acvgmShrPtrPyrBWs[n+1],cv::Size(0,0),.5f,.5f );	
	}
	return;
}

//#define  WEB_CAM
int main ( int argc, char** argv )
{
	initLight();
    //opencv cpp style
#ifdef WEB_CAM
	cv::VideoCapture cap ( 1 ); // 0: open the default camera
								// 1: open the integrated webcam
#else
	cv::VideoCapture cap("VBranches.avi"); //("VDark.avi");//("VTreeTrunk.avi"); //("VRotatePersp.avi");//("VMouth.avi");// ("VCars.avi"); //("VZoomIn.avi");//("VSelf.avi");//("VFootball.mkv");//( "VRectLight.avi" );
	//("VCars.avi"); //("VRotateOrtho.avi"); //("VHand.avi"); 
	//("VPerson.avi");//("VHall.avi");// // ("VZoomOut.avi");// 
#endif

    if ( !cap.isOpened() ) return -1;
	
	btl::image::semidense::CSemiDenseTrackerOrb cSDTOrb;
	btl::image::semidense::CSemiDenseTracker cSDTFast;
	cv::gpu::GpuMat cvgmColorFrame,cvgmGrayFrame,cvgmColorFrameSmall; 
	cv::Mat cvmColorFrame, cvmGrayFrame, cvmTotalFrame;
	cap >> cvmColorFrame; cvgmColorFrame.upload(cvmColorFrame);
	//resize
	const float fScale = 1.f;
	cv::gpu::resize(cvgmColorFrame,cvgmColorFrameSmall,cv::Size(0,0),fScale ,fScale );	
	//to gray
	cv::gpu::cvtColor(cvgmColorFrameSmall,cvgmGrayFrame,cv::COLOR_RGB2GRAY);
	initPyramid(cvgmGrayFrame.rows, cvgmGrayFrame.cols );
	buildPyramid(cvgmGrayFrame);
	cvmTotalFrame.create(cvgmColorFrameSmall.rows*2,cvgmColorFrameSmall.cols*2,CV_8UC3);
	cv::Mat cvmROI0(cvmTotalFrame, cv::Rect(	     0,					        0,			    cvgmColorFrameSmall.cols, cvgmColorFrameSmall.rows));
	cv::Mat cvmROI1(cvmTotalFrame, cv::Rect(       0,  			   cvgmColorFrameSmall.rows, cvgmColorFrameSmall.cols, cvgmColorFrameSmall.rows));
	cv::Mat cvmROI2(cvmTotalFrame, cv::Rect(cvgmColorFrameSmall.cols, cvgmColorFrameSmall.rows, cvgmColorFrameSmall.cols, cvgmColorFrameSmall.rows));
	cv::Mat cvmROI3(cvmTotalFrame, cv::Rect(cvgmColorFrameSmall.cols,          0,              cvgmColorFrameSmall.cols, cvgmColorFrameSmall.rows));
	//copy to total frame
	cvgmColorFrameSmall.download(cvmROI0); cvgmColorFrameSmall.download(cvmROI1); cvgmColorFrameSmall.download(cvmROI2); cvgmColorFrameSmall.download(cvmROI3);

	bool bIsInitSuccessful;
	bIsInitSuccessful = cSDTFast.init( _acvgmShrPtrPyrBWs );
	bIsInitSuccessful = cSDTOrb.init( _acvgmShrPtrPyrBWs );

	while(!bIsInitSuccessful){
		cap >> cvmColorFrame; cvgmColorFrame.upload(cvmColorFrame);
		//resize
		cv::gpu::resize(cvgmColorFrame,cvgmColorFrameSmall,cv::Size(0,0),fScale ,fScale );
		//to gray
		cv::gpu::cvtColor(cvgmColorFrameSmall,cvgmGrayFrame,cv::COLOR_RGB2GRAY);
		//copy into total frame	
		cvgmColorFrameSmall.download(cvmROI0); cvgmColorFrameSmall.download(cvmROI1); cvgmColorFrameSmall.download(cvmROI2); cvgmColorFrameSmall.download(cvmROI3);
		bIsInitSuccessful = cSDTOrb.init( _acvgmShrPtrPyrBWs );
		bIsInitSuccessful = cSDTFast.init( _acvgmShrPtrPyrBWs );
	}

    cv::namedWindow ( "Tracker", 1 );
	bool bStart = false;
	unsigned int uIdx = 0;
    for ( ;;uIdx++ ){
		double t = (double)cv::getTickCount();
		int nKey = cv::waitKey( 0 ) ;
		if ( nKey == 'a' ){
			bStart = true;
		}
		else if ( nKey == 'q'){
			break;
		}
		

		imshow ( "Tracker", cvmTotalFrame );
		if(!bStart) continue;
		//load a new frame
 		cap >> cvmColorFrame; 

		if (cvmColorFrame.empty()) {
			cap.set(CV_CAP_PROP_POS_AVI_RATIO,0);//replay at the end of the video
			cap >> cvmColorFrame; cvgmColorFrame.upload(cvmColorFrame);
			//resize
			cv::gpu::resize(cvgmColorFrame,cvgmColorFrameSmall,cv::Size(0,0),fScale ,fScale );
			//to gray
			cv::gpu::cvtColor(cvgmColorFrameSmall,cvgmGrayFrame,cv::COLOR_RGB2GRAY);
			buildPyramid(cvgmGrayFrame);
			//copy into total frame	
			cvgmColorFrameSmall.download(cvmROI0); cvgmColorFrameSmall.download(cvmROI1); cvgmColorFrameSmall.download(cvmROI2); cvgmColorFrameSmall.download(cvmROI3);
			cSDTOrb.init( _acvgmShrPtrPyrBWs );
			cSDTFast.init( _acvgmShrPtrPyrBWs );
			//get second frame
			cap >> cvmColorFrame; cvgmColorFrame.upload(cvmColorFrame);
			//resize
			cv::gpu::resize(cvgmColorFrame,cvgmColorFrameSmall,cv::Size(0,0),fScale ,fScale );
			//to gray
			cv::gpu::cvtColor(cvgmColorFrameSmall,cvgmGrayFrame,cv::COLOR_RGB2GRAY);
			//copy into total frame	
			cvgmColorFrameSmall.download(cvmROI0); cvgmColorFrameSmall.download(cvmROI1); cvgmColorFrameSmall.download(cvmROI2); cvgmColorFrameSmall.download(cvmROI3);
		}else{
			/*cvgmColorFrame.upload(cvmColorFrame);
			cvgmColorFrame.convertTo(cvgmColorFrame,CV_8UC3,pow(0.9f,_aLight[uIdx%1024]));
			cvgmColorFrame.download(cvmColorFrame);*/

			cvgmColorFrame.upload(cvmColorFrame);
			//resize
			cv::gpu::resize(cvgmColorFrame,cvgmColorFrameSmall,cv::Size(0,0),fScale ,fScale );
			//to gray
			cv::gpu::cvtColor(cvgmColorFrameSmall,cvgmGrayFrame,cv::COLOR_RGB2GRAY);
			buildPyramid(cvgmGrayFrame);
			//copy into total frame	
			cvgmColorFrameSmall.download(cvmROI0); cvgmColorFrameSmall.download(cvmROI1); cvgmColorFrameSmall.download(cvmROI2); cvgmColorFrameSmall.download(cvmROI3);
		}

		
		cSDTFast.trackAll( _acvgmShrPtrPyrBWs );
		cSDTFast.displayCandidates( cvmROI3 );
 		cSDTFast.display(cvmROI2);
		
		cSDTOrb.trackAll( _acvgmShrPtrPyrBWs );
		cSDTOrb.display(cvmROI1);

		t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
		std::cout << "frame time [s]: " << t*1000 << " ms" << std::endl;	


		//interactions
        /*if ( cv::waitKey ( 30 ) >= 0 ){
            break;
        }*/
    }

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
