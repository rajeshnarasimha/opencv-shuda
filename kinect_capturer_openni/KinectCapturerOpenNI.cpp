//https://groups.google.com/forum/?fromgroups=#!topic/openni-dev/uK4G7m5TFjI

/**
* @file main.cpp
* @brief This code is used to capture rgb images and ir (infrayed) image simultaneously using a kinect device.
* - The dependency includes opencv and openni.
* - ussage: " press c to capture; q to exist. s to switch \n"
* - note that RGB and IR image cannot be displayed simultaneously.
* - Only the IR image is shown, if all the centers of circles are detected the markers will shown, then the user may
*   press c to capture both the ir and rgb image simultaneously.
* - the output images will be stored in the local folder.
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.0
* @date 2012-06-08
*/
#define  INFO

#include <iostream>
#include <string>
#include <vector>

#include "Converters.hpp"
#include "Kinect.h"
//using namespace std;

#define CHECK_RC(rc, what)	\
	BTL_ASSERT(rc == XN_STATUS_OK, (what + std::string(xnGetStatusString(rc))) )

//opencv
#include <opencv/cv.h>
#include <opencv/highgui.h>
//openni
#include <XnCppWrapper.h>
#include <XnOpenNI.h> 
#include "KinectCapturerOpenNI.h"

CKinectCapturer::CKinectCapturer(){
	XnStatus nRetVal = XN_STATUS_OK; 
	XnMapOutputMode sModeVGA; 
	sModeVGA.nFPS = 30; 
	sModeVGA.nXRes = 640; 
	sModeVGA.nYRes = 480; 
	_vis = IR;
	//_cContext inizialization 
	nRetVal = _cContext.Init(); CHECK_RC(nRetVal, "Initialize _cContext"); 
	//depth node creation 
	nRetVal = _cDepthGen.Create(_cContext); CHECK_RC(nRetVal, "Create depth generator fail"); 
	//nRetVal = _cDepthGen.StartGenerating(); CHECK_RC(nRetVal, "Start generating Depth fail"); 
	//RGB node creation 
	nRetVal = _cRGBGen.Create(_cContext);			CHECK_RC(nRetVal, "Create rgb generator fail"); 
	nRetVal = _cRGBGen.SetMapOutputMode(sModeVGA); 	CHECK_RC(nRetVal, "Depth SetMapOutputMode XRes for 240, YRes for 320 and FPS for 30"); 
	//nRetVal = _cRGBGen.StartGenerating();			CHECK_RC(nRetVal, "Start generating RGB"); 
	//IR node creation 
	nRetVal = _cIRGen.Create(_cContext);			CHECK_RC(nRetVal, "Create ir generator"); 
	nRetVal = _cIRGen.SetMapOutputMode(sModeVGA);	CHECK_RC(nRetVal, "Depth SetMapOutputMode XRes for 640, YRes for 480 and FPS for 30"); 
	nRetVal = _cIRGen.StartGenerating();			CHECK_RC(nRetVal, "Start generating IR"); 

	_cvmIR.create(480,640,CV_8UC1);
	_cvmRGB.create(480,640,CV_8UC3);
	_uCapturedViews = 0;
}
void CKinectCapturer::mainLoop()
{
	cv::namedWindow("IR", cv::WINDOW_AUTOSIZE );

	bool bRun = true;
	while(bRun){ 
		if(_vis == RGB){
			nRetVal = _cContext.WaitOneUpdateAll(_cRGBGen); 
		}else if(_vis == IR){
			nRetVal = _cContext.WaitOneUpdateAll(_cIRGen); 
		}
		if (nRetVal == XN_STATUS_OK) { 
			switch(_vis){
			case IR:
				getNextFrameIR(_cvmIR);
				cv::imshow("IR",_cvmIR);
				break;
			case RGB:
				getNextFrameRGB(_cvmRGB);
				cv::imshow("IR",_cvmRGB);
			}

			char c = cv::waitKey( 33 );
			switch(c){ 
			case 'c': 
				if(_vis == IR){
					storeIR();
					switchRGBIR();
					storeRGB();
				}else if (_vis == RGB ){
					storeRGB();
					switchRGBIR();
					storeIR();
				}
				_uCapturedViews++; 

				break; 
			case 's':
				switchRGBIR();
				break;
			case 27: 
				bRun = false;
				break; //press "ESC" to break the loop
			} 
		} 
		else{ 
			printf("Failed updating data: %s\n", xnGetStatusString(nRetVal)); 
		} 
	} 
	cvDestroyWindow( "IR"); 
	_cContext.Release(); 
}


void CKinectCapturer::switchRGBIR(){ 
	switch(_vis){ 
	case RGB: 
		_vis = IR; 
		_cRGBGen.StopGenerating(); 
		_cIRGen.StartGenerating(); 
		break; 
	case IR: 
		_vis = RGB; 
		_cIRGen.StopGenerating(); 
		_cRGBGen.StartGenerating(); 
		break; 
	} 
} 
void CKinectCapturer::getNextFrameIR(cv::Mat& cvmIR_){

	const XnIRPixel* _pIRMap = _cIRGen.GetIRMap();

	ushort max = 0; 
	for(int i = 0; i < 640*480; ++i){ 
		if(_pIRMap[i] > max){ 
			max = _pIRMap[i]; 
		} 
	} 

	for(int i = 0; i < 640*480; ++i){ 
		tempIR2[i] = (uchar)((double)_pIRMap[i]/max*256); 
	} 

	cv::Mat cvmIR(480,640,CV_8UC1,(uchar*)tempIR2);
	cvmIR.copyTo(cvmIR_);
}

void CKinectCapturer::getNextFrameRGB(cv::Mat& cvmRGB_){
	const XnRGB24Pixel* _pRGBMap = _cRGBGen.GetRGB24ImageMap();
	cv::Mat cvmRGB(480,640,CV_8UC3,(uchar*)_pRGBMap);
	cv::cvtColor(cvmRGB,cvmRGB_, CV_BGR2RGB);
}


void CKinectCapturer::storeIR(){ 
	char filename[100]; 
	sprintf(filename, "IR%04d.tif", _uCapturedViews); 
	printf("%s\n", filename); 

	getNextFrameIR(_cvmIR);
	cv::imwrite(filename,_cvmIR);
} 

void CKinectCapturer::storeRGB(){ 
	char filename[100]; 
	sprintf(filename, "RGB%04d.tif", _uCapturedViews); 
	printf("%s\n", filename); 

	getNextFrameRGB(_cvmRGB);
	cv::imwrite(filename,_cvmRGB);
} 




int main(int argc, char* argv[]){ 

	CKinectCapturer cKC;

	cKC.mainLoop();
	
	return 0; 
} 

