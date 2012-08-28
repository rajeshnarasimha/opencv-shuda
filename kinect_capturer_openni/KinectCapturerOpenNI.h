/**
* @file main.cpp
* @brief This code is used to storeIR rgb images and ir (infrayed) image simultaneously using a kinect device.
* - The dependency includes opencv and openni.
* - ussage: " press c to storeIR; q to exist. s to switch \n"
* - note that RGB and IR image cannot be displayed simultaneously.
* - Only the IR image is shown, if all the centers of circles are detected the markers will shown, then the user may
*   press c to storeIR both the ir and rgb image simultaneously.
* - the output images will be stored in the local folder.
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.0
* @date 2012-06-08
*/
// source:
// https://groups.google.com/forum/?fromgroups=#!topic/openni-dev/uK4G7m5TFjI


class CKinectCapturer{
	enum Visualization{ RGB, IR	}; 
//openni
	xn::Context _cContext; 
	xn::DepthGenerator _cDepthGen; 
	xn::ImageGenerator _cRGBGen; 
	xn::IRGenerator _cIRGen; 
	XnStatus nRetVal;
//images 
	cv::Mat _cvmIR;
	cv::Mat _cvmRGB;
//auxiliary buffers and images 
	uchar  tempIR2[640*480]; 
	ushort _uCapturedViews;  
	Visualization _vis ; 

//method
public:
	CKinectCapturer();
	void mainLoop();
	void switchRGBIR();
	void getNextFrameIR(cv::Mat& cvmIR_);
	void getNextFrameRGB(cv::Mat& cvmRGB_);
	void storeIR();
	void storeRGB();
};//CKinectCapturer;
