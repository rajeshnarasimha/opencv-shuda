//display kinect depth in real-time
#define INFO
#define TIMER
#include <GL/glew.h>
#include <gl/freeglut.h>
//#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>


#include <iostream>
#include <string>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "Utility.hpp"
#include "Converters.hpp"
using namespace btl::utility;
//camera calibration from a sequence of images
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

#include <XnCppWrapper.h>
#include <gl/freeglut.h>
#include "Kinect.h"
#include "Camera.h"
#include "EigenUtil.hpp"
#include "GLUtil.h"
#include "PlaneObj.h"
#include "Histogram.h"
#include "SemiDenseTracker.h"
#include "SemiDenseTrackerOrb.h"
#include "TrackerSimpleFreak.h"
#include "KeyFrame.h"
#include "CyclicBuffer.h"
#include "VideoSource.h"
//Qt
#include <QResizeEvent>
#include <QGLViewer/qglviewer.h>
#include "SoccerPitch.h"

#include "SimpleViewer.h"
#include <QCoreApplication>
#include "GroundTruth.h"

__device__ __host__ short2 operator + (const short2 s2O1_, const short2 s2O2_);
__device__ __host__ short2 operator - (const short2 s2O1_, const short2 s2O2_);
__device__ __host__ float2 operator * (const float fO1_, const short2 s2O2_);
__device__ __host__ short2 operator * (const short sO1_, const short2 s2O2_);
__device__ __host__ float2 operator + (const float2 f2O1_, const float2 f2O2_);
__device__ __host__ float2 operator - (const float2 f2O1_, const float2 f2O2_);
__device__  short2 convert2s2(const float2 f2O1_);

using namespace std;
Viewer::Viewer(){
	_uPyrHeight = 3; //this value will refreshed by loadFromYml()
	_uResolution = 0;//this value will refreshed by loadFromYml()
	_eivCw = Eigen::Vector3f(0.f,0.f,0.f);
	_bUseNIRegistration = true;
	_uCubicGridResolution = 512;
	_fVolumeSize = 3.f;
	_nMode = 3;//btl::kinect::VideoSourceKinect::PLAYING_BACK
	_oniFileName = std::string("x.oni"); // the openni file 
	_bRepeat = false;// repeatedly play the sequence 
	_nRecordingTimeInSecond = 30;
	_fTimeLeft = _nRecordingTimeInSecond;
	_nStatus = 01;//1 restart; 2 //recording continue 3://pause 4://dump
	_bDisplayImage = false;
	_bLightOn = false;
	_bRenderReference = false;
	_uFrameIdx = 0;
	_strVideoFileName = "fi-bu-uc.avi";
	//_strVideoFileName = "t2_c1.wmv";
}
Viewer::~Viewer()
{
	_pGL->destroyVBOsPBOs();
}

void Viewer::creatGroundTruthMask(cv::Mat* pcvmMask_){
	if( !pcvmMask_->empty() ){
		pcvmMask_->copyTo(_cvmMaskPrev);
		pcvmMask_->setTo(0);
	}
	else {
		pcvmMask_->create(_pVideo->_pCurrFrame->_acvmShrPtrPyrRGBs[0]->size(),CV_8UC1); 
		pcvmMask_->setTo(0);
	}

	cv::Point rook_points[1][4];
	for (int i=0; i<4; i++)	{
		_eivPixelH[i] = _eimHomoCurr*__eivTextureROIWorldHomo[i];
		_eivPixelH[i] /=_eivPixelH[i](2);
		rook_points[0][i].x = _eivPixelH[i](0)+.5f;
		rook_points[0][i].y = _eivPixelH[i](1)+.5f;
	}

	const cv::Point* ppt[1] = { rook_points[0] };
	int npt[] = { 4 };

	//how to use the fillPoly() 
	//http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/core/basic_geometric_drawing/basic_geometric_drawing.html
	cv::fillPoly(*pcvmMask_,ppt,npt,1,255);
	return;
}

void Viewer::other(){
	/*_pitchProcessor.extractFieldMask(*_pVideo->_pCurrFrame->_acvmShrPtrPyrRGBs[0], _cvmPlayField);
	cv::imshow("",_cvmPlayField);*/
	Eigen::Matrix3f eimHTmp;
	if (_pVideo->_uFrameIdx>0){
		const Eigen::Matrix3f& eimHomoInv = _veimHomography[_pVideo->_uFrameIdx];
		_eimHomoCurr = eimHomoInv.inverse();
		_eimHomoCurr /= _eimHomoCurr(2,2);
		creatGroundTruthMask(&_cvmMaskCurr);

		_pTracker->track(_pVideo->_pCurrFrame->_acvmShrPtrPyrBWs);
		cv::Mat cvmHomography = _pTracker->calcHomography(_cvmMaskCurr,_cvmMaskPrev);
		Eigen::Matrix3f eimH01; 
		if (!cvmHomography.empty()) {
			eimH01 << cvmHomography.ptr<double>(0)[0], cvmHomography.ptr<double>(0)[1], cvmHomography.ptr<double>(0)[2],
					  cvmHomography.ptr<double>(1)[0], cvmHomography.ptr<double>(1)[1], cvmHomography.ptr<double>(1)[2],
					  cvmHomography.ptr<double>(2)[0], cvmHomography.ptr<double>(2)[1], cvmHomography.ptr<double>(2)[2];
		}
		else {
			eimH01.setIdentity();
		}
		
		const Eigen::Matrix3f& eimHomoInv2 = _veimHomography[_pVideo->_uFrameIdx-1];
		Eigen::Matrix3f eimHomoInit = eimHomoInv2.inverse() * eimH01.inverse();
		eimHomoInit /= eimHomoInit(2,2);
		_eimHomoCurr = eimHomoInit;

		//_pTracker->track(eimHomoInit,_pVideo->_pCurrFrame->_acvgmShrPtrPyrBWs,_cvmMaskCurr,&_eimHomoCurr);
	}
	else {//first frame
		//warp the 
		creatGroundTruthMask(&_cvmMaskCurr);
		const Eigen::Matrix3f& eimHomoInv = _veimHomography[_pVideo->_uFrameIdx];
		_eimHomoCurr = eimHomoInv.inverse(); 
		_eimHomoCurr /= _eimHomoCurr(2,2);
		creatGroundTruthMask(&_cvmMaskCurr);
	    _pTracker->initialize(_pVideo->_pCurrFrame->_acvmShrPtrPyrBWs);
		//_pTracker->initialize(_pVideo->_pCurrFrame->_acvgmShrPtrPyrBWs,_cvmMaskCurr);//for full-frame tracking
	}
		
	//cv::imwrite("mask.png",cvmMask);
	_pVideo->_pCurrFrame->_acvgmShrPtrPyrRGBs[0]->upload(*_pVideo->_pCurrFrame->_acvmShrPtrPyrRGBs[0]);
	/*std::ostringstream strFileName;
	strFileName << "tmp"; 
	strFileName << _uFrameIdx;
	strFileName << ".png"; 
	cv::imwrite(strFileName.str(),*_pCurrFrame->_acvmShrPtrPyrRGBs[0]);*/
	
	//estimate the camera motion from homography
	//Zhang, Z. (1999). Flexible Camera Calibration by Viewing a Plane from Unknown Orientations. ICCV (Vol. 1, pp. 666¨C673). 
	//page 3 above section 3.2
	Eigen::Matrix3f eimK = _pVideo->_pCamera->getK();
	Eigen::Matrix3f eimKInv = eimK.inverse();

	Eigen::Vector3f eivV1 = eimKInv * _eimHomoCurr.col(0);
	Eigen::Vector3f eivV2 = eimKInv * _eimHomoCurr.col(1);

	float fL = (eivV1.norm()+eivV2.norm())/2.f;
	eivV1/=fL;//normalization
	eivV2/=fL;

	Eigen::Vector3f eivV3 = eivV1.cross(eivV2) ;
	
	Eigen::Matrix3f eimQ; 
	eimQ.col(0) = eivV1;
	eimQ.col(1) = eivV2;
	eimQ.col(2) = eivV3;
	//calc t
	Eigen::Vector3f eivT = eimKInv * _eimHomoCurr.col(2)/fL;

	//eimQ is not an strict rotation matrix, therefore we select the best approximated rotation matrix eimR of eimQ
	//the approach is adopted from Microsoft TR-98-71, appendix C
	Eigen::JacobiSVD<Eigen::Matrix3f> svd = eimQ.jacobiSvd( Eigen::ComputeFullU | Eigen::ComputeFullV );
	Eigen::Matrix3f eimR = svd.matrixU()*svd.matrixV().transpose();

	_pVideo->_pCurrFrame->setRTw(eimR,eivT);

	return;
}
void Viewer::reset(){
	
	string model_file = "temp.bg.xml";
	cv::Mat full_pitch = cv::imread("pitch.png");
	if(!_pitchProcessor.initBgFgModel(model_file)) {
		cout<<"could not find color model file " << model_file <<  ", create a default one ! \n"<<endl;
	};

	if(full_pitch.empty())
		printf("cannot load pitch model! \n");

	_pitchProcessor.initWireFrame(full_pitch);

	bool homography_found = false;
	cv::Mat frame, grayFrame, whitepixels;
	cv::Mat fit_Homography;
	cv::Mat cleaned_fg, smallFrame, full_play_field;
	cv::Mat fitted_img;
	cv::Point2f view_center(-1,-1), pre_view_center(-1,-1);
	bool pauseVideo = false;

	/*std::ostringstream strFileName;
	strFileName << "tmp"; 
	strFileName << _uFrameIdx;
	strFileName << ".png"; 
	cv::imwrite(strFileName.str(),*_pCurrFrame->_acvmShrPtrPyrRGBs[0]);*/

	importGroundTruth(_strVideoFileName + ".warps.yml");//"fi-bu-m1.avi.warps.yml");
	_eivCw = Eigen::Vector3f(.0f,.0f,2300.f);
	loadFromYml();
	_pVideo.reset(new btl::video::VideoSource("VisualTrackingDatasetCameraParam.yml",_uResolution,_uPyrHeight,_eivCw) );
	//_pVideo->setSize(0.5f);
	_pVideo->_strVideoFileName = _strVideoFileName;
	_pGL.reset( new btl::gl_util::CGLUtil( _uResolution,_uPyrHeight,btl::utility::BTL_GL,Eigen::Vector3f(.0f,.0f,0.f)) );
	_pGL->_bDisplayCamera = _bDisplayImage;
	_pGL->_bEnableLighting = _bLightOn;
	_pGL->_bRenderReference = _bRenderReference;
	_pGL->clearColorDepth();
	_pGL->init();
	_pGL->constructVBOsPBOs();
	//set opengl flags
	glDepthFunc  ( GL_LESS );
	glEnable     ( GL_DEPTH_TEST );
	glEnable 	 ( GL_SCISSOR_TEST );
	glDisable	 ( GL_CULL_FACE );
	glShadeModel ( GL_FLAT );
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	int nStatus;
	_pVideo->init();
	_pVideo->getNextFrame(&nStatus);
	/*_pPrevFrame.reset(new btl::kinect::CKeyFrame(_pVideo->_pCamera.get(),_uResolution,_uPyrHeight,_eivCw));
	_pVideo->_pCurrFrame->copyTo(&*_pPrevFrame);*/

	_pTracker.reset( new btl::image::CTrackerSimpleFreak(_uPyrHeight) );

	other();

	//Eigen::Matrix3f eimRotation; eimRotation = Eigen::AngleAxisf(float(M_PI), Eigen::Vector3f::UnitY());
	//_pVideo->_pCurrFrame->setRTFromC( eimRotation, _eivCw );
	_pVideo->_pCurrFrame->setView(&_pGL->_eimModelViewGL);

	return;
}
void Viewer::draw()
{
	_pVideo->getNextFrame(&nStatus);


	other();//intreprate ground truth data

	if(_bTrack) _pVideo->_pCurrFrame->setView(&_pGL->_eimModelViewGL);

	//set viewport
	glViewport (0, 0, width()/2, height());
	glScissor  (0, 0, width()/2, height());

	_pVideo->_pCamera->setGLProjectionMatrix(1,20.f,3000.f);
	glMatrixMode ( GL_MODELVIEW );

	// after set the intrinsics and extrinsic s
	// load the matrix to set camera pose
	_pGL->viewerGL();	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	//_pGL->renderRectangle<float>((float*)__eivTextureROI3D[0].data(),(float*)__eivTextureROI3D[1].data(),(float*)__eivTextureROI3D[2].data(),(float*)__eivTextureROI3D[3].data());
	//texture mapping
	GLuint uTexture;
	loadTexture(*_pVideo->_pCurrFrame->_acvmShrPtrPyrRGBs[_pGL->_usLevel], &uTexture);
	renderTexture( uTexture );

	if (_pGL->_bDisplayCamera)	{
		glColor3f(0.5f,0.5f,0.5f);
		_pVideo->_pCurrFrame->renderCameraInWorldCVCV(_pGL.get(),true,2000,0);
	}
	// render objects
	if (_pGL->_bRenderReference) drawAxis(200.f);
	_pGL->renderPatternGL(20.f,20,20);
	_pGL->renderPatternGL(200.f,10,10);
	_pGL->renderVoxelGL(_fVolumeSize);

	//render current frame

	//show text
	glColor3f(0.f,1.f,0.f);
	renderText(100,20,QString("FPS:")+QString::number(currentFPS()),QFont("Arial", 10, QFont::Normal));
	
	//set viewport 2
	glViewport (width()/2, 0, width()/2, height());
	glScissor  (width()/2, 0, width()/2, height());

	_pVideo->_pCamera->setGLProjectionMatrix(1,0.1f,100.f);
	glMatrixMode ( GL_MODELVIEW );
	glLoadIdentity();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	_pGL->renderAxisGL();
	//_pKinect->_pRGBCamera->LoadTexture(*_pKinect->_pFrame->_acvmShrPtrPyrRGBs[_pGL->_usLevel],&_pGL->_auTexture[_pGL->_usLevel]);
	_pGL->gpuMapRgb2PixelBufferObj(*_pVideo->_pCurrFrame->_acvgmShrPtrPyrRGBs[_pGL->_usLevel],_pGL->_usLevel);
	_pVideo->_pCamera->renderCameraInGLLocal(_pGL->_auTexture[_pGL->_usLevel], 0.2f );
	update();
}

void Viewer::loadTexture ( const cv::Mat& cvmImg_, GLuint* puTexture_ )
{
	glDeleteTextures( 1, puTexture_ );
	glGenTextures ( 1, puTexture_ );
	glBindTexture ( GL_TEXTURE_2D, *puTexture_ );
	glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
	glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
	glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ); // cheap scaling when image bigger than texture
	glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ); // cheap scaling when image smalled than texture  
	// 2d texture, level of detail 0 (normal), 3 components (red, green, blue), x size from image, y size from image,
	// border 0 (normal), rgb color data, unsigned byte data, and finally the data itself.
	if( 3 == cvmImg_.channels())
		glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGB, cvmImg_.cols, cvmImg_.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, cvmImg_.data ); //???????????????????
	else if( 1 == cvmImg_.channels())
		glTexImage2D ( GL_TEXTURE_2D, 0, GL_LUMINANCE, cvmImg_.cols, cvmImg_.rows, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, cvmImg_.data );
	//glTexEnvi ( GL_TEXTURE_2D, GL_TEXTURE_ENV_MODE, GL_REPEAT );
	// 2d texture, 3 colors, width, height, RGB in that order, byte data, and the data.
	//gluBuild2DMipmaps ( GL_TEXTURE_2D, GL_RGB, img.cols, img.rows,  GL_RGB, GL_UNSIGNED_BYTE, img.data );

	glBindTexture(GL_TEXTURE_2D, *puTexture_);

	if( 3 == cvmImg_.channels())
		glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, cvmImg_.cols, cvmImg_.rows, GL_RGB, GL_UNSIGNED_BYTE, cvmImg_.data);
	else if( 1 == cvmImg_.channels())
		glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, cvmImg_.cols, cvmImg_.rows, GL_LUMINANCE, GL_UNSIGNED_BYTE, cvmImg_.data);
	return;
}
void Viewer::renderTexture(const GLuint uTexture_)
{
	glEnable ( GL_TEXTURE_2D );
	glTexEnvf ( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE );
	glBindTexture ( GL_TEXTURE_2D, uTexture_ );

	glColor4f(1.f, 1.f, 1.f, 1.f); glLineWidth(.5);
	glBegin ( GL_QUADS );
	//convert to texture coordinate
	Eigen::Vector2f eivTexture[4];
	for (int n = 0; n < 4; n++)	{
		eivTexture[n](0) = _eivPixelH[n](0)/_pVideo->_pCurrFrame->_acvmShrPtrPyrRGBs[0]->cols;
		eivTexture[n](1) = _eivPixelH[n](1)/_pVideo->_pCurrFrame->_acvmShrPtrPyrRGBs[0]->rows;
	}

	glTexCoord2fv ( eivTexture[0].data() );
	glVertex3fv ( __eivTextureROI3D[0].data() );
	glTexCoord2fv ( eivTexture[1].data() );
	glVertex3fv ( __eivTextureROI3D[1].data() );
	glTexCoord2fv ( eivTexture[2].data() );
	glVertex3fv ( __eivTextureROI3D[2].data() );
	glTexCoord2fv ( eivTexture[3].data() );
	glVertex3fv ( __eivTextureROI3D[3].data() );
	glEnd();
	glDisable ( GL_TEXTURE_2D );
}

void Viewer::init()
{
	// Restore previous viewer state.
	restoreStateFromFile();
	resize(1280,480);
  
	// Opens help window
	//help();

	//
	GLenum eError = glewInit(); 
	if (GLEW_OK != eError){
		PRINTSTR("glewInit() error.");
		PRINT( glewGetErrorString(eError) );
	}
	btl::gl_util::CGLUtil::initCuda();
	btl::gl_util::CGLUtil::setCudaDeviceForGLInteroperation();//initialize before using any cuda component

	reset();//("fi-bu-m1.avi")
}//init()

QString Viewer::helpString() const
{
  QString text("<h2>S i m p l e V i e w e r</h2>");
  text += "Use the mouse to move the camera around the object. ";
  text += "You can respectively revolve around, zoom and translate with the three mouse buttons. ";
  text += "Left and middle buttons pressed together rotate around the camera view direction axis<br><br>";
  text += "Pressing <b>Alt</b> and one of the function keys (<b>F1</b>..<b>F12</b>) defines a camera keyFrame. ";
  text += "Simply press the function key again to restore it. Several keyFrames define a ";
  text += "camera path. Paths are saved when you quit the application and restored at next start.<br><br>";
  text += "Press <b>F</b> to display the frame rate, <b>A</b> for the world axis, ";
  text += "<b>Alt+Return</b> for full screen mode and <b>Control+S</b> to save a snapshot. ";
  text += "See the <b>Keyboard</b> tab in this window for a complete shortcut list.<br><br>";
  text += "Double clicks automates single click actions: A left button double click aligns the closer axis with the camera (if close enough). ";
  text += "A middle button double click fits the zoom of the camera and the right button re-centers the scene.<br><br>";
  text += "A left button double click while holding right button pressed defines the camera <i>Revolve Around Point</i>. ";
  text += "See the <b>Mouse</b> tab and the documentation web pages for details.<br><br>";
  text += "Press <b>Escape</b> to exit the viewer.";
  return text;
}

/*
void Viewer::resizeEvent( QResizeEvent * event )
{
	int nHeight = event->size().height();
	int nWidth  = event->size().width();
	
	float fAsp  = nHeight/float(nWidth);
	if( fabs(fAsp - 0.375f) > 0.0001f )//if the aspect ratio is not 3:4
	{
		int nUnit = nHeight/3;
		int nWidth= nUnit*8;
		int nHeight = nUnit*3;
		resize(nWidth,nHeight);
		update();
		/ *int nHeightO= event->oldSize().height();
		int nWidthO = event->oldSize().width();
		if (nHeight!=nHeightO && abs(nHeightO-nHeight)>3)
		{
			int nUnit = nHeight/3;
			int nWidth= nUnit*4;
			int nHeight = nUnit*3;
			resize(nWidth,nHeight);
			update();
		}else if (nWidth!=nWidthO && abs(nWidthO-nWidth)>4)
		{
			int nUnit = nWidth/4;
			int nWidth= nUnit*4;
			int nHeight = nUnit*3;
			resize(nWidth,nHeight);
			update();
		}* /
	}
	//and event->oldsize()
	QWidget::resizeEvent(event);
}*/

void Viewer::loadFromYml(){

#if __linux__
	cv::FileStorage cFSRead( "/space/csxsl/src/opencv-shuda/Data/kinect_intrinsics.yml", cv::FileStorage::READ );
#else if _WIN32 || _WIN64
	cv::FileStorage cFSRead ( "C:\\csxsl\\src\\opencv-shuda\\visual_tracking_dataset\\VisualTrackingDataset.yml", cv::FileStorage::READ );
#endif
	if( !cFSRead.isOpened() )
		PRINTSTR("VisualTrackingDataset load error.");

	cFSRead["uResolution"] >> _uResolution;
	cFSRead["uPyrHeight"] >> _uPyrHeight;
	cFSRead["bUseNIRegistration"] >> _bUseNIRegistration;
	cFSRead["uCubicGridResolution"] >> _uCubicGridResolution;
	cFSRead["fVolumeSize"] >> _fVolumeSize;
	//rendering
	cFSRead["bDisplayImage"] >> _bDisplayImage;
	cFSRead["bLightOn"] >> _bLightOn;
	cFSRead["bRenderReference"] >> _bRenderReference;
	cFSRead["nMode"] >> _nMode;//1 kinect; 2 recorder; 3 player
	cFSRead["oniFile"] >> _oniFileName;
	cFSRead["bRepeat"] >> _bRepeat;
	cFSRead["nRecordingTimeInSecond"] >> _nRecordingTimeInSecond;
	cFSRead["nStatus"] >> _nStatus;

	cFSRead.release();
}

void Viewer::exportToYml(const std::string& strFileName_){
	cv::FileStorage cFSWrite ( strFileName_, cv::FileStorage::WRITE );

	cv::Mat cvmTest; cvmTest.create(100, 9, CV_32FC1);
	cvmTest.at<float>(0,0) = 0.f;
	cvmTest.at<float>(0,1) = 1.f;
	cvmTest.at<float>(0,2) = 2.f;
	cvmTest.at<float>(0,3) = 3.f;
	cvmTest.at<float>(0,4) = 4.f;
	cFSWrite << "uPyrHeight" << cvmTest;
	cFSWrite.release();
}

void Viewer::importGroundTruth( const std::string& strFileName_ ){

	cv::FileStorage cFSRead( strFileName_, cv::FileStorage::READ );
	
	cFSRead["cvmHomography"] >> _cvmHomographyAll;
	cFSRead.release();

	Eigen::Matrix3f eimHomo; 
	for (int r = 0; r < _cvmHomographyAll.rows; r++)	{
		cv::Mat cvmTmp( 3,3,CV_32FC1, _cvmHomographyAll.ptr<float>(r) );
		eimHomo << cvmTmp.ptr<float>(0)[0],cvmTmp.ptr<float>(0)[1],cvmTmp.ptr<float>(0)[2],
				   cvmTmp.ptr<float>(1)[0],cvmTmp.ptr<float>(1)[1],cvmTmp.ptr<float>(1)[2],
				   cvmTmp.ptr<float>(2)[0],cvmTmp.ptr<float>(2)[1],cvmTmp.ptr<float>(2)[2];
		_veimHomography.push_back( eimHomo );
	}
	
	return;
}

void Viewer::keyPressEvent(QKeyEvent *pEvent_)
{
	// Defines the Alt+R shortcut.
	if (pEvent_->key() == Qt::Key_0) 
	{
		_pGL->setInitialPos();
		_pVideo->_pCurrFrame->setView(&_pGL->_eimModelViewGL);
		updateGL(); // Refresh display
	}
	else if (pEvent_->key() == Qt::Key_9) 
	{
		_pGL->_usLevel = ++_pGL->_usLevel%_pGL->_usPyrHeight;
		updateGL();
	}
	else if (pEvent_->key() == Qt::Key_L && !(pEvent_->modifiers() & Qt::ShiftModifier) ){
		_pGL->_bEnableLighting = !_pGL->_bEnableLighting;
		updateGL();
	}
	else if (pEvent_->key() == Qt::Key_F1){
		_bTrack = !_bTrack;
		updateGL();
	}
	else if (pEvent_->key() == Qt::Key_F2){
		_pGL->_bDisplayCamera = !_pGL->_bDisplayCamera;
		updateGL();
	}
	else if (pEvent_->key() == Qt::Key_F3){
		_pGL->_bRenderReference = !_pGL->_bRenderReference;
		updateGL();
	}
	else if (pEvent_->key() == Qt::Key_R && (pEvent_->modifiers() & Qt::ShiftModifier) ){
		reset();
		updateGL();
	}
	else if (pEvent_->key() == Qt::Key_Escape && !(pEvent_->modifiers() & Qt::ShiftModifier) ){
		QCoreApplication::instance()->quit();
	}
	else if (pEvent_->key() == Qt::Key_F && (pEvent_->modifiers() & Qt::ShiftModifier) ){
		if (!isFullScreen()){
			toggleFullScreen();
		}
		else{
			setFullScreen(false);
			resize(1280,480);
		}
	}

	//QGLViewer::keyPressEvent(pEvent_);
	return;
}

void Viewer::mousePressEvent( QMouseEvent *e )
{
	if( e->button() == Qt::LeftButton ){
		_pGL->_nXMotion = _pGL->_nYMotion = 0;
		_pGL->_nXLeftDown    = e->pos().x();
		_pGL->_nYLeftDown    = e->pos().y();
	}
	updateGL();
	return;
}

void Viewer::mouseReleaseEvent( QMouseEvent *e )
{
	if( e->button() == Qt::LeftButton ){
		_pGL->_dXLastAngle = _pGL->_dXAngle;
		_pGL->_dYLastAngle = _pGL->_dYAngle;
	}
	updateGL();
	return;
}

void Viewer::mouseMoveEvent( QMouseEvent *e ){
	if( e->buttons() & Qt::LeftButton ){
		glDisable     ( GL_BLEND );
		_pGL->_nXMotion = e->pos().x() - _pGL->_nXLeftDown;
		_pGL->_nYMotion = e->pos().y() - _pGL->_nYLeftDown;
		_pGL->_dXAngle  = _pGL->_dXLastAngle + _pGL->_nXMotion;
		_pGL->_dYAngle  = _pGL->_dYLastAngle + _pGL->_nYMotion;
	}
	updateGL();
	return;
}

void Viewer::wheelEvent( QWheelEvent *e )
{
	_pGL->_dZoom += e->delta()/1200.;
	e->accept();
}