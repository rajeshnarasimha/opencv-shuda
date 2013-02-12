//display kinect depth in real-time
#define INFO
#define TIMER
#include <GL/glew.h>
#include <gl/freeglut.h>
//#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <string>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "Utility.hpp"

//camera calibration from a sequence of images
#include <opencv2/gpu/gpu.hpp>
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
#include "KeyFrame.h"
#include "CyclicBuffer.h"
#include "VideoSourceKinect.hpp"
#include "CubicGrids.h"
#include "KinfuTracker.h"

//Qt
#include <QResizeEvent>
#include <QGLViewer/qglviewer.h>
#include "SimpleViewer.h"
#include <QCoreApplication>

using namespace std;
Viewer::Viewer(){
	_uResolution = 0;
	_uPyrHeight = 3;
	_eivCw = Eigen::Vector3f(1.5f,1.5f,1.5f);
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
	_bCapture = false;
	_bTrackOnly = false;
}
Viewer::~Viewer()
{
	_pGL->destroyVBOsPBOs();
}

void Viewer::draw()
{
	//load data from video source and model
	_pKinect->getNextFrame(&_nStatus);
	//_pKinect->_pFrame->gpuTransformToWorldCVCV();
	//set viewport
	//_pGL->timerStart();
	//_pKinect->_pFrame->gpuBroxOpticalFlow(*_pPrevFrame,&*_pcvgmColorGraph);
	//_pGL->timerStop();
	//_pKinect->_pFrame->copyTo(&*_pPrevFrame);

	if ( _bCapture )
	{
		_pTracker->track(&*_pKinect->_pCurrFrame,_bTrackOnly);
		//PRINTSTR("trackICP done.");
	}//if( _bCapture )
	else{
		_nStatus = (_nStatus&(~btl::kinect::VideoSourceKinect::MASK_RECORDER))|btl::kinect::VideoSourceKinect::STOP_RECORDING;
	}

	glViewport (0, 0, width()/2, height());
	glScissor  (0, 0, width()/2, height());


	_pKinect->_pRGBCamera->setGLProjectionMatrix(1,0.1f,100.f);
	if (_bViewLocked){
		_pTracker->setCurrView(&_pGL->_eimModelViewGL);
		_pGL->setInitialPos();
	}
	_pGL->viewerGL();
	//glClearColor(1, 0, 0, 0);
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	_pVirtualFrameWorld->assignRTfromGL();
	_pCubicGrids->gpuRaycast(&*_pVirtualFrameWorld); //get virtual frame
	//std::string strPath("C:\\csxsl\\src\\opencv-shuda\\Data\\");
	//std::string strFileName =  boost::lexical_cast<std::string> ( _nRFIdx ) + "1.yml";
	//_pPrevFrameWorld->exportYML(strPath,strFileName);
	//_pPrevFrameWorld->render3DPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel,0,false);
	_pVirtualFrameWorld->gpuRenderPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel);
	{
		_pGL->renderAxisGL();
		_pGL->renderPatternGL(.1f,20,20);
		_pGL->renderPatternGL(1.f,10,10);
		_pGL->renderVoxelGL(_fVolumeSize);
	}
	float aColor[4] = {0.f,0.f,1.f,1.f};
	if (!_strTrackingMethod.compare("ICP")){
		glColor4fv(aColor);
		renderText(5,40,QString("ICP"),QFont("Arial", 13, QFont::Normal));
	}
	else if(!_strTrackingMethod.compare("ORBICP")){
		glColor4fv(aColor);
		renderText(5,40,QString("ORBICP"),QFont("Arial", 13, QFont::Normal));
	}
	else if(!_strTrackingMethod.compare("SURF")){
		glColor4fv(aColor);
		renderText(5,40,QString("SURF"),QFont("Arial", 13, QFont::Normal));
	}
	else if(!_strTrackingMethod.compare("ORB")){
		glColor4fv(aColor);
		renderText(5,40,QString("ORB"),QFont("Arial", 13, QFont::Normal));
	}
	else if(!_strTrackingMethod.compare("SURFICP")){
		glColor4fv(aColor);
		renderText(5,40,QString("SURFICP"),QFont("Arial", 13, QFont::Normal));
	}
	if( _bCapture )
	{
		if (!_bTrackOnly){
			float aColor[4] = {1.f,0.f,0.f,1.f};
			glColor4fv(aColor);
			renderText(230,20,QString("Reconstructing"),QFont("Arial", 13, QFont::Normal));
		}
		else{
			float aColor[4] = {1.f,0.f,0.f,1.f};
			glColor4fv(aColor);
			renderText(230,20,QString("Tracking"),QFont("Arial", 13, QFont::Normal));
		}
	}
	//_pKinect->_pRGBCamera->setGLProjectionMatrix(1,0.2f,30.f);
	//glMatrixMode ( GL_MODELVIEW );
	//glLoadIdentity();
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	////render color graph
	//_pGL->gpuMapRgb2PixelBufferObj(*_pcvgmColorGraph,0);
	//_pKinect->_pRGBCamera->renderCameraInGLLocal(_pGL->_auTexture[_pGL->_usLevel], 0.2f );
	// after set the intrinsics and extrinsics
	// load the matrix to set camera pose
	//_pGL->viewerGL();	
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// render objects
	//if (_pGL->_bRenderReference) drawAxis();
	//_pGL->renderPatternGL(.1f,20,20);
	//_pGL->renderPatternGL(1.f,10,10);
	//_pGL->renderVoxelGL(_fVolumeSize);

	//render current frame
	//_pKinect->_pFrame->renderCameraInWorldCVCV(_pGL.get(),_pGL->_bDisplayCamera,_pGL->_fSize,_pGL->_usLevel);
	//_pKinect->_pFrame->gpuRenderPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel);
	//_pKinect->_pFrame->render3DPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel,0,false);
	//show text
	glColor3f(0.f,1.f,0.f);
	renderText(100,20,QString("FPS:")+QString::number(currentFPS()),QFont("Arial", 10, QFont::Normal));
	switch(_nMode){ 
	case btl::kinect::VideoSourceKinect::RECORDING:
		glColor3f(0.f,1.f,0.f);
		renderText(5,20,QString("Recorder"),QFont("Arial", 13, QFont::Normal));
		//_pGL->drawString("Recorder", 5, height()-20, aColor, GLUT_BITMAP_8_BY_13);
		if ( (_nStatus&btl::kinect::VideoSourceKinect::MASK_RECORDER) == btl::kinect::VideoSourceKinect::CONTINUE_RECORDING ){
			float aColor[4] = {1.f,0.f,0.f,1.f};
			glColor3f(1.f,0.f,0.f);
			renderText(5,40,QString("Recording..."),QFont("Arial", 13, QFont::Normal));
			//_pGL->drawString("Recording...", 5, height()-40, aColor, GLUT_BITMAP_8_BY_13);
		}
		break;
	case btl::kinect::VideoSourceKinect::PLAYING_BACK:
		glColor3f(0.f,1.f,0.f);
		renderText( 5,20, QString("Player"),QFont("Arial", 13, QFont::Normal));
		//_pGL->drawString("Player", 5, height()-20, aColor, GLUT_BITMAP_8_BY_13);
		break;
	case btl::kinect::VideoSourceKinect::SIMPLE_CAPTURING:
		glColor3f(0.f,1.f,0.f);
		renderText( 5,20, QString("Simple"),QFont("Arial", 13, QFont::Normal));
		//_pGL->drawString("Simple", 5, height()-20, aColor, GLUT_BITMAP_8_BY_13);
		break;
	}
	//set viewport 2
	glViewport (width()/2, 0, width()/2, height());
	glScissor  (width()/2, 0, width()/2, height());

	_pKinect->_pRGBCamera->setGLProjectionMatrix(1,0.1f,100.f);
	glMatrixMode ( GL_MODELVIEW );
	glLoadIdentity();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	_pGL->renderAxisGL();
	//_pKinect->_pRGBCamera->LoadTexture(*_pKinect->_pFrame->_acvmShrPtrPyrRGBs[_pGL->_usLevel],&_pGL->_auTexture[_pGL->_usLevel]);
	_pGL->gpuMapRgb2PixelBufferObj(*_pKinect->_pCurrFrame->_acvgmShrPtrPyrRGBs[_pGL->_usLevel],_pGL->_usLevel);
	_pKinect->_pRGBCamera->renderCameraInGLLocal(_pGL->_auTexture[_pGL->_usLevel], 0.2f );
	update();
}

void Viewer::reset(){
	loadFromYml();
 	_pGL.reset( new btl::gl_util::CGLUtil(_uResolution,_uPyrHeight,btl::utility::BTL_GL) );
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
	glEnable     ( GL_CULL_FACE );
	glShadeModel ( GL_FLAT );
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	_pKinect.reset( new btl::kinect::VideoSourceKinect(_uResolution,_uPyrHeight,_bUseNIRegistration,_eivCw) );
	switch(_nMode)
	{
	case btl::kinect::VideoSourceKinect::SIMPLE_CAPTURING: //the simple capturing mode of the rgbd camera
		_pKinect->initKinect();
		break;
	case btl::kinect::VideoSourceKinect::RECORDING: //record the captured sequence from the camera
		_pKinect->setDumpFileName(_oniFileName);
		_pKinect->initRecorder(_oniFileName,_nRecordingTimeInSecond);
		break;
	case btl::kinect::VideoSourceKinect::PLAYING_BACK: //replay from files
		_pKinect->initPlayer(_oniFileName,_bRepeat);
		break;
	}

	_pPrevFrame.reset(new btl::kinect::CKeyFrame(_pKinect->_pRGBCamera.get(),_uResolution,_uPyrHeight,_eivCw));
	_pVirtualFrameWorld.reset(new btl::kinect::CKeyFrame(_pKinect->_pRGBCamera.get(),_uResolution,_uPyrHeight,_eivCw));
	_pKinect->getNextFrame(&_nStatus);
/*
	_pKinect->_pFrame->gpuTransformToWorldCVCV();
	_pKinect->_pFrame->setView(&_pGL->_eimModelViewGL);
	_pKinect->_pFrame->copyTo(&*_pPrevFrame);*/

	_pcvgmColorGraph.reset(new cv::gpu::GpuMat(btl::kinect::__aKinectH[_uResolution],btl::kinect::__aKinectW[_uResolution],CV_8UC3));

	//initialize the cubic grids
	_pCubicGrids.reset( new btl::geometry::CCubicGrids(_uCubicGridResolution,_fVolumeSize) );
	//initialize the tracker
	_pTracker.reset( new btl::geometry::CKinFuTracker(_pKinect->_pCurrFrame.get(),_pCubicGrids));
	if (!_strTrackingMethod.compare("ICP")){
		_pTracker->setMethod(btl::geometry::CKinFuTracker::ICP);
	}
	else if(!_strTrackingMethod.compare("ORBICP")){
		_pTracker->setMethod(btl::geometry::CKinFuTracker::ORBICP);
	}
	else if(!_strTrackingMethod.compare("SURF")){
		_pTracker->setMethod(btl::geometry::CKinFuTracker::SURF);
	}
	else if(!_strTrackingMethod.compare("ORB")){
		_pTracker->setMethod(btl::geometry::CKinFuTracker::ORB);
	}
	else if(!_strTrackingMethod.compare("ORBICP")){
		_pTracker->setMethod(btl::geometry::CKinFuTracker::ORBICP);
	}
	else{

	}
	_pTracker->init(_pKinect->_pCurrFrame.get());
	_pTracker->setNextView(&_pGL->_eimModelViewGL);//printVolume();



	return;
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

	reset();
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
	cv::FileStorage cFSRead ( "C:\\csxsl\\src\\opencv-shuda\\btl_rgbd\\kinect_fusion_qglviewer\\KinectFusionQGLViewer.yml", cv::FileStorage::READ );
#endif
	cFSRead["uResolution"] >> _uResolution;
	cFSRead["uPyrHeight"] >> _uPyrHeight;
	cFSRead["bUseNIRegistration"] >> _bUseNIRegistration;
	cFSRead["uCubicGridResolution"] >> _uCubicGridResolution;
	cFSRead["fVolumeSize"] >> _fVolumeSize;
	_eivCw = Eigen::Vector3f(_fVolumeSize/2,_fVolumeSize/2,_fVolumeSize/2);
	//rendering
	cFSRead["bDisplayImage"] >> _bDisplayImage;
	cFSRead["bLightOn"] >> _bLightOn;
	cFSRead["bRenderReference"] >> _bRenderReference;
	cFSRead["nMode"] >> _nMode;//1 kinect; 2 recorder; 3 player
	cFSRead["oniFile"] >> _oniFileName;
	cFSRead["bRepeat"] >> _bRepeat;
	cFSRead["nRecordingTimeInSecond"] >> _nRecordingTimeInSecond;
	cFSRead["nStatus"] >> _nStatus;
	cFSRead["Tracking_Method"] >> _strTrackingMethod;

	cFSRead.release();
}

void Viewer::keyPressEvent(QKeyEvent *pEvent_)
{
	// Defines the Alt+R shortcut.
	if (pEvent_->key() == Qt::Key_0) 
	{
		_pTracker->setNextView(&_pGL->_eimModelViewGL);
		_pGL->setInitialPos();
		updateGL(); // Refresh display
	}
	else if (pEvent_->key() == Qt::Key_BracketLeft)
	{
		_pTracker->setPrevView(&_pGL->_eimModelViewGL);
		_pGL->setInitialPos();
		updateGL(); // Refresh display
	}
	else if (pEvent_->key() == Qt::Key_2)
	{
		_bViewLocked = !_bViewLocked;
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
	else if (pEvent_->key() == Qt::Key_F2){
		_pGL->_bDisplayCamera = !_pGL->_bDisplayCamera;
		updateGL();
	}
	else if (pEvent_->key() == Qt::Key_F3){
		_pGL->_bRenderReference = !_pGL->_bRenderReference;
		updateGL();
	}
	else if (pEvent_->key() == Qt::Key_R && !(pEvent_->modifiers() & Qt::ShiftModifier) ){
		if (_nMode == btl::kinect::VideoSourceKinect::PLAYING_BACK)	{
			_pKinect->initPlayer(_oniFileName,_bRepeat);
			_nStatus = (_nStatus&(~btl::kinect::VideoSourceKinect::MASK1))|btl::kinect::VideoSourceKinect::CONTINUE;
		};
		_pKinect->getNextFrame(&_nStatus);
		_pVirtualFrameWorld.reset(new btl::kinect::CKeyFrame(_pKinect->_pRGBCamera.get(),_uResolution,_uPyrHeight,_eivCw));	
		//initialize the tracker
		_pTracker->init(_pKinect->_pCurrFrame.get());
		_pTracker->setNextView(&_pGL->_eimModelViewGL);//printVolume();
		updateGL();
	}
	else if (pEvent_->key() == Qt::Key_R && (pEvent_->modifiers() & Qt::ShiftModifier) ){
		reset();
		updateGL();
	}
	else if (pEvent_->key() == Qt::Key_C && !(pEvent_->modifiers() & Qt::ShiftModifier) ){
		_bCapture = !_bCapture;
		if (_bCapture){
			_nStatus = (_nStatus&(~btl::kinect::VideoSourceKinect::MASK_RECORDER))|btl::kinect::VideoSourceKinect::START_RECORDING;
		}
		else{
			_nStatus = (_nStatus&(~btl::kinect::VideoSourceKinect::MASK_RECORDER))|btl::kinect::VideoSourceKinect::STOP_RECORDING;
		}
		updateGL();
	}
	else if (pEvent_->key() == Qt::Key_T && !(pEvent_->modifiers() & Qt::ShiftModifier) ){
		_bTrackOnly = !_bTrackOnly;
		updateGL();
	}
	else if (pEvent_->key() == Qt::Key_S && !(pEvent_->modifiers() & Qt::ShiftModifier) ){
		_nStatus = (_nStatus&(~btl::kinect::VideoSourceKinect::MASK_RECORDER))|btl::kinect::VideoSourceKinect::DUMP_RECORDING;
		updateGL();
	}
	else if (pEvent_->key() == Qt::Key_P && !(pEvent_->modifiers() & Qt::ShiftModifier) ){
		if ((_nStatus&btl::kinect::VideoSourceKinect::MASK1) == btl::kinect::VideoSourceKinect::PAUSE){
			_nStatus = (_nStatus&(~btl::kinect::VideoSourceKinect::MASK1))|btl::kinect::VideoSourceKinect::CONTINUE;
		}else if ((_nStatus&btl::kinect::VideoSourceKinect::MASK1) == btl::kinect::VideoSourceKinect::CONTINUE){
			_nStatus = (_nStatus&(~btl::kinect::VideoSourceKinect::MASK1))|btl::kinect::VideoSourceKinect::PAUSE;
		}
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
}

void Viewer::mousePressEvent( QMouseEvent *e )
{
	if( e->button() == Qt::LeftButton ){
		_pGL->_nXMotion = _pGL->_nYMotion = 0;
		_pGL->_nXLeftDown    = e->pos().x();
		_pGL->_nYLeftDown    = e->pos().y();
	}
	updateGL();
}

void Viewer::mouseReleaseEvent( QMouseEvent *e )
{
	if( e->button() == Qt::LeftButton ){
		_pGL->_dXLastAngle = _pGL->_dXAngle;
		_pGL->_dYLastAngle = _pGL->_dYAngle;
	}
	updateGL();
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
}

void Viewer::wheelEvent( QWheelEvent *e )
{
	_pGL->_dZoom += e->delta()/1200.;
	e->accept();
}







