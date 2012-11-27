/****************************************************************************

 Copyright (C) 2002-2011 Gilles Debunne. All rights reserved.

 This file is part of the QGLViewer library version 2.3.17.

 http://www.libqglviewer.com - contact@libqglviewer.com

 This file may be used under the terms of the GNU General Public License 
 versions 2.0 or 3.0 as published by the Free Software Foundation and
 appearing in the LICENSE file included in the packaging of this file.
 In addition, as a special exception, Gilles Debunne gives you certain 
 additional rights, described in the file GPL_EXCEPTION in this package.

 libQGLViewer uses dual licensing. Commercial/proprietary software must
 purchase a libQGLViewer Commercial License.

 This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
 WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

*****************************************************************************/

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
#include "KeyFrame.h"
#include "CyclicBuffer.h"
#include "VideoSourceKinect.hpp"
//Qt
#include <QResizeEvent>
#include "simpleViewer.h"

using namespace std;
Viewer::Viewer(){
	_uResolution = 0;
	_uPyrHeight = 3;
	_eivCw = Eigen::Vector3f(1.5f,1.5f,-0.3f);
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
}
Viewer::~Viewer()
{
	_pGL->destroyVBOsPBOs();
}
// Draws a spiral
void Viewer::drawLogo() const{
	const float nbSteps = 200.0;

	glBegin(GL_QUAD_STRIP);
	for (int i=0; i<nbSteps; ++i)
	{
		const float ratio = i/nbSteps;
		const float angle = 21.0*ratio;
		const float c = cos(angle);
		const float s = sin(angle);
		const float r1 = 1.0 - 0.8f*ratio;
		const float r2 = 0.8f - 0.8f*ratio;
		const float alt = ratio - 0.5f;
		const float nor = 0.5f;
		const float up = sqrt(1.0-nor*nor);
		glColor3f(1.0-ratio, 0.2f , ratio);
		glNormal3f(nor*c, up, nor*s);
		glVertex3f(r1*c, alt, r1*s);
		glVertex3f(r2*c, alt+0.05f, r2*s);
	}
	glEnd();
	return;
}//drawLogo()
void Viewer::draw()
{
	//load data from video source and model
	_pKinect->getNextFrame(btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV,&_nStatus);
	_pKinect->_pFrame->gpuTransformToWorldCVCV();
	//set viewport

	glViewport (0, 0, width()/2, height());
	glScissor  (0, 0, width()/2, height());
	_pKinect->_pRGBCamera->setGLProjectionMatrix(1,0.2f,30.f);
	glMatrixMode ( GL_MODELVIEW );
	// after set the intrinsics and extrinsics
	// load the matrix to set camera pose
	_pGL->viewerGL();	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// render objects
	_pGL->renderAxisGL();
	_pGL->renderPatternGL(.1f,20,20);
	_pGL->renderPatternGL(1.f,10,10);
	_pGL->renderVoxelGL(_fVolumeSize);
	//_pGL->timerStart();
	_pKinect->_pFrame->renderCameraInWorldCVCV(_pGL.get(),_pGL->_bDisplayCamera,_pGL->_fSize,_pGL->_usLevel);
	_pKinect->_pFrame->gpuRenderPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel);
	//_pKinect->_pFrame->render3DPtsInWorldCVCV(_pGL.get(),_pGL->_uLevel,0,false);
	/*//show text
	float aColor[4] = {0.f,1.f,0.f,1.f};
	switch(_nMode){ 
	case btl::kinect::VideoSourceKinect::RECORDING:
		_pGL->drawString("Recorder", 5, height()-20, aColor, GLUT_BITMAP_8_BY_13);
		if ( (_nStatus&btl::kinect::VideoSourceKinect::MASK_RECORDER) == btl::kinect::VideoSourceKinect::CONTINUE_RECORDING ){
			float aColor[4] = {1.f,0.f,0.f,1.f};
			_pGL->drawString("Recording...", 5, height()-40, aColor, GLUT_BITMAP_8_BY_13);
		}
		break;
	case btl::kinect::VideoSourceKinect::PLAYING_BACK:
		_pGL->drawString("Player", 5, height()-20, aColor, GLUT_BITMAP_8_BY_13);
		break;
	case btl::kinect::VideoSourceKinect::SIMPLE_CAPTURING:
		_pGL->drawString("Simple", 5, height()-20, aColor, GLUT_BITMAP_8_BY_13);
		break;
	}*/
	//set viewport 2
	glViewport (width()/2, 0, width()/2, height());
	glScissor  (width()/2, 0, width()/2, height());

	_pKinect->_pRGBCamera->setGLProjectionMatrix(1,0.1f,100.f);
	glMatrixMode ( GL_MODELVIEW );
	glLoadIdentity();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	_pGL->renderAxisGL();
	_pKinect->_pRGBCamera->LoadTexture(*_pKinect->_pFrame->_acvmShrPtrPyrRGBs[_pGL->_usLevel],&_pGL->_auTexture[_pGL->_usLevel]);
	_pKinect->_pRGBCamera->renderCameraInGLLocal(_pGL->_auTexture[_pGL->_usLevel], *_pKinect->_pFrame->_acvmShrPtrPyrRGBs[_pGL->_usLevel],0.2f );

	update();
}

void Viewer::init()
{
	loadFromYml();
	// Restore previous viewer state.
	restoreStateFromFile();
	resize(1280,480);
  
	// Opens help window
	help();

	//
	GLenum eError = glewInit(); 
	if (GLEW_OK != eError){
		PRINTSTR("glewInit() error.");
		PRINT( glewGetErrorString(eError) );
	}
	_pGL.reset( new btl::gl_util::CGLUtil(_uResolution,_uPyrHeight,btl::utility::BTL_GL) );
	_pGL->setCudaDeviceForGLInteroperation();//initialize before using any cuda component
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

	_pKinect->getNextFrame(btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV,&_nStatus);
	_pKinect->_pFrame->gpuTransformToWorldCVCV();
	_pKinect->_pFrame->setView(&_pGL->_eimModelViewGL);

	return;
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
		/*int nHeightO= event->oldSize().height();
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
		}*/
	}
	//and event->oldsize()
	QWidget::resizeEvent(event);
}

void Viewer::loadFromYml(){

#if __linux__
	cv::FileStorage cFSRead( "/space/csxsl/src/opencv-shuda/Data/kinect_intrinsics.yml", cv::FileStorage::READ );
#else if _WIN32 || _WIN64
	cv::FileStorage cFSRead ( "C:\\csxsl\\src\\opencv-shuda\\btl_rgbd\\render_qglviewer\\RenderQGlviewer.yml", cv::FileStorage::READ );
#endif
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

void Viewer::keyPressEvent(QKeyEvent *e)
{
	// Defines the Alt+R shortcut.
	if (e->key() == Qt::Key_0) 
	{
		_pKinect->_pFrame->setView(&_pGL->_eimModelViewGL);
		_pGL->setInitialPos();
		updateGL(); // Refresh display
	}
	else if (e->key() == Qt::Key_9) 
	{
		_pGL->_usLevel = ++_pGL->_usLevel%_pGL->_usPyrHeight;
		updateGL();
	}
	QGLViewer::keyPressEvent(e);
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




