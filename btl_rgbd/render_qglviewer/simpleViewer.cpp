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
#include "VideoSourceKinect.hpp"
//Qt
#include <QResizeEvent>
#include "simpleViewer.h"

using namespace std;
Viewer::Viewer()
{
}
Viewer::~Viewer()
{
	_pGL->destroyVBOsPBOs();
}
// Draws a spiral
void Viewer::draw()
{
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

  _pKinect->getNextFrame(btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV);
  //_pKinect->_pFrame->gpuDetectPlane(u);
  _pKinect->_pFrame->gpuTransformToWorldCVCV();

  glMatrixMode ( GL_MODELVIEW );
  // after set the intrinsics and extrinsics
  // load the matrix to set camera pose
  //glLoadIdentity();
  //glLoadMatrixd( _mGLMatrix.data() );
  //_pGL->viewerGL();

  _pKinect->_pFrame->renderCameraInWorldCVCV(_pGL.get(),_pGL->_bDisplayCamera,.05f,_pGL->_usLevel);
  _pKinect->_pFrame->gpuRenderPtsInWorldCVCV(_pGL.get(),_pGL->_usLevel);
}

void Viewer::init()
{
  // Restore previous viewer state.
  restoreStateFromFile();
  resize(1280,960);
  
  // Opens help window
  help();

  //
  GLenum eError = glewInit(); 
  if (GLEW_OK != eError){
	  PRINTSTR("glewInit() error.");
	  PRINT( glewGetErrorString(eError) );
  }
  _pGL.reset( new btl::gl_util::CGLUtil(1,3,btl::utility::BTL_CV) );
  _pGL->setCudaDeviceForGLInteroperation();//initialize before using any cuda component
  _pKinect.reset( new btl::kinect::VideoSourceKinect(1,3,true,1.5f,1.5f,-0.3f) );

  _pGL->constructVBOsPBOs();

  _pGL->_usLevel=2;
  _pKinect->getNextFrame(btl::kinect::VideoSourceKinect::GPU_PYRAMID_CV);
  _pKinect->_pFrame->gpuTransformToWorldCVCV();
}

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
	if( fabs(fAsp - 0.75f) > 0.0001f )//if the aspect ratio is not 3:4
	{
		int nUnit = nHeight/3;
		int nWidth= nUnit*4;
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
