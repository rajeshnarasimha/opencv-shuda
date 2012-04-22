//#define INFO
//#define TIMER

#define INFO
#include <gl/glew.h>
#include <GL/freeglut.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <boost/shared_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpumat.hpp>
#include <vector>
#include <Eigen/Core>

#include "cuda/cv/common.hpp"
#include "OtherUtil.hpp"
#include "Kinect.h"
#include "GLUtil.h"
#include "CudaLib.h"

namespace btl{	namespace gl_util
{
	
CGLUtil::CGLUtil(btl::utility::tp_coordinate_convention eConvention_ /*= btl::utility::BTL_GL*/)
:_eConvention(eConvention_){
	_dZoom = 1.;
	_dZoomLast = 1.;
	_dScale = .1;

	_dXAngle = 0;
	_dYAngle = 0;
	_dXLastAngle = 0;
	_dYLastAngle = 0;
	_dX = 0;
	_dY = 0;
	_dXLast = 0;
	_dYLast = 0;

	_nXMotion = 0;
	_nYMotion = 0;

	_aCentroid[0] = 2.f; _aCentroid[1] = 2.f; _aCentroid[2] = 1.f; 
	_aLight[0] = 2.0f;
	_aLight[1] = 1.7f;
	_aLight[2] =-0.2f;
	_aLight[3] = 1.0f;

	_bRenderNormal = false;
	_bEnableLighting = false;
	_fSize = 0.2f;
	_usPyrLevel=0;
}
void CGLUtil::mouseClick ( int nButton_, int nState_, int nX_, int nY_ )
{
	if ( nButton_ == GLUT_LEFT_BUTTON )
	{
		if ( nState_ == GLUT_DOWN )
		{
			_nXMotion = _nYMotion = 0;
			_nXLeftDown    = nX_;
			_nYLeftDown    = nY_;

			_bLButtonDown = true;
		}
		else if( nState_ == GLUT_UP )// button up
		{
			_dXLastAngle = _dXAngle;
			_dYLastAngle = _dYAngle;
			_bLButtonDown = false;
		}
		glutPostRedisplay();
	}
	else if ( GLUT_RIGHT_BUTTON )
	{
		if ( nState_ == GLUT_DOWN )
		{
			_nXMotion = _nYMotion = 0;
			_nXRightDown  = nX_;
			_nYRightDown  = nY_;
			_dZoomLast    = _dZoom;
			_bRButtonDown = true;
		}
		else if( nState_ == GLUT_UP )
		{
			_dXLast = _dX;
			_dYLast = _dY;
			_bRButtonDown = false;
		}
		glutPostRedisplay();
	}

	return;
}
void CGLUtil::mouseMotion ( int nX_, int nY_ )
{
	if ( _bLButtonDown == true )
	{
		glDisable     ( GL_BLEND );
		_nXMotion = nX_ - _nXLeftDown;
		_nYMotion = nY_ - _nYLeftDown;
		_dXAngle  = _dXLastAngle + _nXMotion;
		_dYAngle  = _dYLastAngle + _nYMotion;
	}
	else if ( _bRButtonDown == true )
	{
		glDisable     ( GL_BLEND );
		_nXMotion = nX_ - _nXRightDown;
		_nYMotion = nY_ - _nYRightDown;
		_dX  = _dXLast + _nXMotion;
		_dY  = _dYLast + _nYMotion;
		_dZoom = _dZoomLast + (_nXMotion + _nYMotion)/200.;
		//PRINT(_dZoom);
	}

	glutPostRedisplay();
}

void CGLUtil::specialKeys( int key, int x, int y ){
	switch ( key ) {
	case GLUT_KEY_F2: //display camera
		_bDisplayCamera = !_bDisplayCamera;
		glutPostRedisplay();
		break;
	case GLUT_KEY_F3:
		_bRenderReference = !_bRenderReference;
		glutPostRedisplay();
		break;
	}
}
void CGLUtil::normalKeys ( unsigned char key, int x, int y )
{
	switch( key )
	{
	case 27:
		exit ( 0 );
		break;
	case 'g':
		//zoom in
		glDisable( GL_BLEND );
		_dZoom += _dScale;
		glutPostRedisplay();
		//PRINT( _dZoom );
		break;
	case 'h':
		//zoom out
		glDisable( GL_BLEND );
		_dZoom -= _dScale;
		glutPostRedisplay();
		//PRINT( _dZoom );
		break;
	case 'l':
		_bEnableLighting = !_bEnableLighting;
		glutPostRedisplay();
		//PRINT( _bEnableLighting );
		break;
	case 'n':
		_bRenderNormal = !_bRenderNormal;
		glutPostRedisplay();
		//PRINT( _bRenderNormal );
		break;
	case 'k':
		_fSize += 0.05f;// range from 0.05 to 1 by step 0.05
		_fSize = _fSize < 1 ? _fSize: 1;
		glutPostRedisplay();
		//PRINT( _fSize );
		break;
	case 'j':
		_fSize -= 0.05f;
		_fSize = _fSize > 0.05f? _fSize : 0.05f;
		glutPostRedisplay();
		//PRINT( _fSize );
		break;
	case '<':
		_dYAngle += 1.0;
		glutPostRedisplay();
		break;
	case '>':
		_dYAngle -= 1.0;
		glutPostRedisplay();
		break;
	case '9':
		_usPyrLevel = ++_usPyrLevel%4;
		//PRINT(_usPyrLevel);
		break;
	case '0'://reset camera location
		_dXAngle = 0.;
		_dYAngle = 0.;
		_dZoom = 1.;
		break;
	}

	return;
}
void CGLUtil::viewerGL()
{
	// load the matrix to set camera pose
	//glLoadMatrixd( _adModelViewGL );
	glLoadMatrixd(_eimModelViewGL.data());
	glTranslated( _aCentroid[0], _aCentroid[1], _aCentroid[2] ); // 5. translate back to the original camera pose
	_dZoom = _dZoom < 0.1? 0.1: _dZoom;
	_dZoom = _dZoom > 10? 10: _dZoom;
	glScaled( _dZoom, _dZoom, _dZoom );                          // 4. zoom in/out
	if( btl::utility::BTL_GL == _eConvention )
		glRotated ( _dXAngle, 0, 1 ,0 );                         // 3. rotate horizontally
	else if( btl::utility::BTL_CV == _eConvention )						//mouse x-movement is the rotation around y-axis
		glRotated ( _dXAngle, 0,-1 ,0 );                        
	glRotated ( _dYAngle, 1, 0 ,0 );                             // 2. rotate vertically
	glTranslated(-_aCentroid[0],-_aCentroid[1],-_aCentroid[2] ); // 1. translate the world origin to align with object centroid

	// light position in 3d
	glLightfv(GL_LIGHT0, GL_POSITION, _aLight);
}
void CGLUtil::renderAxisGL() const
{
	glDisable(GL_LIGHTING);

	glPushMatrix();
	float fAxisLength = 1.f;
	float fLengthWidth = 1;
	Eigen::Vector3f vOrigin,vXAxis,vYAxis,vZAxis;
	vOrigin<< .0f, .0f, .0f;
	vXAxis << fAxisLength, .0f, .0f;
	vYAxis << .0f, fAxisLength, .0f;
	vZAxis << .0f, .0f, fAxisLength;

	glLineWidth( fLengthWidth );
	// x axis
	glColor3f ( 1.f, .0f, .0f );
	glBegin ( GL_LINES );
	glVertex3fv ( vOrigin.data() );
	glVertex3fv ( vXAxis.data() );
	glEnd();
	// y axis
	glColor3f ( .0f, 1.f, .0f );
	glBegin ( GL_LINES );
	glVertex3fv ( vOrigin.data() );
	glVertex3fv ( vYAxis.data() );
	glEnd();
	// z axis
	glColor3f ( .0f, .0f, 1.f );
	glBegin ( GL_LINES );
	glVertex3fv ( vOrigin.data() );
	glVertex3fv ( vZAxis.data());
	glEnd();
	glPopMatrix();
}
void CGLUtil::clearColorDepth()
{
	glClearColor ( 0.1f,0.1f,0.4f,1.0f );
	glClearDepth ( 1.0 );
}
void CGLUtil::init()
{
	//cv::Mat cvmTemp(4,4,CV_64FC1,(void*)_adModelViewGL);
	//cv::setIdentity(cvmTemp);
	_eimModelViewGL.setIdentity();
	//disk list
	_uDisk = glGenLists(1);
	_pQObj = gluNewQuadric();
	gluQuadricDrawStyle(_pQObj, GLU_FILL); //LINE); /* wireframe */
	gluQuadricNormals(_pQObj, GLU_SMOOTH);// FLAT);//
	glNewList(_uDisk, GL_COMPILE);
	gluDisk(_pQObj, 0.0, 0.01, 4, 1);//render a disk on z=0 plane
	glEndList();
	//normal list
	_uNormal = glGenLists(2);
	glNewList(_uNormal, GL_COMPILE);
	glDisable(GL_LIGHTING);
	glBegin(GL_LINES);
	glColor3d(1.,0.,0.);
	glVertex3d(0.,0.,0.);
	glVertex3d(0.,0.,0.016);
	glEnd();
	glEndList();
	//voxel list
	_uVoxel = glGenLists(3);
	glNewList(_uVoxel, GL_COMPILE);
	glDisable(GL_LIGHTING);
	renderVoxelGL(1.f);
	glEndList();

	_uOctTree = glGenLists(3);
	glNewList(_uOctTree, GL_COMPILE);
	glDisable(GL_LIGHTING);
	renderOctTree(0,0,0,2,1);
	glEndList();

	// light
	GLfloat mat_diffuse[] = { 1.0, 1.0, 1.0, 1.0};
	glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);

	GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
	glLightfv (GL_LIGHT0, GL_DIFFUSE, light_diffuse);

	glEnable(GL_RESCALE_NORMAL);
	glEnable(GL_LIGHT0);

	//glEnable(GL_BLEND);
	//glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
}//init();
void CGLUtil::setCudaDeviceForGLInteroperation(){
	cudaDeviceProp  sProp;
	memset( &sProp, 0, sizeof( cudaDeviceProp ) );
	sProp.major = 1;
	sProp.minor = 0;
	int nDev;
	cudaSafeCall( cudaChooseDevice( &nDev, &sProp ) );
	// tell CUDA which nDev we will be using for graphic interop
	// from the programming guide:  Interoperability with OpenGL
	//     requires that the CUDA device be specified by
	//     cudaGLSetGLDevice() before any other runtime calls.
	cudaSafeCall( cudaGLSetGLDevice( nDev ) );

	return;
}//setCudaDeviceForGLInteroperation()

void CGLUtil::createVBO(const unsigned int uRows_, const unsigned int uCols_, const unsigned short usChannel_, const unsigned short usBytes_,
	GLuint* puVBO_, cudaGraphicsResource** ppResourceVBO_ ){
	// the first four are standard OpenGL, the 5th is the CUDA reg 
	// of the VBO these calls exist starting in OpenGL 1.5
	glGenBuffers(1, puVBO_);
	glBindBuffer(GL_ARRAY_BUFFER, *puVBO_);
	glBufferData(GL_ARRAY_BUFFER, uRows_*uCols_*usChannel_*usBytes_, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaSafeCall( cudaGraphicsGLRegisterBuffer( ppResourceVBO_, *puVBO_, cudaGraphicsMapFlagsWriteDiscard) );
}//createVBO()
void CGLUtil::releaseVBO( GLuint uVBO_, cudaGraphicsResource *pResourceVBO_ ){
	// clean up OpenGL and CUDA
	cudaSafeCall( cudaGraphicsUnregisterResource( pResourceVBO_ ) );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
	glDeleteBuffers( 1, &uVBO_ );
}//releaseVBO()
void CGLUtil::createPBO(const unsigned int uRows_, const unsigned int uCols_, const unsigned short usChannel_, const unsigned short usBytes_, GLuint* puPBO_, cudaGraphicsResource** ppResourcePixelBO_, GLuint* pTexture_){
	//Generate a buffer ID called a PBO (Pixel Buffer Object)
	//http://rickarkin.blogspot.co.uk/2012/03/use-pbo-to-share-buffer-between-cuda.html
	//generate a texture
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, pTexture_);
	glBindTexture ( GL_TEXTURE_2D, *pTexture_ );
	glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
	glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
	glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST ); // cheap scaling when image bigger than texture
	glTexParameteri ( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST ); // cheap scaling when image smalled than texture  
	// 2d texture, level of detail 0 (normal), 3 components (red, green, blue), x size from image, y size from image,
	// border 0 (normal), rgb color data, unsigned byte data, and finally the data itself.
	glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGB, uCols_, uRows_, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL ); //???????????????????
	glTexParameteri(GL_TEXTURE_2D , GL_TEXTURE_MIN_FILTER , GL_NEAREST);
	glBindTexture( GL_TEXTURE_2D, 0);
	//generate PBO
	glGenBuffers(1, puPBO_);
	//Make this the current UNPACK buffer
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *puPBO_);
	//Allocate data for the buffer. 4-channel 8-bit image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, uRows_*uCols_*	usChannel_ *usBytes_, NULL, GL_STREAM_DRAW); //GL_STREAM_DRAW //http://www.opengl.org/sdk/docs/man/xhtml/glBufferData.xml
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0 );
	cudaSafeCall( cudaGraphicsGLRegisterBuffer( ppResourcePixelBO_, *puPBO_, cudaGraphicsRegisterFlagsNone) );//cudaGraphicsRegisterFlagsWriteDiscard) ); //
	//cudaSafeCall( cudaGLRegisterBufferObject(*puPBO_) ); //deprecated
}//createVBO()
void CGLUtil::releasePBO( GLuint uPBO_,cudaGraphicsResource *pResourcePixelBO_ ){
	/*// unregister this buffer object with CUDA
	//http://rickarkin.blogspot.co.uk/2012/03/use-pbo-to-share-buffer-between-cuda.html
	cudaSafeCall( cudaGraphicsUnregisterResource( pResourcePixelBO_ ) );
	glDeleteBuffers(1, &uPBO_);*/

}//releaseVBO()
void CGLUtil::constructVBOsPBOs(){
	for (ushort u=0; u<4; u++){
		createVBO( btl::kinect::__aKinectW[u], btl::kinect::__aKinectH[u],3,sizeof(float),&_auPtVBO[u], &_apResourcePtVBO[u] );
		createVBO( btl::kinect::__aKinectW[u], btl::kinect::__aKinectH[u],3,sizeof(float),&_auNlVBO[u], &_apResourceNlVBO[u] );
		createVBO( btl::kinect::__aKinectW[u], btl::kinect::__aKinectH[u],3,sizeof(uchar),&_auRGBVBO[u],&_apResourceRGBVBO[u]);
		createPBO( btl::kinect::__aKinectW[u], btl::kinect::__aKinectH[u],3,sizeof(uchar),&_auRGBPixelBO[u],&_apResourceRGBPxielBO[u],&_auTexture[u]);
	}//for each pyramid level
}//constructVBOsPBOs()
void CGLUtil::destroyVBOsPBOs(){
	for (ushort u=0; u<4; u++){
		releaseVBO( _auPtVBO[u], _apResourcePtVBO[u] );
		releaseVBO( _auNlVBO[u], _apResourceNlVBO[u] );
		releaseVBO( _auRGBVBO[u],_apResourceRGBVBO[u]);
		releasePBO( _auRGBPixelBO[u],_apResourceRGBPxielBO[u]);
	}//for each pyramid level
}
void CGLUtil::gpuMapPtResources(const cv::gpu::GpuMat& cvgmPts_, const ushort usPyrLevel_){
	// map OpenGL buffer object for writing from CUDA
	void *pDev;
	cudaGraphicsMapResources(1, &_apResourcePtVBO[usPyrLevel_], 0);
	size_t nSize; 
	cudaGraphicsResourceGetMappedPointer((void **)&pDev, &nSize, _apResourcePtVBO[usPyrLevel_] );
	cv::gpu::GpuMat cvgmPts(btl::kinect::__aKinectH[usPyrLevel_],btl::kinect::__aKinectW[usPyrLevel_],CV_32FC3,pDev);
	cudaGraphicsUnmapResources(1, &_apResourcePtVBO[usPyrLevel_], 0);
	cvgmPts_.copyTo(cvgmPts);
	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, _auPtVBO[usPyrLevel_]);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	//glColor3f(1.0, 0.0, 0.0);
	//glDrawArrays(GL_POINTS, 0, btl::kinect::__aKinectWxH[usPyrLevel_] );
	//glDisableClientState(GL_VERTEX_ARRAY);
}
void CGLUtil::gpuMapNlResources(const cv::gpu::GpuMat& cvgmNls_, const ushort usPyrLevel_){
	// map OpenGL buffer object for writing from CUDA
	void *pDev;
	cudaGraphicsMapResources(1, &_apResourceNlVBO[usPyrLevel_], 0);
	size_t nSize; 
	cudaGraphicsResourceGetMappedPointer((void **)&pDev, &nSize, _apResourceNlVBO[usPyrLevel_] );
	cv::gpu::GpuMat cvgmNls(btl::kinect::__aKinectH[usPyrLevel_],btl::kinect::__aKinectW[usPyrLevel_],CV_32FC3,pDev);
	cudaGraphicsUnmapResources(1, &_apResourceNlVBO[usPyrLevel_], 0);
	cvgmNls_.copyTo(cvgmNls);
	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, _auNlVBO[usPyrLevel_]);
	glNormalPointer(GL_FLOAT, 0, 0);
	glEnableClientState(GL_NORMAL_ARRAY);
	//glColor3f(1.0, 0.0, 0.0);
	//glDrawArrays(GL_POINTS, 0, btl::kinect::__aKinectWxH[usPyrLevel_] );
	//glDisableClientState(GL_NORMAL_ARRAY);
}
void CGLUtil::gpuMapRGBResources(const cv::gpu::GpuMat& cvgmRGBs_, const ushort usPyrLevel_){
	// map OpenGL buffer object for writing from CUDA
	void *pDev;
	cudaGraphicsMapResources(1, &_apResourceRGBVBO[usPyrLevel_], 0);
	size_t nSize; 
	cudaGraphicsResourceGetMappedPointer((void **)&pDev, &nSize, _apResourceRGBVBO[usPyrLevel_] );
	cv::gpu::GpuMat cvgmRGBs(btl::kinect::__aKinectH[usPyrLevel_],btl::kinect::__aKinectW[usPyrLevel_],CV_8UC3,pDev);
	cudaGraphicsUnmapResources(1, &_apResourceRGBVBO[usPyrLevel_], 0);
	cvgmRGBs_.copyTo(cvgmRGBs);
	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, _auRGBVBO[usPyrLevel_]);
	glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);
	glEnableClientState(GL_COLOR_ARRAY);
}
void CGLUtil::gpuMapRGBPBO(const cv::gpu::GpuMat& cvgmRGB_, const ushort usPyrLevel_ ){
	//http://rickarkin.blogspot.co.uk/2012/03/use-pbo-to-share-buffer-between-cuda.html

	// map OpenGL buffer object for writing from CUDA
	void *pDev;
	cudaSafeCall( cudaGraphicsMapResources(1, &_apResourceRGBPxielBO[usPyrLevel_], 0)); 
	size_t nSize; 
	cudaSafeCall( cudaGraphicsResourceGetMappedPointer((void **)&pDev, &nSize , _apResourceRGBPxielBO[usPyrLevel_]));
	cv::gpu::GpuMat cvgmRGBA( btl::kinect::__aKinectH[usPyrLevel_], btl::kinect::__aKinectW[usPyrLevel_], CV_8UC3, pDev);
	//btl::device::rgb2RGBA(cvgmRGB_,0, &cvgmRGBA);
	cvgmRGB_.copyTo(cvgmRGBA);
	cudaSafeCall( cudaGraphicsUnmapResources(1, &_apResourceRGBPxielBO[usPyrLevel_], 0) );
	/*//deprecated
	void *pDev;
	cudaSafeCall( cudaGLMapBufferObject((void**)&pDev, _auRGBPixelBO[usPyrLevel_]) );
	cv::gpu::GpuMat cvgmRGB( btl::kinect::__aKinectH[usPyrLevel_], btl::kinect::__aKinectW[usPyrLevel_], CV_8UC3, pDev);
	cvgmRGB_.copyTo(cvgmRGB);
	cudaSafeCall( cudaGLUnmapBufferObject(_auRGBPixelBO[usPyrLevel_]) );*/
	//texture mapping
	
	glBindTexture( GL_TEXTURE_2D, _auTexture[usPyrLevel_]);
	glBindBuffer ( GL_PIXEL_UNPACK_BUFFER_ARB, _auRGBPixelBO[usPyrLevel_]);

	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, btl::kinect::__aKinectW[usPyrLevel_], btl::kinect::__aKinectH[usPyrLevel_], 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	//glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, btl::kinect::__aKinectW[usPyrLevel_], btl::kinect::__aKinectH[usPyrLevel_], GL_RGBA, GL_FLOAT, 0);
	{
		GLenum eError = glGetError();
		if (eError != GL_NO_ERROR)
		{
			switch(eError){
			case GL_INVALID_ENUM:
				PRINTSTR("GL_INVALID_ENUM");break;
			case GL_INVALID_VALUE:
				PRINTSTR("GL_INVALID_VALUE");break;
			case GL_INVALID_OPERATION:
				PRINTSTR("GL_INVALID_OPERATION");break;
			case GL_STACK_OVERFLOW:
				PRINTSTR("GL_STACK_OVERFLOW");break;
			case GL_STACK_UNDERFLOW:
				PRINTSTR("GL_STACK_UNDERFLOW");break;
			case GL_OUT_OF_MEMORY:
				PRINTSTR("GL_OUT_OF_MEMORY");break;
			}
		}
	}
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

}
void CGLUtil::renderPatternGL(const float fSize_, const unsigned short usRows_, const unsigned short usCols_ ) const
{
	GLboolean bLightIsOn;
	glGetBooleanv(GL_LIGHTING,&bLightIsOn);
	if (bLightIsOn){
		glDisable(GL_LIGHTING);
	}
	
	const float usStartZ = -usRows_/2*fSize_;
	const float usEndZ =    usRows_/2*fSize_;
	const float usStartX = -usCols_/2*fSize_;
	const float usEndX   =  usCols_/2*fSize_;
	glLineWidth(.01f);
	glPushMatrix();
	glColor3f ( .4f , .4f , .4f );
	glBegin ( GL_LINES );
	//render rows
	for ( unsigned short r = 0; r <= usRows_; r++ ){
		glVertex3f ( usStartX,  0, usStartZ+r*fSize_ );
		glVertex3f ( usEndX,    0, usStartZ+r*fSize_ );
	}
	//render cols
	for ( unsigned short c = 0; c <= usCols_; c++ ){
		glVertex3f ( usStartX+c*fSize_,  0, usStartZ );
		glVertex3f ( usStartX+c*fSize_,  0, usEndZ );
	}
	glEnd();
	glPopMatrix();

	if (bLightIsOn){
		glEnable(GL_LIGHTING);
	}
	return;
}
void CGLUtil::renderVoxelGL( const float fSize_) const
{
	
	// x axis
	glColor3f ( 1.f, .0f, .0f );
	//top
	glBegin ( GL_LINE_LOOP );
	glVertex3f ( 0.f,    0.f, 0.f ); 
	glVertex3f ( fSize_, 0.f, 0.f ); 
	glVertex3f ( fSize_, 0.f, fSize_ ); 
	glVertex3f ( 0.f,    0.f, fSize_ ); 
	glEnd();
	//bottom
	glBegin ( GL_LINE_LOOP );
	glVertex3f ( 0.f,    fSize_, 0.f ); 
	glVertex3f ( fSize_, fSize_, 0.f ); 
	glVertex3f ( fSize_, fSize_, fSize_ ); 
	glVertex3f ( 0.f,    fSize_, fSize_ ); 
	glEnd();
	//middle
	glBegin ( GL_LINES );
	glVertex3f ( 0.f,    0.f, 0.f ); 
	glVertex3f ( 0.f,    fSize_, 0.f );
	glEnd();
	glBegin ( GL_LINES );
	glVertex3f ( fSize_, 0.f, 0.f ); 
	glVertex3f ( fSize_, fSize_, 0.f );
	glEnd();
	glBegin ( GL_LINES );
	glVertex3f ( fSize_, 0.f,   fSize_ ); 
	glVertex3f ( fSize_, fSize_,fSize_ );
	glEnd();
	glBegin ( GL_LINES );
	glVertex3f ( 0.f, 0.f,    fSize_ ); 
	glVertex3f ( 0.f, fSize_, fSize_ );
	glEnd();
	/*//top
	float fHS = fSize_/2.f;
	glBegin ( GL_LINE_LOOP );
	glVertex3f ( fHS, fHS, fHS ); 
	glVertex3f ( fHS, fHS,-fHS ); 
	glVertex3f (-fHS, fHS,-fHS ); 
	glVertex3f (-fHS, fHS, fHS ); 
	glEnd();
	//bottom
	glBegin ( GL_LINE_LOOP );
	glVertex3f ( fHS,-fHS, fHS ); 
	glVertex3f ( fHS,-fHS,-fHS ); 
	glVertex3f (-fHS,-fHS,-fHS ); 
	glVertex3f (-fHS,-fHS, fHS ); 
	glEnd();
	//middle
	glBegin ( GL_LINES );
	glVertex3f ( fHS, fHS, fHS ); 
	glVertex3f ( fHS,-fHS, fHS ); 
	glEnd();
	glBegin ( GL_LINES );
	glVertex3f ( fHS, fHS,-fHS ); 
	glVertex3f ( fHS,-fHS,-fHS ); 
	glEnd();
	glBegin ( GL_LINES );
	glVertex3f (-fHS, fHS,-fHS ); 
	glVertex3f (-fHS,-fHS,-fHS ); 
	glEnd();
	glBegin ( GL_LINES );
	glVertex3f (-fHS, fHS, fHS ); 
	glVertex3f (-fHS,-fHS, fHS ); 
	glEnd();*/
}

void CGLUtil::timerStart(){
	// timer on
	_cT0 =  boost::posix_time::microsec_clock::local_time(); 
}
void CGLUtil::timerStop(){
	// timer off
	_cT1 =  boost::posix_time::microsec_clock::local_time(); 
	_cTDAll = _cT1 - _cT0 ;
	_fFPS = 1000.f/_cTDAll.total_milliseconds();
	PRINT( _fFPS );
}

}//gl_util
}//btl