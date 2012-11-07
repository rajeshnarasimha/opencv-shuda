#ifndef BTL_GL_UTIL
#define BTL_GL_UTIL
/**
* @file GLUtil.h
* @brief opengl rendering utilities
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.0
* @date 2012-02-16
*/

namespace btl{	namespace gl_util
{
class CGLUtil
{
public:
	//type
	typedef boost::shared_ptr<CGLUtil> tp_shared_ptr;
	typedef CGLUtil* tp_ptr;
public:
	CGLUtil(ushort uResolution_, ushort uPyrLevel_,btl::utility::tp_coordinate_convention eConvention_ = btl::utility::BTL_GL);
	void clearColorDepth();
	void init();
	//to initialize for the interoperation with opengl
	void setCudaDeviceForGLInteroperation();

	void renderVoxelGL( const float fSize_) const;
	void renderAxisGL() const;
	void renderPatternGL(const float fSize_, const unsigned short usRows_, const unsigned short usCols_ ) const;
	// Esc: exit; g: zoom in; h:zoom out; 0: set to default position
	//   <: rotate around Y; >:rotate around Y
	void normalKeys ( unsigned char key, int x, int y );
	void specialKeys( int key, int x, int y );
	void mouseClick ( int nButton_, int nState_, int nX_, int nY_ );
	void mouseMotion ( int nX_, int nY_ );

	template< typename T >
	void renderDisk(const T& x, const T& y, const T& z, const T& dNx, const T& dNy, const T& dNz, 
		const unsigned char* pColor_, const T& dSize_, bool bRenderNormal_ );
	template< typename T > 
	void renderDiskFastGL(const T& x, const T& y, const T& z, const T& tAngle_, const T& dNx, const T& dNy,  
		const unsigned char* pColor_, const T& dSize_, bool bRenderNormal_ );

	template< typename T >
	void renderOctTree(const T& x, const T& y, const T& z, const T& dSize_, const unsigned short sLevel_ ) const;
	template< typename T >
	void renderVoxel( const T& x, const T& y, const T& z, const T& dSize_ ) const;
	void timerStart();
	void timerStop();
	//create vertex buffer
	void createVBO(const unsigned int uRows, const unsigned int uCols_, const unsigned short usChannel_, const unsigned short usBytes_, GLuint* puVBO_, cudaGraphicsResource** ppResourceVBO_ );
	void releaseVBO( GLuint uVBO_, cudaGraphicsResource* pResourceVBO_ );
	//create pixel buffer
	void createPBO(const unsigned int uRows_, const unsigned int uCols_, const unsigned short usChannel_, const unsigned short usBytes_, GLuint* puPBO_, cudaGraphicsResource** ppResourcePixelBO_, GLuint* pTexture_);
	void releasePBO( GLuint uPBO_,cudaGraphicsResource *pResourcePixelBO_ );
	void constructVBOsPBOs();
	void destroyVBOsPBOs();
	void gpuMapPtResources(const cv::gpu::GpuMat& cvgmPts_, const ushort usPyrLevel_);
	void gpuMapNlResources(const cv::gpu::GpuMat& cvgmNls_, const ushort usPyrLevel_);
	void gpuMapRGBResources(const cv::gpu::GpuMat& cvgmNls_, const ushort usPyrLevel_);
	void gpuMapRgb2PixelBufferObj(const cv::gpu::GpuMat& cvgmRGBs_, const ushort usPyrLevel_ );
	void errorDetectorGL() const;
	void viewerGL();
	void setInitialPos();
	void getRTFromWorld2CamCV(Eigen::Matrix3f* pRw_, Eigen::Vector3f* pTw_) const;

public:
	Eigen::Matrix4f _eimModelViewGL; //model view transformation matrix in GL convention.
	//double _adModelViewGL[16];
	float _fSize; //disk size
	bool _bRenderNormal;
	bool _bEnableLighting;
	bool _bDisplayCamera;
	bool _bRenderReference;
	bool _bCtlDown;
	unsigned short _usPyrHeight;
	ushort _usLevel;
	ushort _uResolution;
	//Cuda OpenGl interoperability
	GLuint _auPtVBO[4];
	cudaGraphicsResource* _apResourcePtVBO[4];
	GLuint _auNlVBO[4];
	cudaGraphicsResource* _apResourceNlVBO[4];
	GLuint _auRGBVBO[4];
	cudaGraphicsResource* _apResourceRGBVBO[4];
	GLuint _auRGBPixelBO[4];
	cudaGraphicsResource* _apResourceRGBPxielBO[4];
	GLuint _auTexture[4];
private:
	GLuint _uDisk;
	GLuint _uNormal;
	GLuint _uVoxel;
	GLuint _uOctTree;
	GLUquadricObj *_pQObj;

	float _aCentroid[3];
	double _dZoom;
	double _dZoomLast;
	double _dScale;

	double _dXAngle;
	double _dYAngle;
	double _dXLastAngle;
	double _dYLastAngle;
	double _dX;
	double _dY;
	double _dXLast;
	double _dYLast;

	int  _nXMotion;
	int  _nYMotion;
	int  _nXLeftDown, _nYLeftDown;
	int  _nXRightDown, _nYRightDown;
	bool _bLButtonDown;
	bool _bRButtonDown;

	GLfloat _aLight[4];

	btl::utility::tp_coordinate_convention _eConvention;

	//timer
	boost::posix_time::ptime _cT0, _cT1;
	boost::posix_time::time_duration _cTDAll;
	float _fFPS;//frame per second
};//CGLUtil

template< typename T >
void btl::gl_util::CGLUtil::renderVoxel( const T& x, const T& y, const T& z, const T& dSize_ ) const
{
	glColor3f( 0.f,0.f,1.f );
	glPushMatrix();
	glTranslatef( x, y, z );
	glScalef( dSize_, dSize_, dSize_ );
	glCallList(_uVoxel);
	glPopMatrix();
};

template< typename T >
void CGLUtil::renderDisk(const T& x, const T& y, const T& z, const T& dNx, const T& dNy, const T& dNz, const unsigned char* pColor_, const T& dSize_, bool bRenderNormal_ )
{
	glColor3ubv( pColor_ );

	glPushMatrix();
	glTranslatef( x, y, z );

	float fAx,fAy,fA;
	fAx =-dNy; //because of cv-convention
	fAy = dNx;
	//normalization
	float norm = sqrtf(fAx*fAx + fAy*fAy );
	if( norm < 1.0e-10 ) return;
	fAx /= norm;
	fAy /= norm;
	fA = asin(norm)*180.f/M_PI;
	glRotatef( fA,fAx,fAy,0 );

	//T dA = atan2(dNx,dNz);
	//T dxz= sqrt( dNx*dNx + dNz*dNz );
	//T dB = atan2(dNy,dxz);

	//glRotatef(-dB*180 / M_PI,1,0,0 );
	//glRotatef( dA*180 / M_PI,0,1,0 );

	T dR = -z*2; //the further the disk the larger the size
	glScalef( dR*dSize_, dR*dSize_, dR*dSize_ );
	glCallList(_uDisk);
	if( bRenderNormal_ )
	{
		glCallList(_uNormal);
	}
	glPopMatrix();
};

template< typename T >
void CGLUtil::renderDiskFastGL(const T& x, const T& y, const T& z, const T& tAngle_, const T& dNx, const T& dNy, const unsigned char* pColor_, const T& dSize_, bool bRenderNormal_ )
{
	glColor3ubv( pColor_ );
	glPushMatrix();
	glTranslatef( x, y, z );
	//cross product
	glRotatef( tAngle_,dNx,dNy,0.f );
	T dR = -z*2; //the further the disk the larger the size
	glScalef( dR*dSize_, dR*dSize_, dR*dSize_ );
	glCallList(_uDisk);
	if( bRenderNormal_ )
	{glCallList(_uNormal);}
	glPopMatrix();
};

template< typename T >
void CGLUtil::renderOctTree(const T& x, const T& y, const T& z, const T& dSize_, const unsigned short sLevel_ ) const{
	if( 0==sLevel_ ) {renderVoxel<T>( x, y, z, dSize_);return;}
	//render next level 
	T tCx,tCy,tCz, tS = dSize_/4, tL = dSize_/2;
	tCx = x + tS;  tCy = y + tS;  tCz = z + tS;
	renderOctTree<T>(tCx,tCy,tCz,tL, sLevel_-1 );
	tCx = x + tS;  tCy = y + tS;  tCz = z - tS;
	renderOctTree<T>(tCx,tCy,tCz,tL, sLevel_-1 );
	tCx = x + tS;  tCy = y - tS;  tCz = z + tS;
	renderOctTree<T>(tCx,tCy,tCz,tL, sLevel_-1 );
	tCx = x - tS;  tCy = y + tS;  tCz = z + tS;
	renderOctTree<T>(tCx,tCy,tCz,tL, sLevel_-1 );

	tCx = x - tS;  tCy = y - tS;  tCz = z - tS;
	renderOctTree<T>(tCx,tCy,tCz,tL, sLevel_-1 );
	tCx = x - tS;  tCy = y - tS;  tCz = z + tS;
	renderOctTree<T>(tCx,tCy,tCz,tL, sLevel_-1 );
	tCx = x - tS;  tCy = y + tS;  tCz = z - tS;
	renderOctTree<T>(tCx,tCy,tCz,tL, sLevel_-1 );
	tCx = x + tS;  tCy = y - tS;  tCz = z - tS;
	renderOctTree<T>(tCx,tCy,tCz,tL, sLevel_-1 );
};	

}//gl_util
}//btl

#endif