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
public:
	CGLUtil(btl::utility::tp_coordinate_convention eConvention_ = btl::utility::BTL_GL);
	void clearColorDepth();
	void init();
	void viewerGL();
	void renderAxisGL() const;
	void renderPatternGL(const float fSize_, const unsigned short usRows_, const unsigned short usCols_ ) const;
	// Esc: exit; g: zoom in; h:zoom out; 0: set to default position
	//   <: rotate around Y; >:rotate around Y
	void normalKeys ( unsigned char key, int x, int y );
	void mouseClick ( int nButton_, int nState_, int nX_, int nY_ );
	void mouseMotion ( int nX_, int nY_ );

	template< typename T >
	void renderDisk(const T& x, const T& y, const T& z, const T& dNx, const T& dNy, const T& dNz, 
		const unsigned char* pColor_, const T& dSize_, bool bRenderNormal_ );

private:
	GLuint _uDisk;
	GLuint _uNormal;
	GLUquadricObj *_pQObj;

	Eigen::Vector3d _eivCentroid;
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
};

template< typename T >
void CGLUtil::renderDisk(const T& x, const T& y, const T& z, const T& dNx, const T& dNy, const T& dNz, const unsigned char* pColor_, const T& dSize_, bool bRenderNormal_ )
{
	glColor3ubv( pColor_ );

	glPushMatrix();
	glTranslatef( x, y, z );

	if( fabs(dNx) + fabs(dNy) + fabs(dNz) < 0.00001 ) // normal is not computed
	{
		//PRINT( dNz );
		return;
	}

	T dA = atan2(dNx,dNz);
	T dxz= sqrt( dNx*dNx + dNz*dNz );
	T dB = atan2(dNy,dxz);

	glRotatef(-dB*180 / M_PI,1,0,0 );
	glRotatef( dA*180 / M_PI,0,1,0 );
	T dR = -z/0.5;
	glScalef( dR*dSize_, dR*dSize_, dR*dSize_ );
	glCallList(_uDisk);
	if( bRenderNormal_ )
	{
		glCallList(_uNormal);
	}
	glPopMatrix();
};

}//gl_util
}//btl

#endif