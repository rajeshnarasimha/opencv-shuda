//#define INFO
//#define TIMER

//#define INFO
#include <boost/shared_ptr.hpp>
#include <vector>
#include <Eigen/Core>
#include <GL/freeglut.h>
#include "OtherUtil.hpp"
#include "GLUtil.h"

namespace btl{	namespace gl_util
{
CGLUtil::CGLUtil(btl::utility::tp_coordinate_convention eConvention_ /*= btl::utility::BTL_GL*/){
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
	if( btl::utility::BTL_GL == eConvention_ )
		_eivCentroid << 0, 0, -1;
	else if( btl::utility::BTL_CV == eConvention_ )
		_eivCentroid << 0, 0,  1;
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
		PRINT(_dZoom);
	}

	glutPostRedisplay();
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
		PRINT( _dZoom );
		break;
	case 'h':
		//zoom out
		glDisable( GL_BLEND );
		_dZoom -= _dScale;
		glutPostRedisplay();
		PRINT( _dZoom );
		break;
	case '<':
		_dYAngle += 1.0;
		glutPostRedisplay();
		break;
	case '>':
		_dYAngle -= 1.0;
		glutPostRedisplay();
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
	glTranslated( _eivCentroid(0), _eivCentroid(1), _eivCentroid(2) ); // 5. translate back to the original camera pose
	_dZoom = _dZoom < 0.1? 0.1: _dZoom;
	_dZoom = _dZoom > 10? 10: _dZoom;
	glScaled( _dZoom, _dZoom, _dZoom );                          // 4. zoom in/out
	glRotated ( _dXAngle, 0, 1 ,0 );                             // 3. rotate horizontally
	glRotated ( _dYAngle, 1, 0 ,0 );                             // 2. rotate vertically
	glTranslated( -_eivCentroid(0),-_eivCentroid(1),-_eivCentroid(2)); // 1. translate the world origin to align with object centroid
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
	_uDisk = glGenLists(1);
	_pQObj = gluNewQuadric();
	gluQuadricDrawStyle(_pQObj, GLU_FILL); //LINE); /* wireframe */
	gluQuadricNormals(_pQObj, GLU_SMOOTH);// FLAT);//
	glNewList(_uDisk, GL_COMPILE);
	gluDisk(_pQObj, 0.0, 0.01, 9, 1);
	glEndList();

	_uNormal = glGenLists(2);
	glNewList(_uNormal, GL_COMPILE);
	glDisable(GL_LIGHTING);
	glBegin(GL_LINES);
	glColor3d(1.,0.,0.);
	glVertex3d(0.,0.,0.);
	glVertex3d(0.,0.,0.016);
	glEnd();
	glEndList();
}

void CGLUtil::renderPatternGL(const float fSize_, const unsigned short usRows_, const unsigned short usCols_ ) const
{
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
	return;
}


}//gl_util
}//btl