#ifndef BTL_EXTRA_GUI_OSGSCENEMARKERS
#define BTL_EXTRA_GUI_OSGSCENEMARKERS

#include <Eigen/Dense>
#include <osg/Group>
#include <osg/Camera>
/*
 * btl::extra::gui::COsgCameraMarker
 * btl::extra::gui::COsgCoordinateFrameMarker
 * btl::extra::gui::COsgChessboardMarker
 */

namespace btl
{
namespace extra
{
namespace gui
{

using namespace Eigen;

class COsgCameraMarker : public osg::Group
{

public:
    /**
     * @brief
     *
     * @param fov is the vertical feild of view in radian
     * @param aspect is the ratio of W/H
     */
    COsgCameraMarker ( double fov, double aspect, double size );
    COsgCameraMarker ( double fov, double aspect, double size, const osg::Vec4& colour );
    /**
    * @brief
    *
    * @param mK_
    * @param vImgResolution_: (0) width, cols; (1) height rows
    * @param dPhysicalFocalLength_
    * @param vColour_
    */
    COsgCameraMarker ( const Matrix3d& mK_,   const Vector2i& vImgResolution_, const double& dPhysicalFocalLength_, const osg::Vec4& vColour_ );
    COsgCameraMarker ( const Matrix3d& mK_,   const double& dWidth_, const double& dHeight_, const double& dPhysicalFocalLength_, const osg::Vec4& vColour_ );
    /** 
     * @brief
     *
     * @param u,v are the principle point coordinate and must be larger than 0. The image coordinate system is 
     *            defined as left and downward are positive.
     * @param f is the focal-length must be larger than zero.
     * @param dWidth_, dHeight_ are the width and height of the input image.
     * @param dPhysicalFocalLength_ defines the size of the virtual camera.
     * @param vColour_ is the color of the virtual camera frame.
     */
    COsgCameraMarker ( const double& u, const double& v, const double& f, const double& dWidth_, const double& dHeight_, const double& dPhysicalFocalLength_, const osg::Vec4& vColour_ );


private:

    void __init ( double fov, double aspect, double size, const osg::Vec4& colour );
    void __init ( const double& u, const double& v, const double& f, const double& dWidth_, const double& dHeight_, const double& size, const osg::Vec4& colour );
};

class COsgCoordinateFrameMarker : public osg::Group
{
public:    
    COsgCoordinateFrameMarker ( float fSize_ ); // the size of the coordinate system marker
private:
    osg::Group* __createAnAxis( const osg::Vec4& vColor_ );

    float _fCylinderLength;
    float _fCylinderRadius;
    float _fConeRadius;
    float _fConeLength;
    float _fOrigRadius;
    float _fLineLength;
};


class COsgChessboardMarker : public osg::Group
{
public:
    COsgChessboardMarker(const float& fSize_,const unsigned int& uRows_,const unsigned int& uCols_);
private:
    
};

class COsgCamera : public osg::Camera
{
public:
    COsgCamera( const Matrix3d& mK_, const double& dWidth_, const double& dHeight_, const double& dNear_=.01, const double dFar_=20. );
    COsgCamera( const double& u, const double& v, const double& f, const double& dWidth_, const double& dHeight_, const double& dNear_=.01, const double dFar_=20. );
    COsgCamera( const double& dLeft_, const double& dRight_, const double& dTop_, const double& dBottom_, const double& dWidth_, const double& dHeight_, const double& dNear_=.01, const double dFar_=20. );
private:
    void __setupWindowView( const Matrix3d& mK_, const double& dWidth_, const double& dHeight_, const double& dNear_, const double dFar_ );
    void __setupWindowView(const double& u, const double& v, const double& f, const double& dWidth_, const double& dHeight_, const double& dNear_, const double dFar_ );
    void __setupWindowView(const double& dLeft_, const double& dRight_, const double& dTop_, const double& dBottom_, const double& dWidth_, const double& dHeight_, const double& dNear_, const double dFar_ );

};


}
}
} // namespace btl::extra::gui

#endif // BTL_EXTRA_GUI_OSGCAMERAMARKER
