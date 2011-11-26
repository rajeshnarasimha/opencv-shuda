#ifndef BTL_EXTRA_GUI_OSGSCENEMARKERS
#define BTL_EXTRA_GUI_OSGSCENEMARKERS

#include <osg/Geometry>
#include <Eigen/Dense>

namespace btl
{
namespace extra
{
namespace gui
{

using namespace Eigen;

class OsgCameraMarker : public osg::Geometry
{

public:
    OsgCameraMarker ( double fov, double aspect, double size );
    OsgCameraMarker ( double fov, double aspect, double size, const osg::Vec4& colour );
    /**
    * @brief
    *
    * @param mK_
    * @param vImgResolution_: (0) width, cols; (1) height rows
    * @param dPhysicalFocalLength_
    * @param vColour_
    */
    OsgCameraMarker ( const Matrix3d& mK_,   const Vector2i& vImgResolution_, const double& dPhysicalFocalLength_, const osg::Vec4& vColour_ );
    OsgCameraMarker ( const Matrix3d& mK_,   const double& dWidth_, const double& dHeight_, const double& dPhysicalFocalLength_, const osg::Vec4& vColour_ );
    OsgCameraMarker ( const double& u, const double& v, const double& f, const double& dWidth_, const double& dHeight_, const double& dPhysicalFocalLength_, const osg::Vec4& vColour_ );


private:

    void _init ( double fov, double aspect, double size, const osg::Vec4& colour );
    void _init ( const double& u, const double& v, const double& f, const double& dWidth_, const double& dHeight_, const double& size, const osg::Vec4& colour );
};

class OsgCoordinateFrameMarker : public osg::Geometry
{
public:
    OsgCoordinateFrameMarker ( double size );
};

}
}
} // namespace btl::extra::gui

#endif // BTL_EXTRA_GUI_OSGCAMERAMARKER
