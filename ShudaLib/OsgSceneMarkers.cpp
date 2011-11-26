#include "OsgSceneMarkers.hpp"

namespace btl
{
namespace extra
{
namespace gui
{

OsgCameraMarker::OsgCameraMarker(double fov, double aspect, double size)
{
   _init(fov, aspect, size, osg::Vec4(1.0, 1.0, 1.0, 1.0));
}

OsgCameraMarker::OsgCameraMarker(double fov, double aspect, double size, const osg::Vec4& colour)
{
   _init(fov, aspect, size, colour);
}

OsgCameraMarker::OsgCameraMarker(const Matrix3d& mK_,   const Vector2i& vImgResolution_, const double& dPhysicalFocalLength_, const osg::Vec4& vColour_)
{
   OsgCameraMarker(mK_, vImgResolution_(0), vImgResolution_(1) , dPhysicalFocalLength_, vColour_); 
}

OsgCameraMarker::OsgCameraMarker(const Matrix3d& mK_,   const double& dWidth_, const double& dHeight_, const double& dPhysicalFocalLength_, const osg::Vec4& vColour_)
{
   double u = mK_(0,2);
   double v = mK_(1,2);
   double f = ( mK_(0,0) + mK_(1,1) )/2.;

   _init(u, v, f, dWidth_, dHeight_, dPhysicalFocalLength_, vColour_); 
}

OsgCameraMarker::OsgCameraMarker(const double& u, const double& v, const double& f, const double& dWidth_, const double& dHeight_, const double& dPhysicalFocalLength_, const osg::Vec4& vColour_)
{
   _init( u,  v,  f,  dWidth_,  dHeight_,  dPhysicalFocalLength_, vColour_);
}

void OsgCameraMarker::_init(const double& u, const double& v, const double& f, const double& dWidth_,const double& dHeight_,const double& size, const osg::Vec4& colour)
{

   double dLeft, dRight, dBottom, dTop;
   //Two assumptions:
   //1. assuming the principle point is inside the image
   //2. assuming the x axis pointing right and y axis pointing downwards. principle point at ( u, v )
   //first compute the tan() in four directions 
   dTop    =             v   /f;
   dBottom = ( v - dHeight_ )/f;
   dLeft   =           - u   /f;
   dRight  = ( dWidth_ - u  )/f;
   // then compute the physical distance of top, bottom, left and right.
   // dSize is actually the physical length of focal length. Usually, focal length is measured in pixels, 
   // but here dSize, the physical focal length, is measured as the same unit as opengl.
   dTop   *= size;    
   dBottom*= size;
   dLeft  *= size;
   dRight *= size;
 
   osg::Vec3Array* frameVerts = new osg::Vec3Array;
   frameVerts->push_back(osg::Vec3(0, 0, 0)); //principle point
   frameVerts->push_back(osg::Vec3(dTop,   dLeft , size));
   frameVerts->push_back(osg::Vec3(dTop,   dRight, size));
   frameVerts->push_back(osg::Vec3(dBottom,dRight, size));
   frameVerts->push_back(osg::Vec3(dBottom,dLeft,  size));
   frameVerts->push_back(osg::Vec3(0,           0, size));
   frameVerts->push_back(osg::Vec3( -u,         0, size));
   frameVerts->push_back(osg::Vec3(0,           v, size));
   frameVerts->push_back(osg::Vec3(0,        0,  2*size));

   this->setVertexArray(frameVerts);

   const osg::Vec4 frameFCol = colour;
   const osg::Vec4 frameBCol = osg::Vec4(colour.r(), colour.g(), colour.b(), 0.6*colour.a());
   osg::Vec4Array* frameColours = new osg::Vec4Array;
   frameColours->push_back(frameBCol);
   frameColours->push_back(frameBCol);
   frameColours->push_back(frameBCol);
   frameColours->push_back(frameBCol);
   frameColours->push_back(frameFCol);
   frameColours->push_back(frameFCol);
   frameColours->push_back(frameFCol);
   frameColours->push_back(frameFCol);
   frameColours->push_back(frameFCol);
   frameColours->push_back(frameFCol);
   frameColours->push_back(frameFCol);
   this->setColorArray(frameColours);
   this->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE);

   osg::DrawElementsUInt* frameLines =
      new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
   frameLines->push_back(0); frameLines->push_back(1);
   frameLines->push_back(0); frameLines->push_back(2);
   frameLines->push_back(0); frameLines->push_back(3);
   frameLines->push_back(0); frameLines->push_back(4);
   frameLines->push_back(1); frameLines->push_back(2);
   frameLines->push_back(2); frameLines->push_back(3);
   frameLines->push_back(3); frameLines->push_back(4);
   frameLines->push_back(4); frameLines->push_back(1);
   frameLines->push_back(5); frameLines->push_back(6);
   frameLines->push_back(5); frameLines->push_back(7);
   frameLines->push_back(0); frameLines->push_back(8);
   this->addPrimitiveSet(frameLines);
}

void OsgCameraMarker::_init(double fov, double aspect, double size, const osg::Vec4& colour)
{
   const double extentx = size * tan(fov / 2.0);
   const double extenty = extentx / aspect;

   osg::Vec3Array* frameVerts = new osg::Vec3Array;
   frameVerts->push_back(osg::Vec3(0, 0, 0));
   frameVerts->push_back(osg::Vec3(-extentx, -extenty, size));
   frameVerts->push_back(osg::Vec3(extentx, -extenty, size));
   frameVerts->push_back(osg::Vec3(extentx,  extenty, size));
   frameVerts->push_back(osg::Vec3(-extentx,  extenty, size));
   frameVerts->push_back(osg::Vec3(0,        0, size));
   frameVerts->push_back(osg::Vec3(extentx,        0, size));
   frameVerts->push_back(osg::Vec3(0,  extenty, size));
    frameVerts->push_back(osg::Vec3(0,        0,2*size));

   this->setVertexArray(frameVerts);

   const osg::Vec4 frameFCol = colour;
   const osg::Vec4 frameBCol = osg::Vec4(colour.r(), colour.g(), colour.b(), 0.6*colour.a());
   osg::Vec4Array* frameColours = new osg::Vec4Array;
   frameColours->push_back(frameBCol);
   frameColours->push_back(frameBCol);
   frameColours->push_back(frameBCol);
   frameColours->push_back(frameBCol);
   frameColours->push_back(frameFCol);
   frameColours->push_back(frameFCol);
   frameColours->push_back(frameFCol);
   frameColours->push_back(frameFCol);
   frameColours->push_back(frameFCol);
   frameColours->push_back(frameFCol);
   frameColours->push_back(frameFCol);
   this->setColorArray(frameColours);
   this->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE);

   osg::DrawElementsUInt* frameLines =
      new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
   frameLines->push_back(0); frameLines->push_back(1);
   frameLines->push_back(0); frameLines->push_back(2);
   frameLines->push_back(0); frameLines->push_back(3);
   frameLines->push_back(0); frameLines->push_back(4);
   frameLines->push_back(1); frameLines->push_back(2);
   frameLines->push_back(2); frameLines->push_back(3);
   frameLines->push_back(3); frameLines->push_back(4);
   frameLines->push_back(4); frameLines->push_back(1);
   frameLines->push_back(5); frameLines->push_back(6);
   frameLines->push_back(5); frameLines->push_back(7);
   frameLines->push_back(0); frameLines->push_back(8);
   this->addPrimitiveSet(frameLines);
}

OsgCoordinateFrameMarker::OsgCoordinateFrameMarker(double size)
{
   osg::Vec3Array* frameVerts = new osg::Vec3Array;
   frameVerts->push_back(osg::Vec3(0, 0, 0));
   frameVerts->push_back(osg::Vec3(size, 0, 0));
   frameVerts->push_back(osg::Vec3(0, size, 0));
   frameVerts->push_back(osg::Vec3(0, 0, size));
   this->setVertexArray(frameVerts);

   osg::Vec4Array* frameColours = new osg::Vec4Array;
   frameColours->push_back(osg::Vec4(1.0, 0.0, 0.0, 1.0));
   frameColours->push_back(osg::Vec4(0.0, 1.0, 0.0, 1.0));
   frameColours->push_back(osg::Vec4(0.0, 0.0, 1.0, 1.0));
   this->setColorArray(frameColours);
   this->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE);

   osg::DrawElementsUInt* frameLines =
      new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
   frameLines->push_back(0); frameLines->push_back(1);
   frameLines->push_back(0); frameLines->push_back(2);
   frameLines->push_back(0); frameLines->push_back(3);
   this->addPrimitiveSet(frameLines);
}

}
}
} // namespace btl::extra::gui
