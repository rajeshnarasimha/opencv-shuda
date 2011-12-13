#include "OsgSceneMarkers.hpp"

#include <osg/Geometry>
#include <osg/ShapeDrawable>
#include <osg/Geode>
#include <osg/LineWidth>
#include <osg/MatrixTransform>

#include <math.h>

using namespace std;

namespace btl
{
namespace extra
{
namespace gui
{

COsgCameraMarker::COsgCameraMarker(double fov, double aspect, double size)
{
    __init(fov, aspect, size, osg::Vec4(1.0, 1.0, 1.0, 1.0));
}

COsgCameraMarker::COsgCameraMarker(double fov, double aspect, double size, const osg::Vec4& colour)
{
    __init(fov, aspect, size, colour);
}

COsgCameraMarker::COsgCameraMarker(const Matrix3d& mK_,   const Vector2i& vImgResolution_, const double& dPhysicalFocalLength_, const osg::Vec4& vColour_)
{
    COsgCameraMarker(mK_, vImgResolution_(0), vImgResolution_(1) , dPhysicalFocalLength_, vColour_);
}

COsgCameraMarker::COsgCameraMarker(const Matrix3d& mK_,   const double& dWidth_, const double& dHeight_, const double& dPhysicalFocalLength_, const osg::Vec4& vColour_)
{
    double u = mK_(0,2);
    double v = mK_(1,2);
    double f = ( mK_(0,0) + mK_(1,1) )/2.;

    __init(u, v, f, dWidth_, dHeight_, dPhysicalFocalLength_, vColour_);
}

COsgCameraMarker::COsgCameraMarker(const double& u, const double& v, const double& f, const double& dWidth_, const double& dHeight_, const double& dPhysicalFocalLength_, const osg::Vec4& vColour_)
{
    __init( u,  v,  f,  dWidth_,  dHeight_,  dPhysicalFocalLength_, vColour_);
}

void COsgCameraMarker::__init(double fov, double aspect, double size, const osg::Vec4& colour)
{
    double dWidth = aspect* size;
    double dHeight= size;
    double u = dWidth/2.;
    double v = dHeight/2.;
    double f = v/tan(fov/2.);
    __init(u, v, f, dWidth,dHeight,size, colour);
}

/*
 * size >= 0
 *
 * */
void COsgCameraMarker::__init(const double& u, const double& v, const double& f, const double& dWidth_,const double& dHeight_,const double& size, const osg::Vec4& colour)
{
    // the camera coordinate follows the convention of opencv:
    // x left
    // y down
    // z forward
    // right hand rule and origin at camera center
    // notice that the default openGL convention is
    // x left
    // y up
    // z backward
    // the default OSG convention is ???
    //
    osg::Group* pCoordinateFrameMarker = new COsgCoordinateFrameMarker(size);
    addChild( pCoordinateFrameMarker );

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

    osg::Geometry* pDrawable = new osg::Geometry;
    osg::Vec3Array* frameVerts = new osg::Vec3Array;
    frameVerts->push_back(osg::Vec3(0, 0, 0)); //projection centre                      0
    frameVerts->push_back(osg::Vec3(dLeft,  dTop, -size));//                            1
    frameVerts->push_back(osg::Vec3(dRight, dTop, -size));//                            2
    frameVerts->push_back(osg::Vec3(dRight,dBottom,-size));//                           3
    frameVerts->push_back(osg::Vec3(dLeft, dBottom,-size));//                           4
    frameVerts->push_back(osg::Vec3(0,           0, -size));//principle point           5
    frameVerts->push_back(osg::Vec3(dRight,      0, -size));//x positive                6
    frameVerts->push_back(osg::Vec3(0,     dBottom, -size));//y positive                7

    pDrawable->setVertexArray(frameVerts);

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
    pDrawable->setColorArray(frameColours);
    pDrawable->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE);

    osg::DrawElementsUInt* frameLines = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
    frameLines->push_back(0);
    frameLines->push_back(1);
    frameLines->push_back(0);
    frameLines->push_back(2);
    frameLines->push_back(0);
    frameLines->push_back(3);
    frameLines->push_back(0);
    frameLines->push_back(4);
    frameLines->push_back(1);
    frameLines->push_back(2);
    frameLines->push_back(2);
    frameLines->push_back(3);
    frameLines->push_back(3);
    frameLines->push_back(4);
    frameLines->push_back(4);
    frameLines->push_back(1);
    frameLines->push_back(5);
    frameLines->push_back(6);
    frameLines->push_back(5);
    frameLines->push_back(7);
    pDrawable->addPrimitiveSet(frameLines);

    osg::Geode* pCamGeode = new osg::Geode();
    pCamGeode->addDrawable(pDrawable);
    addChild( pCamGeode );
}


COsgCoordinateFrameMarker::COsgCoordinateFrameMarker(float fSize_)
{
    _fCylinderLength = fSize_;
    _fCylinderRadius = _fCylinderLength/100.f;// .1f;
    _fConeRadius = _fCylinderRadius*3.f;//.3f;
    _fConeLength = _fCylinderLength/5.f;
    _fOrigRadius = _fCylinderRadius*2.f;
    _fLineLength = (_fConeLength + _fCylinderLength); // besides the cylinder-cone style axis, a line is rendered for maintaining the correctness of the coordinate
    // being rendered is consistent with the OpenGL's native
    // coordinate convention.
//learn how to render point, line, and others
// a sphere denotes the origin geode
    osg::ref_ptr<osg::ShapeDrawable> pSphere = new osg::ShapeDrawable;//
    pSphere->setShape( new osg::Sphere(osg::Vec3(0.0f, 0.0f, 0.0f), _fOrigRadius) );
    pSphere->setColor( osg::Vec4(.8f, .8f, .8f, 1.f) );

    osg::Geode* pOriginGeode = new osg::Geode();
    pOriginGeode->addDrawable(pSphere);

/////////////////////////////////////////////////
    //X
    osg::ref_ptr<osg::Group> pXAxisGroup = __createAnAxis( osg::Vec4(1.f,.0f,.0f,1.f ) );
    // rotate and translate the Axis and cone together to align with x-axis
    // pTR represents the X-Axis group
    osg::ref_ptr<osg::MatrixTransform> pTR1 = new osg::MatrixTransform;
    {
        osg::Matrixf mR;
        mR.setRotate( osg::Quat(0., osg::X_AXIS, M_PI/2., osg::Y_AXIS, 0.,osg::Z_AXIS ) );
        osg::Matrixf mT;
        mT.setTrans( osg::Vec3(_fCylinderLength/2.f, .0f, .0f ) );
        osg::Matrixf mRT = mR * mT; // this equivalent to first rotate then translate according to experiment. ????
        pTR1->setMatrix(mRT);
        pTR1->addChild( pXAxisGroup );
    }
    //Y
    osg::ref_ptr<osg::Group> pYAxisGroup = __createAnAxis( osg::Vec4(.0f,1.f,.0f,1.f) );
    // rotate and translate the Axis and cone together to align with x-axis
    // pTR represents the X-Axis group
    osg::ref_ptr<osg::MatrixTransform> pTR2 = new osg::MatrixTransform;
    {
        osg::Matrixf mR;
        mR.setRotate( osg::Quat(-M_PI/2., osg::X_AXIS, 0., osg::Y_AXIS, 0.,osg::Z_AXIS ) );
        osg::Matrixf mT;
        mT.setTrans( osg::Vec3(.0f, _fCylinderLength/2.f, .0f ) );
        osg::Matrixf mRT = mR * mT; // this equivalent to first rotate then translate according to experiment. ????
        pTR2->setMatrix(mRT);
        pTR2->addChild( pYAxisGroup );
    }
    //Z
    osg::ref_ptr<osg::Group> pZAxisGroup = __createAnAxis( osg::Vec4(.0f,0.f,1.0f,1.f) );
    // rotate and translate the Axis and cone together to align with x-axis
    // pTR represents the X-Axis group
    osg::ref_ptr<osg::MatrixTransform> pTR3 = new osg::MatrixTransform;
    {
        osg::Matrixf mT;
        mT.setTrans( osg::Vec3(.0f, .0f, _fCylinderLength/2.f ) );
        pTR3->setMatrix(mT);
        pTR3->addChild( pZAxisGroup );
    }

/////////////////////////////////////////////////

//line-based axis geode
    osg::Geode* pLineAxisGeode = new osg::Geode();
    osg::Geometry* pAxisGeometry = new osg::Geometry();
    pLineAxisGeode->addDrawable(pAxisGeometry);

    osg::Vec3Array* pAxis = new osg::Vec3Array;
    pAxis->push_back( osg::Vec3(  0,0,0 ) );
    pAxis->push_back( osg::Vec3( _fLineLength,0,0 ) );
    pAxis->push_back( osg::Vec3( 0,_fLineLength,0 ) );
    pAxis->push_back( osg::Vec3( 0,0,_fLineLength ) );

    pAxisGeometry->setVertexArray( pAxis );
    //render x
    osg::DrawElementsUInt* pAxisX = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
    pAxisX->push_back(0);
    pAxisX->push_back(1);
    pAxisGeometry->addPrimitiveSet(pAxisX);
    //render y
    osg::DrawElementsUInt* pAxisY = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
    pAxisY->push_back(0);
    pAxisY->push_back(2);
    pAxisGeometry->addPrimitiveSet(pAxisY);
    //render z
    osg::DrawElementsUInt* pAxisZ = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
    pAxisZ->push_back(0);
    pAxisZ->push_back(3);
    pAxisGeometry->addPrimitiveSet(pAxisZ);
    // set line width
    osg::StateSet* pStateSet = new osg::StateSet;
    osg::LineWidth* pLineWidth = new osg::LineWidth();
    pLineWidth->setWidth(1.0f);
    pStateSet->setAttributeAndModes(pLineWidth,osg::StateAttribute::ON);
    pStateSet->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    pAxisGeometry->setStateSet(pStateSet);
    //color
    osg::Vec4Array* pAxisColors = new osg::Vec4Array;
    //pAxisColors->push_back(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f) ); //index 0 white
    pAxisColors->push_back(osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f) ); //index 1 red
    pAxisColors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f) ); //index 2 green
    pAxisColors->push_back(osg::Vec4(0.0f, 0.0f, 1.0f, 1.0f) ); //index 3 blue
    //The next step is to associate the array of colors with the geometry,
    //assign the color indices created above to the geometry and set the
    //binding mode to _PER_VERTEX.
    pAxisGeometry->setColorArray(pAxisColors);
    pAxisGeometry->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE);

    addChild(pLineAxisGeode);
    addChild(pOriginGeode);
    addChild(pTR1);
    addChild(pTR2);
    addChild(pTR3);

    return;
}

osg::Group* COsgCoordinateFrameMarker::__createAnAxis(const osg::Vec4& vColor_)
{
    osg::Group* pAxisGroup = new osg::Group;
    {
        // x-axis-cylinder red geode
        osg::ref_ptr<osg::Geode> pXAxisGeode = new osg::Geode;
        {
            osg::ref_ptr<osg::ShapeDrawable> pXAxis = new osg::ShapeDrawable;
            pXAxis->setShape( new osg::Cylinder(osg::Vec3(0.0f, 0.0f, 0.0f), _fCylinderRadius,_fCylinderLength) ); //center of the cylinder, r, height
            pXAxis->setColor( vColor_ );
            pXAxisGeode->addDrawable(pXAxis);
        }

        // x-axis-cone red geode
        osg::ref_ptr<osg::Geode> pXConeGeode = new osg::Geode;
        {
            osg::ref_ptr<osg::ShapeDrawable> pXCone = new osg::ShapeDrawable;
            pXCone->setShape( new osg::Cone(osg::Vec3(0.0f, 0.0f, 0.0f), _fConeRadius,_fConeLength) ); //center of the cylinder, r, height
            pXCone->setColor( vColor_ );
            pXConeGeode->addDrawable(pXCone);
        }

        //translate the cone onto the top of the
        osg::ref_ptr<osg::MatrixTransform> pT = new osg::MatrixTransform;
        osg::Matrixf mT;
        mT.setTrans(osg::Vec3(.0f,.0f, _fCylinderLength/2.f));
        pT->setMatrix( mT );
        pT->addChild(pXConeGeode);//this should transform the pframidGeode to the next location-shuda

        pAxisGroup->addChild( pT );//add cone
        pAxisGroup->addChild( pXAxisGeode );//add cylinder
    }
    return pAxisGroup;
}

COsgChessboardMarker::COsgChessboardMarker(const float& fMetersPerGrid_,const unsigned int& uRows_,const unsigned int& uCols_)
{
    osg::ref_ptr<osg::Geode> pChessboardGeode = new osg::Geode();
    osg::ref_ptr<osg::Geometry> pChessboardGeometry = new osg::Geometry();
    pChessboardGeode->addDrawable(pChessboardGeometry);
    addChild( pChessboardGeode );

    osg::Vec3Array* pVertices = new osg::Vec3Array;
    for( unsigned int r = 0; r < uRows_; r ++ )
        for( unsigned int c = 0; c < uCols_; c ++ )
        {
            pVertices->push_back( osg::Vec3( c*fMetersPerGrid_, r*fMetersPerGrid_, 0) ); // front left
        }

    pChessboardGeometry->setVertexArray( pVertices );

    osg::DrawElementsUInt* pLines = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);

    unsigned int uTotalCorners = uRows_*uCols_;
    unsigned int i = 0;
    for( unsigned int r = 0; r < uRows_; r ++ )
        for( unsigned int c = 0; c < uCols_; c ++ )
        {
            // horizontal line segments
            if( i+1-r*uCols_ < uCols_ )
            {
                pLines->push_back(i);
                pLines->push_back(i+1);
            }
            // vertical line segments
            if( i+uCols_ < uTotalCorners )
            {
                pLines->push_back(i);
                pLines->push_back(i+uCols_);
            }
            i++;
        }

    pChessboardGeometry->addPrimitiveSet( pLines );
    osg::ref_ptr<osg::Vec4Array> pColors = new osg::Vec4Array;
    pColors->push_back( osg::Vec4( 0.8f, 0.8f, 0.8f, 1.f ) );
    pChessboardGeometry->setColorArray( pColors );

    osg::TemplateIndexArray
    <unsigned int, osg::Array::UIntArrayType,4,1> *pColorIndexArray;
    pColorIndexArray =
        new osg::TemplateIndexArray<unsigned int, osg::Array::UIntArrayType,4,1>;
    pColorIndexArray->push_back(0); // vertex 0 assigned color array element 0

    pChessboardGeometry->setColorIndices(pColorIndexArray);
    pChessboardGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);

    return;
}

         
COsgCamera::COsgCamera( const Matrix3d& mK_, const double& dWidth_, const double& dHeight_, const double& dNear_, const double dFar_ )
{
    __setupWindowView( mK_, dWidth_, dHeight_, dNear_, dFar_ );
}


COsgCamera::COsgCamera( const double& u, const double& v, const double& f, const double& dWidth_, const double& dHeight_, const double& dNear_, const double dFar_ )
{
    __setupWindowView(u, v, f, dWidth_, dHeight_, dNear_, dFar_ );
}

COsgCamera::COsgCamera( const double& dLeft_, const double& dRight_, const double& dTop_, const double& dBottom_, const double& dWidth_, const double& dHeight_, const double& dNear_, const double dFar_ )
{
    __setupWindowView(dLeft_, dRight_, dTop_, dBottom_, dWidth_, dHeight_, dNear_, dFar_ );
}

void COsgCamera::__setupWindowView( const Matrix3d& mK_, const double& dWidth_, const double& dHeight_, const double& dNear_, const double dFar_ )
{
    double u       = mK_( 0, 2 );
    double v       = mK_( 1, 2 );
    double f       =(mK_( 0, 0 ) + mK_( 1,1 ))/2.;
    __setupWindowView(u, v, f, dWidth_, dHeight_, dNear_, dFar_ );
}


void COsgCamera::__setupWindowView(const double& u, const double& v, const double& f, const double& dWidth_, const double& dHeight_, const double& dNear_, const double dFar_ )
{
    double dTop    =              v    / f;
    double dBottom = - ( dHeight_-v )  / f;
    double dLeft   =             -u    / f;
    double dRight  = ( dWidth_  - u )  / f;
    __setupWindowView(dLeft, dRight, dTop, dBottom, dWidth_, dHeight_, dNear_, dFar_ );
}

void COsgCamera::__setupWindowView(const double& dLeft_, const double& dRight_, const double& dTop_, const double& dBottom_, const double& dWidth_, const double& dHeight_, const double& dNear_, const double dFar_ )
{
    setViewport(new osg::Viewport(0,0, dWidth_, dHeight_));
    setProjectionMatrixAsFrustum ( dLeft_, dRight_, dBottom_, dTop_, 0.01, 30 );
    getViewMatrix().makeIdentity();
    //both way works
    setViewMatrixAsLookAt(osg::Vec3d(0.0, 20.0, 0.0),osg::Vec3d(0.0,0.0,0.0), osg::Vec3d(0.0,-1.0,0.0));
    //setViewMatrix(osg::Matrix::lookAt(osg::Vec3(20, 20, 20), osg::Vec3(0, 0, 0), osg::Vec3(0, 1, 0)));
}


}
}
} // namespace btl::extra::gui
