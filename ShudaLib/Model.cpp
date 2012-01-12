

#include <vector>
#include "Utility.hpp"
#include "VideoSourceKinect.hpp"
#include "Model.h"

namespace btl
{
namespace extra
{

CModel::CModel(VideoSourceKinect& cKinect_)
	:_cKinect(cKinect_)
{
	_uCurrent = -1;
	//allocate 
	_vcvmRGBs.resize(_uMaxFrames);
	_vcvmDepths.resize(_uMaxFrames);
	_veivCentroids.resize(_uMaxFrames);
	//allocate
	_pPointL0 = new double[ KINECT_WxHx3 ];//aligned to RGB image of the X,Y,Z coordinate
	_pPointL1 = new double[ KINECT_WxHx3_L1 ];
	_pPointL2 = new double[ KINECT_WxHx3_L2 ];
	_pPointL3 = new double[ KINECT_WxHx3_L2/4];
	//control
	_nKNearest = 6;
	_eNormalExtraction = CModel::_FAST;
	//load
	loadFrame();
}

CModel::~CModel(void)
{
	delete [] _pPointL0;
	delete [] _pPointL1;
	delete [] _pPointL2;
	delete [] _pPointL3;
}

void CModel::loadFrame()
{
	_uCurrent++;
	//load from video source
	_cKinect.getNextFrame();
	_cKinect.cloneFrame(&_cvmRGB,&_cvmDepth);
	_cKinect.centroid(&_eivCentroid);
	//store 
	_vcvmRGBs.push_back(_cvmRGB.clone());
	_vcvmDepths.push_back(_cvmDepth.clone());
	_veivCentroids.push_back(_eivCentroid);
	//convert raw data to point cloud data
	_cKinect.unprojectRGB(_cvmDepth,_pPointL0);
	switch(_eNormalExtraction)
	{
	case _FAST:
        PRINTSTR(  "CModel::_FAST" );
		btl::utility::normalEstimationGL<double>(_pPointL0,_cvmRGB,&_vColors,&_vPts,&_vNormals);
		break;
	case _PCL:
		PRINTSTR(  "CModel::_PCL" );
		PRINT(_nKNearest);
		btl::utility::normalEstimationGLPCL<double>(_pPointL0,_cvmRGB,_nKNearest, &_vColors,&_vPts,&_vNormals);
		break;
	}
	return;
}

}//extra
}//btl