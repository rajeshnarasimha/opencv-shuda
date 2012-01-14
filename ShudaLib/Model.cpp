

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
	//allocate
	_pPointL0 = new double[ KINECT_WxHx3 ];//aligned to RGB image of the X,Y,Z coordinate
	//control
	_nKNearest = 6;
	_cKinect._uPyrHeight = 1;
	_eNormalExtraction = CModel::_FAST;
	//load
	loadFrame();
}
CModel::~CModel(void)
{
	delete [] _pPointL0;
}
void CModel::loadFrame()
{
	_cKinect._uPyrHeight = 1;
	loadPyramid();
	return;
}
void CModel::loadPyramid()
{
	_vvPyramidColors.clear();
	_vvPyramidPts.clear();
	_vvPyramidNormals.clear();
	_uCurrent++;
	//load from video source
	if(_cKinect._ePreFiltering != VideoSourceKinect::PYRAMID_BILATERAL_FILTERED_IN_DISPARTY)
		_cKinect._ePreFiltering = VideoSourceKinect::PYRAMID_BILATERAL_FILTERED_IN_DISPARTY;
	_cKinect.getNextFrame();
	_cKinect.clonePyramid(&_vcvmPyramidRGBs,&_vcvmPyramidDepths);
	_cKinect.centroid(&_eivCentroid);
	//convert raw data to point cloud data
	for(unsigned int i=0;i<_vcvmPyramidDepths.size();i++)
	{
		std::vector<const unsigned char*> vColors;
		std::vector<Eigen::Vector3d> vPts;
		std::vector<Eigen::Vector3d> vNormals;
		convert2PointCloudModel(_vcvmPyramidDepths[i],_vcvmPyramidRGBs[i],&vColors,&vPts,&vNormals,i);
		_vvPyramidColors.push_back(vColors);
		_vvPyramidPts.push_back(vPts);
		_vvPyramidNormals.push_back(vNormals);
	}
	
	return;
}
void CModel::convert2PointCloudModel(const cv::Mat& cvmDepth_,const cv::Mat& cvmRGB_, std::vector<const unsigned char*>* pvColor_, 
	std::vector<Eigen::Vector3d>* pvPt_, std::vector<Eigen::Vector3d>* pvNormal_,int nLevel_/*=0*/)
{
	//convert raw data to point cloud data
	_cKinect.unprojectRGB(cvmDepth_, _pPointL0, nLevel_);
	switch(_eNormalExtraction)
	{
	case _FAST:
		PRINTSTR(  "CModel::_FAST" );
		btl::utility::normalEstimationGL<double>(_pPointL0,cvmRGB_,pvColor_,pvPt_,pvNormal_);
		break;
	case _PCL:
		PRINTSTR(  "CModel::_PCL" );
		PRINT(_nKNearest);
		btl::utility::normalEstimationGLPCL<double>(_pPointL0,cvmRGB_,_nKNearest, pvColor_,pvPt_,pvNormal_);
		break;
	}
	return;
}
}//extra
}//btl