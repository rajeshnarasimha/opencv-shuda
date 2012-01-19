#define INFO

#include <vector>
#include "Utility.hpp"
#include "VideoSourceKinect.hpp"
#include "Model.h"

#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

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
	_vvX.clear();_vvY.clear();
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
		std::vector< int > vX,vY; 
		convert2PointCloudModelGL(_vcvmPyramidDepths[i],_vcvmPyramidRGBs[i],i, &vColors,&vPts,&vNormals,&vX,&vY);
		_vvPyramidColors.push_back(vColors);
		_vvPyramidPts.push_back(vPts);
		_vvPyramidNormals.push_back(vNormals);
		_vvX.push_back(vX);
		_vvY.push_back(vY);
	}
	
	return;
}

void CModel::convert2PointCloudModelGL(const cv::Mat& cvmDepth_,const cv::Mat& cvmRGB_, unsigned int uLevel_, 
	std::vector<const unsigned char*>* pvColor_,std::vector<Eigen::Vector3d>* pvPt_, std::vector<Eigen::Vector3d>* pvNormal_, 
	std::vector< int >* pvX_/*=NULL*/, std::vector< int >* pvY_/*=NULL*/)
{
	//convert raw data to point cloud data
	_cKinect.unprojectRGB(cvmDepth_, _pPointL0, uLevel_);
	switch(_eNormalExtraction)
	{
	case _FAST:
		PRINTSTR(  "CModel::_FAST" );
		btl::utility::normalEstimationGL<double>(_pPointL0,cvmRGB_,pvColor_,pvPt_,pvNormal_,pvX_,pvY_);
		break;
	case _PCL:
		PRINTSTR(  "CModel::_PCL" );
		PRINT(_nKNearest);
		btl::utility::normalEstimationGLPCL<double>(_pPointL0,cvmRGB_,_nKNearest, pvColor_,pvPt_,pvNormal_);
		break;
	}
	return;
}
void CModel::loadPyramidAndDetectPlanePCL()
{
	_vvPyramidColors.clear();
	_vvPyramidPts.clear();
	_vvPyramidNormals.clear();
	_uCurrent++;
	//load from video source
	if(_cKinect._ePreFiltering != VideoSourceKinect::PYRAMID_BILATERAL_FILTERED_IN_DISPARTY)
		_cKinect._ePreFiltering = VideoSourceKinect::PYRAMID_BILATERAL_FILTERED_IN_DISPARTY;
	_cKinect._uPyrHeight = 3;
	_cKinect.getNextFrame();
	_cKinect.clonePyramid(&_vcvmPyramidRGBs,&_vcvmPyramidDepths);
	_cKinect.centroid(&_eivCentroid);
	//detect plane in the top pyramid
	std::vector<int> vX,vY;
	detectPlanePCL(2,&vX,&vY);
	//extract plane
	extractPlaneGL(2,vX,vY,&_veivPlane);
	//convert raw data to point cloud data
	for(unsigned int i=0;i<_vcvmPyramidDepths.size();i++)
	{
		std::vector<const unsigned char*> vColors;
		std::vector<Eigen::Vector3d> vPts;
		std::vector<Eigen::Vector3d> vNormals;
		convert2PointCloudModelGL(_vcvmPyramidDepths[i],_vcvmPyramidRGBs[i],i,&vColors,&vPts,&vNormals);
		_vvPyramidColors.push_back(vColors);
		_vvPyramidPts.push_back(vPts);
		_vvPyramidNormals.push_back(vNormals);
	}
	return;
}
void CModel::extractPlaneGL(unsigned int uLevel_, const std::vector<int>& vX_, const std::vector<int>& vY_, std::vector<Eigen::Vector3d>* pvPlane_)
{
	cv::Mat& cvmDepth = _vcvmPyramidDepths[uLevel_];
	pvPlane_->clear();
	for (unsigned int i=0;i<vX_.size(); i++)
	{
		int x = vX_[i];
		int y = vY_[i];
		Eigen::Vector3d eivPt;
		_cKinect.unprojectRGBGL(cvmDepth,y,x,eivPt.data(),uLevel_);
		pvPlane_->push_back(eivPt);
		cvmDepth.at<float>(y,x)=0.f;
	}
}
void CModel::detectPlanePCL(unsigned int uLevel_,std::vector<int>* pvXIdx_, std::vector<int>* pvYIdx_)
{
	const cv::Mat& cvmDepth = _vcvmPyramidDepths[uLevel_];
	//do plane detection in disparity domain
	cv::Mat	cvmDisparity;
	float fMin,fMax,fRange,fRatio;
	btl::utility::convert2DisparityDomain<float>(cvmDepth,&cvmDisparity,&fMax,&fMin);
	//normalize the x y into the same scale as disparity
	fRange = fMax - fMin;
	PRINT(fRange);
	fRatio = fRange/cvmDepth.cols;//make the x
	//each pixel in cvmDisparity is now equivalent to (x*fRatio, y*fRatio, disparity)
	//construct PCL point cloud data
	float* pDisparity = (float*)cvmDisparity.data;
	pcl::PointCloud<pcl::PointXYZ> pclNoneZero;
	
	for(int r = 0; r<cvmDisparity.rows; r++)
	for(int c = 0; c<cvmDisparity.cols; c++)
	{
		float dz = *pDisparity;
		if( fabs(dz) > SMALL )
		{
			pcl::PointXYZ point(c*fRatio,r*fRatio,dz);
			pclNoneZero.push_back(point);
		}
		pDisparity++;
	}
	//detect
	pcl::ModelCoefficients::Ptr pCoefficients (new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr pInliers (new pcl::PointIndices);
	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZ> cSeg;
	// Optional
	cSeg.setOptimizeCoefficients (true);
	// Mandatory
	cSeg.setModelType (pcl::SACMODEL_PLANE);
	cSeg.setMethodType (pcl::SAC_RANSAC);
	cSeg.setDistanceThreshold (fRange/1000.);
	cSeg.setInputCloud (pclNoneZero.makeShared ());
	cSeg.segment (*pInliers, *pCoefficients);
	// retrieve inliers
	pvXIdx_->clear();pvYIdx_->clear();
	for (size_t i = 0; i < pInliers->indices.size (); ++i)
	{
		int y = int( pclNoneZero.points[pInliers->indices[i]].y/fRatio + .5);
		int x = int( pclNoneZero.points[pInliers->indices[i]].x/fRatio + .5);
		pvYIdx_->push_back(y);
		pvXIdx_->push_back(x);
	}
	return;
}
void CModel::clusterNormal()
{
	//define constants
	const unsigned int uTopLevel=_cKinect._uPyrHeight-1;
	const int nSampleElevation = 4;
	const int nClusterSize = btl::extra::videosource::__aKinectWxH[uTopLevel]/100;
	const double dCosThreshold = std::cos(M_PI_4/nSampleElevation);
	const std::vector< Eigen::Vector3d >& vNormals = _vvPyramidNormals[uTopLevel];
	//make a histogram on the top pyramid
	_vvNormalIdx.clear();//_vvIdx is organized as r(elevation)*c(azimuth) and stores the idx of Normals
	btl::utility::normalHistogram<double>(vNormals,nSampleElevation,&_vvNormalIdx);
	//calculate the average normal for each bin of the sampling space which is larger than 100
	std::vector< unsigned int > vAvgNlSamplingIdx; // the sampling idx of those larger than 100
	std::vector< Eigen::Vector3d > vAvgNl; 
	unsigned int i=0;
	for(std::vector< std::vector< unsigned int > >::const_iterator cit_vvNormalIdx = _vvNormalIdx.begin(); 
		cit_vvNormalIdx!=_vvNormalIdx.end(); cit_vvNormalIdx++,i++)
	{
		// average the normal for the bin larger than 100
		if(cit_vvNormalIdx->size()>100)
		{
			//calculate its average normal
			//and record its index as well
			Eigen::Vector3d eivNl; 
			btl::utility::avgNormals<double>(vNormals,*cit_vvNormalIdx,&eivNl);
			PRINT(eivNl);
			vAvgNl.push_back(eivNl);
			vAvgNlSamplingIdx.push_back(i);
		}
	}
	//re-cluster the normals
	_vvLabelNormalIdx.clear();
	std::vector<short> vLabel(vNormals.size(),-1);
	std::vector<Eigen::Vector3d>::const_iterator cit_vAvgNl = vAvgNl.begin();
	short nLabel =0;
	for (std::vector<unsigned int>::const_iterator cit_vSamplingIdx = vAvgNlSamplingIdx.begin();
		cit_vSamplingIdx!=vAvgNlSamplingIdx.end(); cit_vSamplingIdx++,cit_vAvgNl++,nLabel++)
	{
		//get neighborhood of a sampling bin
		std::vector<unsigned int> vNeighourhood; 
		btl::utility::getNeighbourIdxCylinder< unsigned int >(nSampleElevation,nSampleElevation*4,*cit_vSamplingIdx,&vNeighourhood);

		//traverse the neighborhood and cluster the 
		std::vector<unsigned int> vLabelNormalIdx;
		for( std::vector<unsigned int>::const_iterator cit_vNeighbourhood=vNeighourhood.begin();
			cit_vNeighbourhood!=vNeighourhood.end();cit_vNeighbourhood++)
		{
			btl::utility::normalCluster<double>(vNormals,_vvNormalIdx[*cit_vNeighbourhood],*cit_vAvgNl,dCosThreshold,nLabel,&vLabel,&vLabelNormalIdx);
		}
		Eigen::Vector3d eivAvgNl;
		btl::utility::avgNormals<double>(vNormals,vLabelNormalIdx,&eivAvgNl);
		_vvLabelNormalIdx.push_back(vLabelNormalIdx);
		_vLabelAvgNormals.push_back(eivAvgNl);
	}
	return;
}
void CModel::loadPyramidAndDetectPlane()
{
//load pyramids
	_cKinect._uPyrHeight = 4;
	loadPyramid(); //output _vvN
//cluster the top pyramid
	clusterNormal();
//enforce position continuity
	//construct the label mat
	const unsigned int uTopLevel=_cKinect._uPyrHeight-1;
	const cv::Mat& cvmDepth = _vcvmPyramidDepths[uTopLevel];
	const std::vector< Eigen::Vector3d >& vPts = _vvPyramidPts[uTopLevel];
	short sLabel = 0;
	std::vector< Eigen::Vector3d >::const_iterator cit_vLabelAvgNormals = _vLabelAvgNormals.begin();
	for(std::vector< std::vector< unsigned int > >::const_iterator cit_vvLabelNormalIdx = _vvLabelNormalIdx.begin();
	cit_vvLabelNormalIdx!=_vvLabelNormalIdx.end(); cit_vvLabelNormalIdx++,sLabel++,cit_vLabelAvgNormals)
	{
		for(std::vector< unsigned int >::const_iterator cit_vNormalIdx = cit_vvLabelNormalIdx->begin();
			cit_vNormalIdx!=cit_vvLabelNormalIdx->end(); cit_vNormalIdx++)
		{
			double dDist = vPts[*cit_vNormalIdx].dot(*cit_vLabelAvgNormals);
			
		}
	}
	//convert Depth to disparity domain;
	cv::Mat cvmDisparity; float fMin, fMax;
	btl::utility::convert2DisparityDomain<float>(cvmDepth,&cvmDisparity,&fMax,&fMin);
}

}//extra
}//btl