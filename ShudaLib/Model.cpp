#define INFO

#include <vector>
#include "Utility.hpp"
#include "opencv2/gpu/gpu.hpp"
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
		//PRINTSTR(  "CModel::_FAST" );
		btl::utility::normalEstimationGL<double>(_pPointL0,cvmRGB_,pvColor_,pvPt_,pvNormal_,pvX_,pvY_);
		break;
	case _PCL:
		//PRINTSTR(  "CModel::_PCL" );
		//PRINT(_nKNearest);
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

void CModel::normalHistogram( const std::vector<Eigen::Vector3d>& vNormal_, int nSamples_, std::vector< tp_normal_hist_bin >* pvNormalHistogram_)
{
	//clear and re-initialize pvvIdx_
	int nSampleAzimuth_ = nSamples_<<2; //nSamples*4
	pvNormalHistogram_->clear();
	pvNormalHistogram_->resize(nSamples_*nSampleAzimuth_,tp_normal_hist_bin(std::vector<unsigned int>(),Eigen::Vector3d(0,0,0)));
	const double dS = M_PI_2/nSamples_;//sampling step
	unsigned int i=0;
	std::vector< Eigen::Vector3d >::const_iterator cit = vNormal_.begin();
	for( ; cit!= vNormal_.end(); cit++,i++)
	{
		int r,c,rc;
		btl::utility::normalVotes<double>(cit->data(),dS,&r,&c);
		rc = r*nSampleAzimuth_+c;
		(*pvNormalHistogram_)[rc].first.push_back(i);
		(*pvNormalHistogram_)[rc].second += *cit;
	}
	//average the 
	for(std::vector<tp_normal_hist_bin>::iterator it_vNormalHist = pvNormalHistogram_->begin();it_vNormalHist!=pvNormalHistogram_->end(); it_vNormalHist++)
	{
		if(it_vNormalHist->first.size()>0)
		{
			it_vNormalHist->second /= it_vNormalHist->first.size();
		}
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

	std::vector< tp_normal_hist_bin > vNormalHist;//idx of sampling the unit half sphere of top pyramid
	//_vvIdx is organized as r(elevation)*c(azimuth) and stores the idx of Normals
	normalHistogram(vNormals,nSampleElevation,&vNormalHist);
	
	//re-cluster the normals
	_vvLabelPointIdx.clear();
	std::vector<short> vLabel(vNormals.size(),-1);
	short nLabel =0;
	for(unsigned int uIdxBin = 0; uIdxBin < vNormalHist.size(); uIdxBin++)
	{
		//get neighborhood of a sampling bin
		std::vector<unsigned int> vNeighourhood; 
		btl::utility::getNeighbourIdxCylinder< unsigned int >(nSampleElevation,nSampleElevation*4,uIdxBin,&vNeighourhood);

		//traverse the neighborhood and cluster the 
		std::vector<unsigned int> vLabelNormalIdx;
		for( std::vector<unsigned int>::const_iterator cit_vNeighbourhood=vNeighourhood.begin();
			cit_vNeighbourhood!=vNeighourhood.end();cit_vNeighbourhood++)
		{
			btl::utility::normalCluster<double>(vNormals,vNormalHist[*cit_vNeighbourhood].first,vNormalHist[*cit_vNeighbourhood].second,dCosThreshold,nLabel,&vLabel,&vLabelNormalIdx);
		}
		Eigen::Vector3d eivAvgNl;
		btl::utility::avgNormals<double>(vNormals,vLabelNormalIdx,&eivAvgNl);
		_vvLabelPointIdx.push_back(vLabelNormalIdx);
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
	//clear
	_vvClusterPointIdx.clear();
	//construct the label mat
	const unsigned int uTopLevel=_cKinect._uPyrHeight-1;
	const cv::Mat& cvmDepth = _vcvmPyramidDepths[uTopLevel];
	const std::vector< Eigen::Vector3d >& vPts = _vvPyramidPts[uTopLevel];
	const std::vector< Eigen::Vector3d >& vNls = _vvPyramidNormals[uTopLevel];
	const double dLow  = -3;
	const double dHigh =  3;
	const int nSamples = 400;
	const double dSampleStep = ( dHigh - dLow )/nSamples; 
	const unsigned int nArea = 1;
	typedef std::pair< double,unsigned int > tp_pair_hist_element; 
	typedef std::pair< std::vector< tp_pair_hist_element >, double >      tp_pair_hist_bin;
	typedef std::vector< tp_pair_hist_bin >								  tp_hist;
	tp_hist	vDistHist; //histogram of distancte vector< vDist, cit_vIdx > 

	for(std::vector< std::vector< unsigned int > >::const_iterator cit_vvLabelPointIdx = _vvLabelPointIdx.begin();
	    cit_vvLabelPointIdx!=_vvLabelPointIdx.end(); cit_vvLabelPointIdx++)
	{
		vDistHist.clear();
		vDistHist.resize(nSamples,tp_pair_hist_bin(std::vector<tp_pair_hist_element>(), 0.) );
		//collect the distance 
		for(std::vector< unsigned int >::const_iterator cit_vPointIdx = cit_vvLabelPointIdx->begin();
			cit_vPointIdx!=cit_vvLabelPointIdx->end(); cit_vPointIdx++)
		{
			double dDist = vPts[*cit_vPointIdx].dot(vNls[*cit_vPointIdx]);

			int nBin = floor( (dDist -dLow)/ dSampleStep );
			if( nBin >= 0 && nBin <nSamples)
			{
				vDistHist[nBin].first.push_back(tp_pair_hist_element(dDist,*cit_vPointIdx));
				vDistHist[nBin].second += dDist;
			}
		}

		//calc the avg distance for each bin 
		//construct a list for sorting
		for(std::vector< tp_pair_hist_bin >::iterator cit_vDistHist = vDistHist.begin();
			cit_vDistHist != vDistHist.end(); cit_vDistHist++ )
		{
			unsigned int uBinSize = cit_vDistHist->first.size();
			if( uBinSize==0 ) continue;

			//calculate avg distance
			cit_vDistHist->second /= uBinSize;
		}
		const double dMergeDistance = dSampleStep*1.5;
		//merge the bins whose distance is similar
	
		std::vector< tp_flag > vMergeFlags(nSamples, CModel::EMPTY); //==0 no merging, ==1 merge with left, ==2 merge with right, ==3 merging with both
		std::vector< tp_flag >::iterator it_vMergeFlags = vMergeFlags.begin()+1; 
		std::vector< tp_flag >::iterator it_prev;
		std::vector< tp_pair_hist_bin >::const_iterator cit_prev;
		std::vector< tp_pair_hist_bin >::const_iterator cit_endm1 = vDistHist.end() - 1;
	
		for(std::vector< tp_pair_hist_bin >::const_iterator cit_vDistHist = vDistHist.begin() + 1;
			cit_vDistHist != cit_endm1; cit_vDistHist++,it_vMergeFlags++ )

		{
			unsigned int uBinSize = cit_vDistHist->first.size();
			if(0==uBinSize) continue;
			*it_vMergeFlags = CModel::NO_MERGE;
			cit_prev = cit_vDistHist -1;
			it_prev  = it_vMergeFlags-1;
			if( CModel::EMPTY == *it_prev ) continue;

			if( fabs(cit_prev->second - cit_vDistHist->second) < dMergeDistance ) //avg distance smaller than the sample step.
			{
				//previou bin
				if(CModel::NO_MERGE==*it_prev)
					*it_prev = CModel::MERGE_WITH_RIGHT;
				else if(CModel::MERGE_WITH_LEFT==*it_prev)
					*it_prev = CModel::MERGE_WITH_BOTH;
				//current bin
				*it_vMergeFlags = CModel::MERGE_WITH_LEFT;
			}//if mergable
			
		}//for each bin
		//merge
		std::vector< unsigned int > vCluster;
		std::vector< tp_flag >::const_iterator cit_vMergeFlags = vMergeFlags.begin();
		for(std::vector< tp_pair_hist_bin >::const_iterator cit_vDistHist = vDistHist.begin() + 1;
			cit_vDistHist != cit_endm1; cit_vDistHist++,cit_vMergeFlags++ )
		{
			if(CModel::EMPTY==*cit_vMergeFlags) continue;

			if(CModel::NO_MERGE==*cit_vMergeFlags||CModel::MERGE_WITH_RIGHT==*cit_vMergeFlags||CModel::MERGE_WITH_BOTH==*cit_vMergeFlags||CModel::MERGE_WITH_LEFT==*cit_vMergeFlags)
			{
				for( std::vector<tp_pair_hist_element>::const_iterator cit_vPair = cit_vDistHist->first.begin();
					cit_vPair != cit_vDistHist->first.end(); cit_vPair++ )
				{
					vCluster.push_back( cit_vPair->second );
				}
			}
			
			if(CModel::NO_MERGE==*cit_vMergeFlags||CModel::MERGE_WITH_LEFT==*cit_vMergeFlags)
			{
				_vvClusterPointIdx.push_back(vCluster);
				vCluster.clear();
			}
		}

	}//for each normal label

	return;
}

}//extra
}//btl
