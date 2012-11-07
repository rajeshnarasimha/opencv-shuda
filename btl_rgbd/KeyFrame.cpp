#define INFO
//gl
#include <gl/glew.h>
#include <gl/freeglut.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
//boost
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/math/special_functions/fpclassify.hpp> //isnan
#include <boost/lexical_cast.hpp>
//stl
#include <vector>
#include <iostream>
#include <fstream>
#include <list>
#include <math.h>
//openncv
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "OtherUtil.hpp"
#include "Converters.hpp"
#include "EigenUtil.hpp"
#include "Camera.h"
#include "Kinect.h"
#include "GLUtil.h"
#include "PlaneObj.h"
#include "Histogram.h"
#include "KeyFrame.h"
#include "CVUtil.hpp"
#include "Utility.hpp"
#include "cuda/CudaLib.h"
#include "cuda/pcl/internal.h"
#include "cuda/Registartion.h"

btl::utility::SNormalHist btl::kinect::CKeyFrame::_sNormalHist;
btl::utility::SDistanceHist btl::kinect::CKeyFrame::_sDistanceHist;
boost::shared_ptr<cv::Mat> btl::kinect::CKeyFrame::_acvmShrPtrAA[4];
boost::shared_ptr<cv::gpu::GpuMat> btl::kinect::CKeyFrame::_acvgmShrPtrAA[4];//for rendering
boost::shared_ptr<cv::gpu::GpuMat> btl::kinect::CKeyFrame::_acvgmShrPtrPyrDisparity[4];
boost::shared_ptr<cv::gpu::GpuMat> btl::kinect::CKeyFrame::_acvgmShrPtrPyr32FC1Tmp[4];

boost::shared_ptr<cv::gpu::SURF_GPU> btl::kinect::CKeyFrame::_pSurf;
boost::shared_ptr<cv::gpu::ORB_GPU>  btl::kinect::CKeyFrame::_pOrb;

btl::kinect::CKeyFrame::CKeyFrame( btl::kinect::SCamera::tp_ptr pRGBCamera_, ushort uResolution_, ushort uPyrLevel_, float fCwX_, float fCwY_, float fCwZ_ )
:_pRGBCamera(pRGBCamera_),_uResolution(uResolution_),_uPyrHeight(uPyrLevel_){
	allocate();
	_eivInitCw << fCwX_, fCwY_, fCwZ_; 
	initRT();
}
btl::kinect::CKeyFrame::CKeyFrame( CKeyFrame::tp_ptr pFrame_ )
{
	_pRGBCamera = pFrame_->_pRGBCamera;
	_uResolution = pFrame_->_uResolution;
	_uPyrHeight = pFrame_->_uPyrHeight;
	allocate();
	pFrame_->copyTo(this);
}
void btl::kinect::CKeyFrame::allocate(){
	//disparity
	for(int i=0; i<_uPyrHeight; i++){
		int nRows = __aKinectH[_uResolution]>>i; 
		int nCols = __aKinectW[_uResolution]>>i;
		//host
		_acvmShrPtrPyrPts[i] .reset(new cv::Mat(nRows,nCols,CV_32FC3));
		_acvmShrPtrPyrNls[i] .reset(new cv::Mat(nRows,nCols,CV_32FC3));
		_acvmShrPtrPyrRGBs[i].reset(new cv::Mat(nRows,nCols,CV_8UC3));
		_acvmShrPtrPyrBWs[i] .reset(new cv::Mat(nRows,nCols,CV_8UC1));
		_acvmPyrDepths[i]	 .reset(new cv::Mat(nRows,nCols,CV_32FC1));
		//device
		_acvgmShrPtrPyrPts[i] .reset(new cv::gpu::GpuMat(nRows,nCols,CV_32FC3));
		_acvgmShrPtrPyrNls[i] .reset(new cv::gpu::GpuMat(nRows,nCols,CV_32FC3));
		_acvgmShrPtrPyrRGBs[i].reset(new cv::gpu::GpuMat(nRows,nCols,CV_8UC3));
		_acvgmShrPtrPyrBWs[i] .reset(new cv::gpu::GpuMat(nRows,nCols,CV_8UC1));
		_acvgmShrPtrPyrDepths[i]	  .reset(new cv::gpu::GpuMat(nRows,nCols,CV_32FC1));
		//plane detection
		_acvmShrPtrNormalClusters[i].reset(new cv::Mat(nRows,nCols,CV_16SC1));
		_acvmShrPtrDistanceClusters[i].reset(new cv::Mat(nRows,nCols,CV_32FC1));
		//device
		_acvgmShrPtrPyrDisparity[i].reset(new cv::gpu::GpuMat(nRows,nCols,CV_32FC1));
		_acvgmShrPtrPyr32FC1Tmp[i].reset(new cv::gpu::GpuMat(nRows,nCols,CV_32FC1));
	}

	_eConvention = btl::utility::BTL_CV;
	//setRT(0,0,0,.5,.5,-0.1);
	

	_bIsReferenceFrame = false;
	_bRenderPlane = false;
	_eClusterType = NORMAL_CLUSTER;//DISTANCE_CLUSTER;
	_nColorIdx = 0;

	//rendering
	glPixelStorei ( GL_UNPACK_ALIGNMENT, 4 ); // 4
}

void btl::kinect::CKeyFrame::setRT(float fXA_, float fYA_, float fZA_, float fCwX_,float fCwY_,float fCwZ_){
	cv::Mat_<float> cvmR,cvmRVec(3,1);
	cvmRVec << fXA_,fYA_,fZA_;
	cv::Rodrigues(cvmRVec,cvmR);
	using namespace btl::utility;
	_eimRw << cvmR;
	Eigen::Vector3f eivC(fCwX_,fCwY_,fCwZ_); //camera location in the world cv-convention
	_eivTw = -_eimRw*eivC;
	updateMVInv();
}

void btl::kinect::CKeyFrame::initRT(){
	_eimRw << 1.f, 0.f, 0.f,
		      0.f, 1.f, 0.f,
			  0.f, 0.f, 1.f;
	_eivTw = -_eivInitCw; 
	updateMVInv();
}

void btl::kinect::CKeyFrame::copyTo( CKeyFrame* pKF_, const short sLevel_ ){
	//host
	_acvmShrPtrPyrPts[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrPts[sLevel_]);
	_acvmShrPtrPyrNls[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrNls[sLevel_]);
	_acvmShrPtrPyrRGBs[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrRGBs[sLevel_]);
	_acvmShrPtrPyrBWs[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrBWs[sLevel_]);
	_acvmShrPtrDistanceClusters[sLevel_]->copyTo(*pKF_->_acvmShrPtrDistanceClusters[sLevel_]);
	//device
	_acvgmShrPtrPyrPts[sLevel_]->copyTo(*pKF_->_acvgmShrPtrPyrPts[sLevel_]);
	_acvgmShrPtrPyrNls[sLevel_]->copyTo(*pKF_->_acvgmShrPtrPyrNls[sLevel_]);
	_acvgmShrPtrPyrRGBs[sLevel_]->copyTo(*pKF_->_acvgmShrPtrPyrRGBs[sLevel_]);
	_acvgmShrPtrPyrBWs[sLevel_]->copyTo(*pKF_->_acvgmShrPtrPyrBWs[sLevel_]);
	pKF_->_eConvention = _eConvention;
}

void btl::kinect::CKeyFrame::copyTo( CKeyFrame* pKF_ ) {
	for(int i=0; i<_uPyrHeight; i++) {
		copyTo(pKF_,i);
	}
	_acvgmShrPtrPyrDepths[0]->copyTo(*pKF_->_acvgmShrPtrPyrDepths[0]);
	//copy surf features
	
	if( !_vKeyPoints.empty() ){
		_cvgmKeyPoints.copyTo(pKF_->_cvgmKeyPoints);
		_cvgmDescriptors.copyTo(pKF_->_cvgmDescriptors);
		pKF_->_vKeyPoints.resize(_vKeyPoints.size());
		std::copy( _vKeyPoints.begin(), _vKeyPoints.end(), pKF_->_vKeyPoints.begin() );
	}
	//other
	pKF_->_bIsReferenceFrame = _bIsReferenceFrame;
	pKF_->_eimRw = _eimRw;
	pKF_->_eivTw = _eivTw;
	pKF_->updateMVInv();
}
void btl::kinect::CKeyFrame::exportPCL(const std::string& strPath_, const std::string& strYMLName_){
	pcl::PointCloud<pcl::PointXYZ>  cloudVertecies;
	pcl::PointCloud<pcl::Normal>	cloudNormals;

	int nPyrLevel = 2;
	// Fill in the cloud data
	cloudVertecies.width    = _acvmShrPtrPyrPts[nPyrLevel ]->cols;
	cloudVertecies.height   = _acvmShrPtrPyrNls[nPyrLevel ]->rows;
	cloudVertecies.is_dense = false;
	cloudVertecies.points.resize (cloudVertecies.width * cloudVertecies.height);
	const float* pPts = (const float* )_acvmShrPtrPyrPts[nPyrLevel ]->data;// points
	// 
	cloudNormals.width    = _acvmShrPtrPyrPts[nPyrLevel ]->cols;
	cloudNormals.height   = _acvmShrPtrPyrPts[nPyrLevel ]->rows;
	cloudNormals.is_dense = false;
	cloudNormals.points.resize (cloudNormals.width * cloudNormals.height);
	const float* pNls = (const float* )_acvmShrPtrPyrNls[nPyrLevel ]->data;// normal
	for (size_t i = 0; i < cloudVertecies.points.size (); ++i)
	{
		cloudVertecies.points[i].x = *pPts++;
		cloudVertecies.points[i].y = *pPts++;
		cloudVertecies.points[i].z = *pPts++;

		cloudNormals.points[i].normal[0] = *pNls++;
		cloudNormals.points[i].normal[1] = *pNls++;
		cloudNormals.points[i].normal[2] = *pNls++;
	}

	pcl::io::savePCDFileASCII ("point.pcd", cloudVertecies);
	pcl::io::savePCDFileASCII ("normal.pcd",cloudNormals);
	std::cerr << "Saved " << cloudVertecies.points.size () << " data points to test_pcd.pcd." << std::endl;

	return;
}
void btl::kinect::CKeyFrame::exportYML(const std::string& strPath_, const std::string& strYMLName_){
	using namespace btl::utility;

	std::string strPathFileName = strPath_ + strYMLName_;
	cv::FileStorage cFSWrite( strPathFileName.c_str(), cv::FileStorage::WRITE );

	cFSWrite << "uPyrHeight" << _uPyrHeight;
	cFSWrite << "uResolution" << _uResolution;
	cv::Mat cvmRw; cvmRw << _eimRw;
	cv::Mat cvmTw; cvmTw << _eivTw;
	cFSWrite << "eimRw" << cvmRw;
	cFSWrite << "eivTw" << cvmTw;

	std::string strVariableName;
	for (int i = 0; i < _uPyrHeight; i++){
		strVariableName = "acvmShrPtrPyrPts";	strVariableName += boost::lexical_cast<std::string> ( i ); cFSWrite << strVariableName.c_str() << *_acvmShrPtrPyrPts[i];
		strVariableName = "acvmShrPtrPyrNls";	strVariableName += boost::lexical_cast<std::string> ( i ); cFSWrite << strVariableName.c_str() << *_acvmShrPtrPyrNls[i];
		strVariableName = "acvmShrPtrPyrRGBs";	strVariableName += boost::lexical_cast<std::string> ( i ); cFSWrite << strVariableName.c_str() << *_acvmShrPtrPyrRGBs[i];
		strVariableName = "acvmShrPtrPyrBWs";	strVariableName += boost::lexical_cast<std::string> ( i ); cFSWrite << strVariableName.c_str() << *_acvmShrPtrPyrBWs[i];
		strPathFileName = strPath_+strYMLName_; strPathFileName += boost::lexical_cast<std::string> ( i );	strPathFileName += ".bmp";	cv::imwrite(strPathFileName,*_acvmShrPtrPyrBWs[i]);
	}
	
	cFSWrite.release();
}

void btl::kinect::CKeyFrame::importYML(const std::string& strPath_, const std::string& strYMLName_){
	using namespace btl::utility;
	std::string strPathFileName = strPath_ + strYMLName_;
	cv::FileStorage cFSRead( strPathFileName.c_str(), cv::FileStorage::READ );

	cFSRead["uPyrHeight"] >> _uPyrHeight;
	cFSRead["uResolution"] >> _uResolution;
	cv::Mat cvmRw; Eigen::MatrixXf eimMat;
	cv::Mat cvmTw; Eigen::VectorXf eimVec;
	cFSRead["eimRw"] >> cvmRw;
	cFSRead["eivTw"] >> cvmTw;
	_eimRw = (eimMat << cvmRw);
	_eivTw = (eimVec << cvmTw);
	updateMVInv();

	std::string strVariableName;
	for (int i = 0; i < _uPyrHeight; i++){
		strVariableName = "acvmShrPtrPyrPts";  strVariableName += boost::lexical_cast<std::string> ( i ); cFSRead[strVariableName.c_str()] >> *_acvmShrPtrPyrPts[i];
		strVariableName = "acvmShrPtrPyrNls";  strVariableName += boost::lexical_cast<std::string> ( i ); cFSRead[strVariableName.c_str()] >> *_acvmShrPtrPyrNls[i];
		strVariableName = "acvmShrPtrPyrRGBs"; strVariableName += boost::lexical_cast<std::string> ( i ); cFSRead[strVariableName.c_str()] >> *_acvmShrPtrPyrRGBs[i];
		strVariableName = "acvmShrPtrPyrBWs";  strVariableName += boost::lexical_cast<std::string> ( i ); cFSRead[strVariableName.c_str()] >> *_acvmShrPtrPyrBWs[i];
	}

	cFSRead.release();
}



void btl::kinect::CKeyFrame::establishPlaneCorrespondences( const CKeyFrame& sReferenceKF_) {
/*
	CHECK ( !_vMatches.empty(), "SKeyFrame::calcRT() _vMatches should not calculated." );
	//calculate the R and T
	Eigen::Matrix3d eimRNew;
	Eigen::Vector3d eivTNew;
	_vPlaneCorrespondences.clear();
	std::vector<SPlaneCorrespondence> vPlaneCorrTmp;
	//search for pairs of correspondences with depth data available.
	const float*const  _pCurrentPlane   = (const float*)              _acvmShrPtrDistanceClusters[0]->data;
	const float*const  _pReferencePlane = (const float*)sReferenceKF_._acvmShrPtrDistanceClusters[0]->data;
	std::vector< int > _vDepthIdxCur, _vDepthIdxRef, _vSelectedPairs;
	unsigned int uMatchIdx = 0;
	for ( std::vector< cv::DMatch >::const_iterator cit = _vMatches.begin(); cit != _vMatches.end(); cit++,uMatchIdx++ ) {
		int nKeyPointIdxCur = cit->queryIdx;
		int nKeyPointIdxRef = cit->trainIdx;

		int nXCur = cvRound ( 			    _vKeyPoints[ nKeyPointIdxCur ].pt.x/8 );
		int nYCur = cvRound ( 			    _vKeyPoints[ nKeyPointIdxCur ].pt.y/8 );
		int nXRef = cvRound ( sReferenceKF_._vKeyPoints[ nKeyPointIdxRef ].pt.x/8 );
		int nYRef = cvRound ( sReferenceKF_._vKeyPoints[ nKeyPointIdxRef ].pt.y/8 );

		int nDepthIdxCur = nYCur * 80 + nXCur;
		int nDepthIdxRef = nYRef * 80 + nXRef;

		if ( _pCurrentPlane[nDepthIdxCur] > 0 && _pReferencePlane[nDepthIdxCur] > 0 ) {
			vPlaneCorrTmp.push_back(SPlaneCorrespondence(_pCurrentPlane[nDepthIdxCur],_pReferencePlane[nDepthIdxCur],uMatchIdx));
		}
	}//for each surf matches
	//analyze _vPlaneCorrespondences and finalize it
	std::sort( vPlaneCorrTmp.begin(), vPlaneCorrTmp.end() );
	float fPlaneCur = vPlaneCorrTmp.begin()->_fCur;
	std::map<int,short> _mCorrespondenceHistogram; _mCorrespondenceHistogram.insert(std::pair<int,short>(vPlaneCorrTmp.begin()->_fRef,1));
	for (std::vector<SPlaneCorrespondence>::iterator itCor = vPlaneCorrTmp.begin()+1; itCor != vPlaneCorrTmp.end(); itCor++) {
		if( fabs(itCor->_fCur - fPlaneCur)< std::numeric_limits<float>::epsilon() ){
			;
		} 
	}
	return;*/

}//establishPlaneCorrespondences()

void btl::kinect::CKeyFrame::extractSurfFeatures ()  {
	(*_pSurf)(*_acvgmShrPtrPyrBWs[0], cv::gpu::GpuMat(), _cvgmKeyPoints, _cvgmDescriptors);
	_pSurf->downloadKeypoints(_cvgmKeyPoints, _vKeyPoints);
	//from current to reference
	//_cvgmKeyPoints.copyTo(sReferenceKF_._cvgmKeyPoints); _cvgmDescriptors.copyTo(sReferenceKF_._cvgmDescriptors);

	return;
}

void btl::kinect::CKeyFrame::extractOrbFeatures ()  {
	(*_pOrb)(*_acvgmShrPtrPyrBWs[0], cv::gpu::GpuMat(), _cvgmKeyPoints, _cvgmDescriptors);
	_pOrb->downloadKeyPoints(_cvgmKeyPoints, _vKeyPoints);
	return;
}

double btl::kinect::CKeyFrame::calcRT ( const CKeyFrame& sPrevKF_, const unsigned short sLevel_ , const double dDistanceThreshold_, unsigned short* pInliers_) {
	// - The reference frame must contain a calibrated Rw and Tw. 
	// - The point cloud in the reference frame must be transformed into the world coordinate system.
	// - The current frame's Rw and Tw must be initialized as the reference's Rw Tw. (This is for fDist = norm3<float>() ) 
	// - The point cloud in the current frame must be in the camera coordinate system.
	BTL_ASSERT(sPrevKF_._vKeyPoints.size()>10,"extractSurfFeatures() Too less SURF features detected in the reference frame")
	//matching from current to reference
	cv::gpu::BruteForceMatcher_GPU< cv::L2<float> > cBruteMatcher;
	cv::gpu::GpuMat cvgmTrainIdx, cvgmDistance;
	cBruteMatcher.matchSingle( this->_cvgmDescriptors,  sPrevKF_._cvgmDescriptors, cvgmTrainIdx, cvgmDistance);
	cv::gpu::BruteForceMatcher_GPU< cv::L2<float> >::matchDownload(cvgmTrainIdx, cvgmDistance, _vMatches);
	std::sort( _vMatches.begin(), _vMatches.end() );
	if (_vMatches.size()> 300) { _vMatches.erase( _vMatches.begin()+ 300, _vMatches.end() ); }
	//CHECK ( !_vMatches.empty(), "SKeyFrame::calcRT() _vMatches should not calculated." );
	//calculate the R and T
	//search for pairs of correspondences with depth data available.
	const float*const  _pCurrentPts   = (const float*)              _acvmShrPtrPyrPts[sLevel_]->data;
	const float*const  _pReferencePts = (const float*)sPrevKF_._acvmShrPtrPyrPts[sLevel_]->data;
	std::vector< int > _vDepthIdxCur, _vDepthIdxRef, _vSelectedPairs;
	for ( std::vector< cv::DMatch >::const_iterator cit = _vMatches.begin(); cit != _vMatches.end(); cit++ ) {
		int nKeyPointIdxCur = cit->queryIdx;
		int nKeyPointIdxRef = cit->trainIdx;

		int nXCur = cvRound ( 		   _vKeyPoints[ nKeyPointIdxCur ].pt.x );
		int nYCur = cvRound ( 		   _vKeyPoints[ nKeyPointIdxCur ].pt.y );
		int nXRef = cvRound ( sPrevKF_._vKeyPoints[ nKeyPointIdxRef ].pt.x );
		int nYRef = cvRound ( sPrevKF_._vKeyPoints[ nKeyPointIdxRef ].pt.y );

		int nDepthIdxCur = nYCur * __aKinectW[_uResolution] * 3 + nXCur * 3;
		int nDepthIdxRef = nYRef * __aKinectW[_uResolution] * 3 + nXRef * 3;

		if ( !boost::math::isnan<float>( _pCurrentPts[ nDepthIdxCur + 2 ] ) && !boost::math::isnan<float> (_pReferencePts[ nDepthIdxRef + 2 ]  ) ){
			float fDist = btl::utility::norm3<float>( _pCurrentPts + nDepthIdxCur, _pReferencePts + nDepthIdxRef, _eimRw.data(), _eivTw.data() );
			if(  fDist < dDistanceThreshold_ ) {
				_vDepthIdxCur  .push_back ( nDepthIdxCur );
				_vDepthIdxRef  .push_back ( nDepthIdxRef );
				_vSelectedPairs.push_back ( nKeyPointIdxCur );
				_vSelectedPairs.push_back ( nKeyPointIdxRef );
			}//if(  fDist < dDistanceThreshold_ ) 
		}//if ( !boost::math::isnan<float>( _pCurrentPts[ nDepthIdxCur + 2 ] ) && !boost::math::isnan<float> (_pReferencePts[ nDepthIdxRef + 2 ]  ) )
	}//for ( std::vector< cv::DMatch >::const_iterator cit = _vMatches.begin(); cit != _vMatches.end(); cit++ )

  ////for visualize the point correspondences calculated
  //      cv::Mat cvmCorr  ( sReferenceKF_._acvmShrPtrPyrRGBs[0]->cols + sReferenceKF_._acvmShrPtrPyrRGBs[0]->rows, sReferenceKF_._acvmShrPtrPyrRGBs[0]->cols, CV_8UC3 );
  //      cv::Mat cvmCorr2 ( sReferenceKF_._acvmShrPtrPyrRGBs[0]->rows + _acvmShrPtrPyrRGBs[0]->rows, sReferenceKF_._acvmShrPtrPyrRGBs[0]->cols, CV_8UC3 );
  //      
  //      cv::Mat roi1 ( cvmCorr, cv::Rect ( 0, 0, _acvmShrPtrPyrRGBs[0]->cols, _acvmShrPtrPyrRGBs[0]->rows ) );
  //      cv::Mat roi2 ( cvmCorr, cv::Rect ( 0, _acvmShrPtrPyrRGBs[0]->rows, sReferenceKF_._acvmShrPtrPyrRGBs[0]->cols, sReferenceKF_._acvmShrPtrPyrRGBs[0]->rows ) );
  //      _acvmShrPtrPyrRGBs[0]->copyTo ( roi1 );
  //      sReferenceKF_._acvmShrPtrPyrRGBs[0]->copyTo ( roi2 );
  //      
  //      static CvScalar colors = {{255, 255, 255}};
  //      int i = 0;
  //      int nKey;
  //      cv::namedWindow ( "myWindow", 1 );
  //      
  //      while ( true ) {
  //          cvmCorr.copyTo ( cvmCorr2 );
  //          cv::line ( cvmCorr2, _vKeyPoints[ _vSelectedPairs[i] ].pt, cv::Point ( sReferenceKF_._vKeyPoints [ _vSelectedPairs[i+1] ].pt.x, sReferenceKF_._vKeyPoints [ _vSelectedPairs[i+1] ].pt.y + _acvmShrPtrPyrRGBs[0]->rows ), colors );
  //          cv::imshow ( "myWindow", cvmCorr2 );
  //          nKey = cv::waitKey ( 30 );
  //      
  //          if ( nKey == 32 ){
  //              i += 2;
  //              if ( i > _vSelectedPairs.size() ){
  //                  break;
  //              }
  //          }
  //      
  //          if ( nKey == 27 ){
  //              break;
  //          }
  //      }
                
    int nSize = _vDepthIdxCur.size(); 
	PRINT(nSize);
    Eigen::MatrixXf eimCurCam ( 3, nSize ), eimRefWorld ( 3, nSize );
    std::vector< int >::const_iterator cit_Cur = _vDepthIdxCur.begin();
    std::vector< int >::const_iterator cit_Ref = _vDepthIdxRef.begin();

    for ( int i = 0 ; cit_Cur != _vDepthIdxCur.end(); cit_Cur++, cit_Ref++ ){
        eimCurCam ( 0, i ) = _pCurrentPts[ *cit_Cur     ];
        eimCurCam ( 1, i ) = _pCurrentPts[ *cit_Cur + 1 ];
        eimCurCam ( 2, i ) = _pCurrentPts[ *cit_Cur + 2 ];
        eimRefWorld ( 0, i ) = _pReferencePts[ *cit_Ref     ];
        eimRefWorld ( 1, i ) = _pReferencePts[ *cit_Ref + 1 ];
        eimRefWorld ( 2, i ) = _pReferencePts[ *cit_Ref + 2 ];
        i++;
    }
    float dS2;
    float fErrorBest = btl::utility::absoluteOrientation < float > ( eimRefWorld, eimCurCam , false, &_eimRw, &_eivTw, &dS2 ); // eimB_ = R * eimA_ + T;

	//PRINT ( fErrorBest );
	//PRINT ( _eimR );
	//PRINT ( _eivT );
	double dThreshold = fErrorBest;
        
    /*
    if ( nSize > 30 ) {
                            
            // random generator
            boost::mt19937 rng;
            boost::uniform_real<> gen ( 0, 1 );
            boost::variate_generator< boost::mt19937&, boost::uniform_real<> > dice ( rng, gen );
            double dError;
            Eigen::Matrix3d eimR;
            Eigen::Vector3d eivT;
            double dS;
            std::vector< int > vVoterIdx;
            Eigen::Matrix3d eimRBest;
            Eigen::Vector3d eivTBest;
            std::vector< int > vVoterIdxBest;
            int nMax = 0;
            std::vector < int > vRndIdx;
            Eigen::MatrixXd eimXTmp ( 3, 5 ), eimYTmp ( 3, 5 );
            
            for ( int n = 0; n < 500; n++ ) {
                select5Rand (  eimRefWorld, eimCurCam, dice, &eimYTmp, &eimXTmp );
                dError = btl::utility::absoluteOrientation < double > ( eimYTmp, eimXTmp, false, &eimR, &eivT, &dS );
            
                if ( dError > dThreshold ) {
                    continue;
                }
            
                //voting
                int nVotes = voting ( eimRefWorld, eimCurCam, eimR, eivT, dThreshold, &vVoterIdx );
                if ( nVotes > eimCurCam.cols() *.75 ) {
                    nMax = nVotes;
                    eimRBest = eimR;
                    eivTBest = eivT;
                    vVoterIdxBest = vVoterIdx;
                    break;
                }
            
                if ( nVotes > nMax ){
                    nMax = nVotes;
                    eimRBest = eimR;
                    eivTBest = eivT;
                    vVoterIdxBest = vVoterIdx;
                }
            }
            
            if ( nMax <= 6 ){
            	std::cout << "try increase the threshould" << std::endl;
                return fErrorBest;
            }
            
            Eigen::MatrixXd eimXInlier ( 3, vVoterIdxBest.size() );
            Eigen::MatrixXd eimYInlier ( 3, vVoterIdxBest.size() );
            selectInlier ( eimRefWorld, eimCurCam, vVoterIdxBest, &eimYInlier, &eimXInlier );
            fErrorBest = btl::utility::absoluteOrientation < double > (  eimYInlier , eimXInlier , false, &eimRNew, &eivTNew, &fS2 );
            
            PRINT ( nMax );
            PRINT ( fErrorBest );
            //PRINT ( _eimR );
            //PRINT ( _eivT );
            *pInliers_ = (unsigned short)nMax;
        }//if*/
    
	//apply new pose
	updateMVInv();
    return fErrorBest;
}// calcRT

double btl::kinect::CKeyFrame::calcRTOrb ( const CKeyFrame& sPrevKF_, const unsigned short sLevel_ , const double dDistanceThreshold_, unsigned short* pInliers_) {
	// - The reference frame must contain a calibrated Rw and Tw. 
	// - The point cloud in the reference frame must be transformed into the world coordinate system.
	// - The current frame's Rw and Tw must be initialized as the reference's Rw Tw. (This is for fDist = norm3<float>() ) 
	// - The point cloud in the current frame must be in the camera coordinate system.
	BTL_ASSERT(sPrevKF_._vKeyPoints.size()>10,"extractSurfFeatures() Too less SURF features detected in the reference frame")
	//matching from current to reference
	cv::gpu::BruteForceMatcher_GPU< cv::HammingLUT > cBruteMatcher;
	cBruteMatcher.match(_cvgmDescriptors, sPrevKF_._cvgmDescriptors, _vMatches);  
	PRINT(_vMatches.size());
	//std::sort( _vMatches.begin(), _vMatches.end() );
	//if (_vMatches.size()> 300) { _vMatches.erase( _vMatches.begin()+ 300, _vMatches.end() ); }
	//CHECK ( !_vMatches.empty(), "SKeyFrame::calcRT() _vMatches should not calculated." );
	//calculate the R and T
	//search for pairs of correspondences with depth data available.
	const float*const  _pCurrPts = (const float*)         _acvmShrPtrPyrPts[sLevel_]->data;
	const float*const  _pPrevPts = (const float*)sPrevKF_._acvmShrPtrPyrPts[sLevel_]->data;
	std::vector< int > _vDepthIdxCur, _vDepthIdxRef, _vSelectedPairs;
	for ( std::vector< cv::DMatch >::const_iterator cit = _vMatches.begin(); cit != _vMatches.end(); cit++ ) {
		int nKeyPointIdxCur = cit->queryIdx;
		int nKeyPointIdxRef = cit->trainIdx;

		int nXCur = cvRound ( 		   _vKeyPoints[ nKeyPointIdxCur ].pt.x );
		int nYCur = cvRound ( 		   _vKeyPoints[ nKeyPointIdxCur ].pt.y );
		int nXRef = cvRound ( sPrevKF_._vKeyPoints[ nKeyPointIdxRef ].pt.x );
		int nYRef = cvRound ( sPrevKF_._vKeyPoints[ nKeyPointIdxRef ].pt.y );

		int nDepthIdxCur = nYCur * __aKinectW[_uResolution] * 3 + nXCur * 3;
		int nDepthIdxRef = nYRef * __aKinectW[_uResolution] * 3 + nXRef * 3;

		if ( !boost::math::isnan<float>( _pCurrPts[ nDepthIdxCur + 2 ] ) && !boost::math::isnan<float> (_pPrevPts[ nDepthIdxRef + 2 ]  ) ){
			float fDist = btl::utility::norm3<float>( _pCurrPts + nDepthIdxCur, _pPrevPts + nDepthIdxRef, _eimRw.data(), _eivTw.data() );
			if(  fDist < dDistanceThreshold_ ) {
				_vDepthIdxCur  .push_back ( nDepthIdxCur );
				_vDepthIdxRef  .push_back ( nDepthIdxRef );
				_vSelectedPairs.push_back ( nKeyPointIdxCur );
				_vSelectedPairs.push_back ( nKeyPointIdxRef );
			}//if(  fDist < dDistanceThreshold_ ) 
		}//if ( !boost::math::isnan<float>( _pCurrPts[ nDepthIdxCur + 2 ] ) && !boost::math::isnan<float> (_pPrevPts[ nDepthIdxRef + 2 ]  ) )
	}//for ( std::vector< cv::DMatch >::const_iterator cit = _vMatches.begin(); cit != _vMatches.end(); cit++ )
            
    int nSize = _vDepthIdxCur.size(); 
	*pInliers_ = nSize;
	PRINT(nSize);
	//if nSize smaller than a threshould, quit
    Eigen::MatrixXf eimCurCam ( 3, nSize ), eimRefWorld ( 3, nSize );
    std::vector< int >::const_iterator cit_Cur = _vDepthIdxCur.begin();
    std::vector< int >::const_iterator cit_Ref = _vDepthIdxRef.begin();

    for ( int i = 0 ; cit_Cur != _vDepthIdxCur.end(); cit_Cur++, cit_Ref++ ){
        eimCurCam ( 0, i ) = _pCurrPts[ *cit_Cur     ];
        eimCurCam ( 1, i ) = _pCurrPts[ *cit_Cur + 1 ];
        eimCurCam ( 2, i ) = _pCurrPts[ *cit_Cur + 2 ];
        eimRefWorld ( 0, i ) = _pPrevPts[ *cit_Ref     ];
        eimRefWorld ( 1, i ) = _pPrevPts[ *cit_Ref + 1 ];
        eimRefWorld ( 2, i ) = _pPrevPts[ *cit_Ref + 2 ];
        i++;
    }
    float fS2;
    float dErrorBest = btl::utility::absoluteOrientation < float > ( eimRefWorld, eimCurCam , false, &_eimRw, &_eivTw, &fS2 ); // eimB_ = R * eimA_ + T;

	//PRINT ( dErrorBest );
	//PRINT ( _eimR );
	//PRINT ( _eivT );
	float dThreshold = dErrorBest;
        
    /*
    if ( nSize > 30 ) {
                            
            // random generator
            boost::mt19937 rng;
            boost::uniform_real<> gen ( 0, 1 );
            boost::variate_generator< boost::mt19937&, boost::uniform_real<> > dice ( rng, gen );
            double dError;
            Eigen::Matrix3d eimR;
            Eigen::Vector3d eivT;
            double dS;
            std::vector< int > vVoterIdx;
            Eigen::Matrix3d eimRBest;
            Eigen::Vector3d eivTBest;
            std::vector< int > vVoterIdxBest;
            int nMax = 0;
            std::vector < int > vRndIdx;
            Eigen::MatrixXd eimXTmp ( 3, 5 ), eimYTmp ( 3, 5 );
            
            for ( int n = 0; n < 500; n++ ) {
                select5Rand (  eimRefWorld, eimCurCam, dice, &eimYTmp, &eimXTmp );
                dError = btl::utility::absoluteOrientation < double > ( eimYTmp, eimXTmp, false, &eimR, &eivT, &dS );
            
                if ( dError > dThreshold ) {
                    continue;
                }
            
                //voting
                int nVotes = voting ( eimRefWorld, eimCurCam, eimR, eivT, dThreshold, &vVoterIdx );
                if ( nVotes > eimCurCam.cols() *.75 ) {
                    nMax = nVotes;
                    eimRBest = eimR;
                    eivTBest = eivT;
                    vVoterIdxBest = vVoterIdx;
                    break;
                }
            
                if ( nVotes > nMax ){
                    nMax = nVotes;
                    eimRBest = eimR;
                    eivTBest = eivT;
                    vVoterIdxBest = vVoterIdx;
                }
            }
            
            if ( nMax <= 6 ){
            	std::cout << "try increase the threshould" << std::endl;
                return dErrorBest;
            }
            
            Eigen::MatrixXd eimXInlier ( 3, vVoterIdxBest.size() );
            Eigen::MatrixXd eimYInlier ( 3, vVoterIdxBest.size() );
            selectInlier ( eimRefWorld, eimCurCam, vVoterIdxBest, &eimYInlier, &eimXInlier );
            dErrorBest = btl::utility::absoluteOrientation < double > (  eimYInlier , eimXInlier , false, &eimRNew, &eivTNew, &fS2 );
            
            PRINT ( nMax );
            PRINT ( dErrorBest );
            //PRINT ( _eimR );
            //PRINT ( _eivT );
            *pInliers_ = (unsigned short)nMax;
        }//if*/
    
	//apply new pose
	updateMVInv();
    return dErrorBest;
}// calcRT

void btl::kinect::CKeyFrame::render3DPtsInLocalGL(btl::gl_util::CGLUtil::tp_ptr pGL_, const unsigned short uLevel_,const bool bRenderPlane_) const {
	//////////////////////////////////
	//for rendering the detected plane
	const unsigned char* pColor;
	const float* pLabel;
	if(bRenderPlane_){
		//if(NORMAL_CLUSTER ==_eClusterType){
		//	pLabel = (const short*)_acvmShrPtrNormalClusters[pGL_->_usPyrHeight]->data;
		//}
		//else if(DISTANCE_CLUSTER ==_eClusterType){
			pLabel = (const float*)_acvmShrPtrDistanceClusters[pGL_->_usLevel]->data;
		//}
	}
	//////////////////////////////////
	float dNx,dNy,dNz;
	float dX, dY, dZ;
	const float* pPt = (const float*) _acvmShrPtrPyrPts[uLevel_]->data;
	const float* pNl = (const float*) _acvmShrPtrPyrNls[uLevel_]->data;
	const unsigned char* pRGB = (const unsigned char*) _acvmShrPtrPyrRGBs[uLevel_]->data;
	// Generate the data
	if( pGL_ && pGL_->_bEnableLighting ){glEnable(GL_LIGHTING);}
	else                            	{glDisable(GL_LIGHTING);}
	for( unsigned int i = 0; i < btl::kinect::__aKinectWxH[uLevel_]; i++,pRGB+=3,pNl+=3,pPt+=3){
		//////////////////////////////////
		//for rendering the detected plane
		if(bRenderPlane_ && pLabel[i]>0){
			pColor = btl::utility::__aColors[int(pLabel[i])/*+_nColorIdx*/%BTL_NUM_COLOR];
		}
		else{pColor = pRGB;}

		if(btl::utility::BTL_GL == _eConvention ){
			dNx = pNl[0];		dNy = pNl[1];		dNz = pNl[2];
			dX =  pPt[0];		dY =  pPt[1];		dZ =  pPt[2];
		}
		else if(btl::utility::BTL_CV == _eConvention ){
			dNx = pNl[0];		dNy =-pNl[1];		dNz =-pNl[2];
			dX =  pPt[0];		dY = -pPt[1];		dZ = -pPt[2];
		}
		else{ BTL_THROW("render3DPts() shouldnt be here!");	}
		if ( pGL_ )	{pGL_->renderDisk<float>(dX,dY,dZ,dNx,dNy,dNz,pColor,pGL_->_fSize*(uLevel_+1.f)*.5f,pGL_->_bRenderNormal); }
		else { glColor3ubv ( pColor ); glVertex3f ( dX, dY, dZ );}
	}
	return;
} 
void btl::kinect::CKeyFrame::gpuRenderPtsInWorldCVCV(btl::gl_util::CGLUtil::tp_ptr pGL_,const ushort usPyrLevel_){

	if( pGL_ && pGL_->_bEnableLighting ){glEnable(GL_LIGHTING);}
	else                            	{glDisable(GL_LIGHTING);}
	glPointSize(0.1f*(usPyrLevel_+1)*20);
	if (usPyrLevel_ >= _uPyrHeight) return;

	pGL_->gpuMapPtResources(*_acvgmShrPtrPyrPts[usPyrLevel_],usPyrLevel_);
	pGL_->gpuMapNlResources(*_acvgmShrPtrPyrNls[usPyrLevel_],usPyrLevel_);
	pGL_->gpuMapRGBResources(*_acvgmShrPtrPyrRGBs[usPyrLevel_],usPyrLevel_);
	glDrawArrays(GL_POINTS, 0, btl::kinect::__aKinectWxH[usPyrLevel_] );
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

}//gpuRenderVoxelInWorldCVGL()
void btl::kinect::CKeyFrame::gpuRender3DPtsInLocalCVGL(btl::gl_util::CGLUtil::tp_ptr pGL_,const ushort usColorIdx_, const unsigned short uLevel_, const bool bRenderPlane_) const {
	//////////////////////////////////
	//for rendering the detected plane
	const unsigned char* pColor/* = (const unsigned char*)_pVS->_vcvmPyrRGBs[_uPyrHeight-1]->data*/;
	const float* pLabel;
	if(bRenderPlane_){
		/*if( NORMAL_CLUSTER ==_eClusterType){
			pLabel = (const short*)_acvmShrPtrNormalClusters[pGL_->_usPyrHeight]->data;
		}
		else if( DISTANCE_CLUSTER ==_eClusterType){*/
			pLabel = (const float*)_acvmShrPtrDistanceClusters[pGL_->_usLevel]->data;
		//}
	}
	//////////////////////////////////
	_acvgmShrPtrAA[uLevel_]->setTo(0);
	btl::device::cudaNormalSetRotationAxisCVGL(*_acvgmShrPtrPyrNls[uLevel_],&*_acvgmShrPtrAA[uLevel_]);
	_acvgmShrPtrAA[uLevel_]->download(*_acvmShrPtrAA[uLevel_]);
	//////////////////////////////////
	const float* pPt = (const float*) _acvmShrPtrPyrPts[uLevel_]->data;
	const float* pAA = (const float*) _acvmShrPtrAA[uLevel_]->data;
	const unsigned char* pRGB = (const unsigned char*) _acvmShrPtrPyrRGBs[uLevel_]->data;
	// Generate the data
	if( pGL_ && pGL_->_bEnableLighting ){glEnable(GL_LIGHTING);}
	else                            	{glDisable(GL_LIGHTING);}
	for( unsigned int i = 0; i < btl::kinect::__aKinectWxH[uLevel_]; i++,pRGB+=3,pAA+=3,pPt+=3){
		//////////////////////////////////
		//for rendering the detected plane
		if(bRenderPlane_ && pLabel[i]>0){
			pColor = btl::utility::__aColors[int(pLabel[i])+usColorIdx_%BTL_NUM_COLOR];
		}//render planes
		else{
			pColor = pRGB;
		}//render original color
		if(pGL_) pGL_->renderDiskFastGL<float>(pPt[0],-pPt[1],-pPt[2],pAA[2],pAA[0],pAA[1],pColor,pGL_->_fSize*(uLevel_+1.f)*.5f,pGL_->_bRenderNormal);
	}
	return;
} 

void btl::kinect::CKeyFrame::renderPlanesInWorld(btl::gl_util::CGLUtil::tp_ptr pGL_, int nColorIdx_, const unsigned short usLevel_) const
{
	//////////////////////////////////
	const float* pNl = (const float*) _acvmShrPtrPyrNls[usLevel_]->data;
	const float* pPt = (const float*) _acvmShrPtrPyrPts[usLevel_]->data;
	const uchar* pRGB = (const uchar*)_acvmShrPtrPyrRGBs[usLevel_]->data;
	//render detected plane
	const float* pDistNormalCluster = (const float*) _acvmShrPtrDistanceClusters[usLevel_]->data;
	//const short* pNormalCluster = (const short*) _acvmShrPtrNormalClusters[usLevel_]->data;
	const unsigned char* pColor;
	// Generate the data
	if( pGL_ && pGL_->_bEnableLighting ){glEnable(GL_LIGHTING);}
	else                            	{glDisable(GL_LIGHTING);}
	glPointSize(0.1f*(usLevel_+1)*20);
	glBegin(GL_POINTS);
	unsigned char ucColor[3];
	for (unsigned int uIdx = 0; uIdx < btl::kinect::__aKinectWxH[_uResolution+usLevel_]; uIdx++){
		if( pDistNormalCluster[uIdx]>0){
			int nColor = (int)pDistNormalCluster[uIdx];
			pColor = btl::utility::__aColors[(nColor+nColorIdx_)%BTL_NUM_COLOR];
			ucColor[0] = uchar(0.5*pRGB[0] + 0.5*pColor[0]);
			ucColor[1] = uchar(0.5*pRGB[1] + 0.5*pColor[1]);
			ucColor[2] = uchar(0.5*pRGB[2] + 0.5*pColor[2]);
		}//if render planes
		else{
			ucColor[0] = pRGB[0];
			ucColor[1] = pRGB[1];
			ucColor[2] = pRGB[2]; 
			//pColor = pRGB;
		}//if not
		glColor3ubv ( ucColor ); pRGB += 3;
		glVertex3fv ( pPt );  pPt  += 3;
		glNormal3fv ( pNl );  pNl  += 3;
	}
	glEnd();
}

void btl::kinect::CKeyFrame::renderPlanesInLocalGL(btl::gl_util::CGLUtil::tp_ptr pGL_, const unsigned short uLevel_) const
{
	float dNx,dNy,dNz;
	float dX, dY, dZ;
	const float* pPt = (const float*)_acvmShrPtrPyrPts[uLevel_]->data;
	const float* pNl = (const float*)_acvmShrPtrPyrNls[uLevel_]->data;
	/*const unsigned char* pColor = (const unsigned char*)_pVS->_vcvmPyrRGBs[_uPyrHeight-1]->data;*/
	const float* pLabel;
	//if(NORMAL_CLUSTER ==_eClusterType){
	//	//pLabel = (const short*)_pModel->_acvmShrPtrNormalClusters[pGL_->_usLevel]->data;
	//	pLabel = (const short*)_acvmShrPtrNormalClusters[uLevel_]->data;
	//}
	//else if(DISTANCE_CLUSTER ==_eClusterType){
		pLabel = (const float*)_acvmShrPtrDistanceClusters[uLevel_]->data;
	//}
	for( unsigned int i = 0; i < btl::kinect::__aKinectWxH[uLevel_]; i++,pNl+=3,pPt+=3){
		int nColor = pLabel[i];
		if(nColor<0) { continue; }
		const unsigned char* pColor = btl::utility::__aColors[nColor/*+_nColorIdx*/%BTL_NUM_COLOR];
		if(btl::utility::BTL_GL == _eConvention ){
			dNx = pNl[0];		dNy = pNl[1];		dNz = pNl[2];
			dX =  pPt[0];		dY =  pPt[1];		dZ =  pPt[2];
		}
		else if(btl::utility::BTL_CV == _eConvention ){
			dNx = pNl[0];		dNy =-pNl[1];		dNz =-pNl[2];
			dX =  pPt[0];		dY = -pPt[1];		dZ = -pPt[2];
		}
		else{ BTL_THROW("render3DPts() shouldnt be here!");	}
		if( fabs(dNx) + fabs(dNy) + fabs(dNz) > 0.000001 ) {
			if ( pGL_ )	{pGL_->renderDisk<float>(dX,dY,dZ,dNx,dNy,dNz,pColor,pGL_->_fSize*(uLevel_+1.f)*.5f,pGL_->_bRenderNormal); }
			else { glColor3ubv ( pColor ); glVertex3f ( dX, dY, dZ );}
		}
	}
	return;
}
void btl::kinect::CKeyFrame::renderPlaneObjsInLocalCVGL(btl::gl_util::CGLUtil::tp_ptr pGL_,const unsigned short uLevel_) const{
	//////////////////////////////////
	const float* pNl = (const float*) _acvmShrPtrPyrNls[uLevel_]->data;
	const float* pPt = (const float*) _acvmShrPtrPyrPts[uLevel_]->data;
	/*const unsigned char* pColor; */short sColor = 0;
	// Generate the data
	if( pGL_ && pGL_->_bEnableLighting ){glEnable(GL_LIGHTING);}
	else                            	{glDisable(GL_LIGHTING);}
	glPointSize(0.1f*(uLevel_+1)*20);
	glBegin(GL_POINTS);
	for(btl::geometry::tp_plane_obj_list::const_iterator citPlaneObj = _vPlaneObjsDistanceNormal[uLevel_].begin(); citPlaneObj!=_vPlaneObjsDistanceNormal[uLevel_].end();citPlaneObj++,sColor++){
		const unsigned char* pColor = btl::utility::__aColors[citPlaneObj->_uIdx+_nColorIdx%BTL_NUM_COLOR];
		renderASinglePlaneObjInLocalCVGL(pPt,pNl,citPlaneObj->_vIdx,pColor);
		/*
		for(std::vector<unsigned int>::const_iterator citIdx = citPlaneObj->_vIdx.begin(); citIdx != citPlaneObj->_vIdx.end(); citIdx++ ){
					unsigned int uIdx = *citIdx*3;
					glColor3ubv ( pColor ); 
					glVertex3f ( pPt[uIdx], -pPt[uIdx+1], -pPt[uIdx+2] ); 
					glNormal3f ( pNl[uIdx], -pNl[uIdx+1], -pNl[uIdx+2] );
				}// for each point*/
		
	}//for each plane object
	glEnd();
}
void btl::kinect::CKeyFrame::renderASinglePlaneObjInLocalCVGL(const float*const pPt_, const float*const pNl_, const std::vector<unsigned int>& vIdx_, const unsigned char* pColor_) const {
	for(std::vector<unsigned int>::const_iterator citIdx = vIdx_.begin(); citIdx != vIdx_.end(); citIdx++ ){
		unsigned int uIdx = *citIdx*3;
		glColor3ubv ( pColor_ ); 
		glVertex3f ( pPt_[uIdx], -pPt_[uIdx+1], -pPt_[uIdx+2] ); 
		glNormal3f ( pNl_[uIdx], -pNl_[uIdx+1], -pNl_[uIdx+2] );
	}// for each point
}
void btl::kinect::CKeyFrame::renderCameraInWorldCVCV( btl::gl_util::CGLUtil::tp_ptr pGL_, bool bRenderCamera_, const double& dSize_,const unsigned short uLevel_ ) {
	if (pGL_->_usPyrHeight!=_uPyrHeight) return;
	glPushMatrix();
	loadGLMVIn();
	if( _bIsReferenceFrame ){ glColor3d( 1, 0, 0 );glLineWidth(2); }
	else					{ glColor3d( 1, 1, 1 );glLineWidth(1); }
#if USE_PBO
		if(bRenderCamera_) pGL_->gpuMapRgb2PixelBufferObj(*_acvgmShrPtrPyrRGBs[pGL_->_usLevel],pGL_->_usLevel);
		_pRGBCamera->renderCameraInGLLocal ( pGL_->_auTexture[pGL_->_usLevel],*_acvmShrPtrPyrRGBs[uLevel_], dSize_, bRenderCamera_);
#else 
		if(bRenderCamera_) _pRGBCamera->LoadTexture(*_acvmShrPtrPyrRGBs[uLevel_],&pGL_->_auTexture[pGL_->_usLevel]);
		_pRGBCamera->renderCameraInGLLocal ( pGL_->_auTexture[pGL_->_usLevel],*_acvmShrPtrPyrRGBs[uLevel_], dSize_, bRenderCamera_);
#endif	
	glPopMatrix();
}

void btl::kinect::CKeyFrame::render3DPtsInWorldCVCV(btl::gl_util::CGLUtil::tp_ptr pGL_,const ushort usLevel_,int nColorIdx_, bool bRenderPlanes_){
	if(usLevel_>=_uPyrHeight) return;
	//////////////////////////////////
	const float* pNl = (const float*) _acvmShrPtrPyrNls[usLevel_]->data;
	const float* pPt = (const float*) _acvmShrPtrPyrPts[usLevel_]->data;
	const uchar* pRGB = (const uchar*)_acvmShrPtrPyrRGBs[usLevel_]->data;
	//render detected plane
	const float* pDistNormalCluster = (const float*) _acvmShrPtrDistanceClusters[usLevel_]->data;
	//const short* pNormalCluster = (const short*) _acvmShrPtrNormalClusters[usLevel_]->data;
	const unsigned char* pColor;
	// Generate the data
	if( pGL_ && pGL_->_bEnableLighting ){glEnable(GL_LIGHTING);}
	else                            	{glDisable(GL_LIGHTING);}
	glPointSize(0.1f*(usLevel_+1)*20);
	glBegin(GL_POINTS);
	for (unsigned int uIdx = 0; uIdx < btl::kinect::__aKinectWxH[_uResolution+usLevel_]; uIdx++){
		if(bRenderPlanes_ && pDistNormalCluster[uIdx]>0){
			int nColor = (int)pDistNormalCluster[uIdx];
			pColor = btl::utility::__aColors[(nColor+nColorIdx_)%BTL_NUM_COLOR];
		}//if render planes
		else{
			pColor = pRGB;
		}//if not
		glColor3ubv ( pColor ); pRGB += 3;
		glVertex3fv ( pPt );  pPt  += 3;
		glNormal3fv ( pNl );  pNl  += 3;
	}
	glEnd();
}

void btl::kinect::CKeyFrame::renderASinglePlaneObjInWorldCVCV(const float*const pPt_, const float*const pNl_, const std::vector<unsigned int>& vIdx_, const unsigned char* pColor_) const {
	for(std::vector<unsigned int>::const_iterator citIdx = vIdx_.begin(); citIdx != vIdx_.end(); citIdx++ ){
		unsigned int uIdx = *citIdx*3;
		glColor3ubv ( pColor_ ); 
		glVertex3f ( pPt_[uIdx],pPt_[uIdx+1],pPt_[uIdx+2] ); 
		glNormal3f ( pNl_[uIdx],pNl_[uIdx+1],pNl_[uIdx+2] );
	}// for each point
}

void btl::kinect::CKeyFrame::selectInlier ( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, const std::vector< int >& vVoterIdx_, Eigen::MatrixXd* peimXInlier_, Eigen::MatrixXd* peimYInlier_ ) {
	CHECK ( vVoterIdx_.size() == peimXInlier_->cols(), " vVoterIdx_.size() must be equal to peimXInlier->cols(). " );
	CHECK ( vVoterIdx_.size() == peimYInlier_->cols(), " vVoterIdx_.size() must be equal to peimYInlier->cols(). " );
	std::vector< int >::const_iterator cit = vVoterIdx_.begin();

	for ( int i = 0; cit != vVoterIdx_.end(); cit++, i++ ) {
		( *peimXInlier_ ).col ( i ) = eimX_.col ( *cit );
		( *peimYInlier_ ).col ( i ) = eimY_.col ( *cit );
	}
	return;
}

int btl::kinect::CKeyFrame::voting ( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, 
	const Eigen::Matrix3d& eimR_, const Eigen::Vector3d& eivV_, const double& dThreshold, std::vector< int >* pvVoterIdx_ ) {
	int nV = 0;
	pvVoterIdx_->clear();

	for ( int i = 0; i < eimX_.cols(); i++ ) {
		Eigen::Vector3d vX = eimX_.col ( i );
		Eigen::Vector3d vY = eimY_.col ( i );
		Eigen::Vector3d vN = vY - eimR_ * vX - eivV_;

		if ( dThreshold > vN.norm() ) {
			pvVoterIdx_->push_back ( i );
			nV++;
		}
	}

	return nV;
}// end of function voting

void btl::kinect::CKeyFrame::select5Rand ( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, boost::variate_generator< boost::mt19937&, boost::uniform_real<> >& dice_, 
	Eigen::MatrixXd* eimXTmp_, Eigen::MatrixXd* eimYTmp_, std::vector< int >* pvIdx_/* = NULL */){
	//randomly select 5 paris of points
	CHECK ( eimX_.rows() == 3, "select5Rnd() eimX_ must have 3 rows" );
	CHECK ( eimY_.rows() == 3, "select5Rnd() eimY_ must have 3 rows" );
	CHECK ( eimX_.cols() == eimY_.cols(), "select5Rnd() eimX_ and eimY_ must contain the same # of cols" );
	CHECK ( eimXTmp_->rows() == 3, "select5Rnd() eimXTmp_ must have 3 rows" );
	CHECK ( eimYTmp_->cols() == 5, "select5Rnd() eimYTmp_ must have 5 cols" );

	if ( eimX_.cols() < 6 ) {
		return;
	}

	if ( pvIdx_ ) {
		pvIdx_->clear();
	}

	// generate 5 non-repeat random index
	std::list< int > lIdx;

	for ( int i = 0; i < eimX_.cols(); i++ )  {
		lIdx.push_back ( i );
	}

	//PRINT ( lIdx );
	std::list< int >::iterator it_Idx;
	double dRand;
	int nIdx;

	for ( int i = 0; i < 5; i++ ) {
		// generate 5 non-repeat random index
		dRand = dice_();
		nIdx = int ( dRand * lIdx.size() - 1 + .5 );
		//locate inside the list
		it_Idx = lIdx.begin();

		while ( nIdx-- > 0 )   {
			it_Idx++;
		}

		( *eimXTmp_ ) ( 0, i ) = eimX_ ( 0, * it_Idx );
		( *eimXTmp_ ) ( 1, i ) = eimX_ ( 1, * it_Idx );
		( *eimXTmp_ ) ( 2, i ) = eimX_ ( 2, * it_Idx );

		( *eimYTmp_ ) ( 0, i ) = eimY_ ( 0, * it_Idx );
		( *eimYTmp_ ) ( 1, i ) = eimY_ ( 1, * it_Idx );
		( *eimYTmp_ ) ( 2, i ) = eimY_ ( 2, * it_Idx );

		if ( pvIdx_ ) {	//PRINT( *it_Idx );
			pvIdx_->push_back ( *it_Idx );
		}
		lIdx.erase ( it_Idx );	//PRINT ( lIdx );
	}
	return;
}//end of select5Rand()
void btl::kinect::CKeyFrame::gpuTransformToWorldCVCV(const ushort usLevel_){
	if (usLevel_>=_uPyrHeight) return;
	btl::device::transformLocalToWorldCVCV(_eimRw.data(),_eivTw.data(),&*_acvgmShrPtrPyrPts[usLevel_],&*_acvgmShrPtrPyrNls[usLevel_]);
	_acvgmShrPtrPyrPts[usLevel_]->download(*_acvmShrPtrPyrPts[usLevel_]);
	_acvgmShrPtrPyrNls[usLevel_]->download(*_acvmShrPtrPyrNls[usLevel_]);
#if !USE_PBO
#endif
}//gpuTransformToWorldCVCV()

void btl::kinect::CKeyFrame::gpuTransformToWorldCVCV(){
	for (ushort usI=0;usI<_uPyrHeight;usI++) {
		gpuTransformToWorldCVCV(usI);
	}
}//gpuTransformToWorldCVCV()

void btl::kinect::CKeyFrame::gpuDetectPlane (const short usLevel_){
	//get next frame
	BTL_ASSERT(btl::utility::BTL_CV == _eConvention, "CKeyFrame data convention must be opencv convention");
	btl::geometry::tp_plane_obj_list vPlaneObjsNormalClusters,vPlaneObsNormalDistanceClusters;
	//clear previous plane objs
	_vPlaneObjsDistanceNormal[usLevel_].clear();
	//cluster the top pyramid
	_sNormalHist.gpuClusterNormal(*_acvgmShrPtrPyrNls[usLevel_],*_acvmShrPtrPyrNls[usLevel_],usLevel_,&*_acvmShrPtrNormalClusters[usLevel_],&vPlaneObjsNormalClusters);
	//enforce position continuity
	_sDistanceHist.clusterDistanceHist(*_acvmShrPtrPyrPts[usLevel_],*_acvmShrPtrPyrNls[usLevel_],usLevel_,vPlaneObjsNormalClusters,&*_acvmShrPtrDistanceClusters[usLevel_],&vPlaneObsNormalDistanceClusters);
	//merge clusters according to avg normal and position.
	btl::geometry::mergePlaneObj(&vPlaneObsNormalDistanceClusters,&*_acvmShrPtrDistanceClusters[usLevel_]);
	//spacial continuity constraint
	//btl::geometry::separateIntoDisconnectedRegions(&*_acvmShrPtrDistanceClusters[usLevel_]);
	/*
	//recalc planes
	typedef btl::geometry::tp_plane_obj_list::iterator tp_plane_obj_list_iterator;
	typedef std::map< unsigned int, tp_plane_obj_list_iterator > tp_plane_obj_map;
	tp_plane_obj_map mPlaneObjs; //key is plane id, 

	const float *pLabel = (float*) _acvmShrPtrDistanceClusters[usLevel_]->data; 
	const float *pNormal= (float*) _acvmShrPtrPyrNls[usLevel_]->data;
	for (unsigned int uIdx = 0; uIdx < btl::kinect::__aKinectWxH[usLevel_]; uIdx++ ){
		if(pLabel[uIdx]>0){
			tp_plane_obj_map::iterator itMap = mPlaneObjs.find(pLabel[uIdx]);			unsigned int uNlIdx = uIdx*3; //get 3-channel index
			if( itMap == mPlaneObjs.end() ){
				_vPlaneObjsDistanceNormal[usLevel_].push_back(btl::geometry::tp_plane_obj());
				tp_plane_obj_list_iterator itPlaneObj = _vPlaneObjsDistanceNormal[usLevel_].end(); --itPlaneObj; //get the iterator point
				itPlaneObj->_eivAvgNormal += Eigen::Vector3d(pNormal[uNlIdx],pNormal[uNlIdx+1],pNormal[uNlIdx+2]);
				itPlaneObj->_vIdx.push_back(uIdx);
				itPlaneObj->_uIdx = pLabel[uIdx]; 
				mPlaneObjs[pLabel[uIdx]] = itPlaneObj;//insert the new plane into map
			}//it its a new plane, add a new plane obj into the map
			else{
				itMap->second->_eivAvgNormal += Eigen::Vector3d(pNormal[uNlIdx],pNormal[uNlIdx+1],pNormal[uNlIdx+2]);
				itMap->second->_vIdx.push_back(uIdx);
			}//if its an existing plane, accumulate avgnormal and store vertex index
		}//if pLabel[uIdx]>0
	}
	//calc the avgPosition
	const float* pPt= (float*) _acvmShrPtrPyrPts[usLevel_]->data;
	for (btl::geometry::tp_plane_obj_list::iterator itPlane = _vPlaneObjsDistanceNormal[usLevel_].begin(); itPlane!= _vPlaneObjsDistanceNormal[usLevel_].end(); itPlane++ ){
		//average the accumulated normals
		itPlane->_eivAvgNormal.normalize();
		//accumulate plane positions
		for(std::vector<unsigned int>::iterator itVertexID =  itPlane->_vIdx.begin(); itVertexID != itPlane->_vIdx.end(); itVertexID++ ){
			unsigned int uPtIdx = *itVertexID*3; //get 3-channel index
			itPlane->_dAvgPosition += pPt[uPtIdx]*itPlane->_eivAvgNormal(0) + pPt[uPtIdx+1]*itPlane->_eivAvgNormal(1)+ pPt[uPtIdx+2]*itPlane->_eivAvgNormal(2);
		}//for each vertex of the plane
		//average the plane position
		itPlane->_dAvgPosition /= itPlane->_vIdx.size();
	}//for each plane
	*/
	/*
	const float *pLabel = (float*) _acvmShrPtrDistanceClusters[usLevel_]->data; 
	const float *pNormal= (float*) _acvmShrPtrPyrNls[usLevel_]->data;
	typedef btl::geometry::tp_plane_obj_list::iterator tp_plane_obj_list_iterator;
	typedef std::map< unsigned int, tp_plane_obj_list_iterator > tp_plane_obj_map;
	tp_plane_obj_map mPlaneObjs; //key is plane id, 
	for (btl::geometry::tp_plane_obj_list::iterator itPlane = vPlaneObjsNormalClusters.begin(); itPlane!= vPlaneObjsNormalClusters.end(); itPlane++){
		for(std::vector<unsigned int>::iterator itVertexID =  itPlane->_vIdx.begin(); itVertexID != itPlane->_vIdx.end(); itVertexID++ ){
			tp_plane_obj_map::iterator itMap = mPlaneObjs.find(pLabel[*itVertexID]);			unsigned int uNlIdx = *itVertexID*3; //get 3-channel index
			if( itMap == mPlaneObjs.end() ){
				_vPlaneObjsDistanceNormal[usLevel_].push_back(btl::geometry::tp_plane_obj());
				tp_plane_obj_list_iterator itPlaneObj = _vPlaneObjsDistanceNormal[usLevel_].end(); --itPlaneObj; //get the iterator point
				itPlaneObj->_eivAvgNormal += Eigen::Vector3d(pNormal[uNlIdx],pNormal[uNlIdx+1],pNormal[uNlIdx+2]);
				itPlaneObj->_vIdx.push_back(*itVertexID);
				mPlaneObjs[pLabel[*itVertexID]] = itPlaneObj;//insert the new plane into map
			}//it its a new plane
			else{
				itMap->second->_eivAvgNormal += Eigen::Vector3d(pNormal[uNlIdx],pNormal[uNlIdx+1],pNormal[uNlIdx+2]);
				itMap->second->_vIdx.push_back(*itVertexID);
			}//it plane exists accumulates to avgnormal
		}//for each vertex in the plane
	}//for each plane
	//calc the avgPosition
	const float* pPt= (float*) _acvmShrPtrPyrPts[usLevel_]->data;
	for (btl::geometry::tp_plane_obj_list::iterator itPlane = _vPlaneObjsDistanceNormal[usLevel_].begin(); itPlane!= _vPlaneObjsDistanceNormal[usLevel_].end(); itPlane++ ){
		//average the accumulated normals
		itPlane->_eivAvgNormal.normalize();
		//accumulate plane positions
		for(std::vector<unsigned int>::iterator itVertexID =  itPlane->_vIdx.begin(); itVertexID != itPlane->_vIdx.end(); itVertexID++ ){
			unsigned int uPtIdx = *itVertexID*3; //get 3-channel index
			itPlane->_dAvgPosition += pPt[uPtIdx]*itPlane->_eivAvgNormal(0) + pPt[uPtIdx+1]*itPlane->_eivAvgNormal(1)+ pPt[uPtIdx+2]*itPlane->_eivAvgNormal(2);
		}//for each vertex of the plane
		//average the plane position
		itPlane->_dAvgPosition /= itPlane->_vIdx.size();
		//transform the planes into world coordinates
		btl::geometry::transformPlaneIntoWorldCVCV(*itPlane,_eimRw,_eivTw);
		//btl::geometry::transformPlaneIntoLocalCVCV(*itPlane,_eimRw,_eivTw);
	}//for each plane
	*/
	return;
}
void btl::kinect::CKeyFrame::transformPlaneObjsToWorldCVCV(const ushort usLevel_){
	//transform the planes into world coordinates
	for (btl::geometry::tp_plane_obj_list::iterator itPlane = _vPlaneObjsDistanceNormal[usLevel_].begin(); itPlane!= _vPlaneObjsDistanceNormal[usLevel_].end(); itPlane++ ){
		btl::geometry::transformPlaneIntoWorldCVCV(*itPlane,_eimRw,_eivTw);
		//btl::geometry::transformPlaneIntoLocalCVCV(*itPlane,_eimRw,_eivTw);
	}//for each plane
}
void btl::kinect::CKeyFrame::applyRelativePose( const CKeyFrame& sReferenceKF_ ){
	//1.when the Rw and Tw is: Rw * Cam_Ref + Tw = Cam_Cur
	//_eivTw = _eimRw*sReferenceKF_._eivTw + _eivTw;//1.order matters 
	//_eimRw = _eimRw*sReferenceKF_._eimRw;//2.
	//2.when the Rw and Tw is: Rw * World_Ref + Tw = World_cur
	_eivTw = sReferenceKF_._eivTw + sReferenceKF_._eimRw*_eivTw;//1.order matters 
	_eimRw = sReferenceKF_._eimRw*_eimRw;
	updateMVInv();
}

void btl::kinect::CKeyFrame::updateMVInv(){
	Eigen::Matrix4f mGLM1;	setView( &mGLM1 );
	_eimGLMVInv = mGLM1.inverse().eval();
}

bool btl::kinect::CKeyFrame::isMovedwrtReferencInRadiusM(const CKeyFrame* const pRefFrame_, double dRotAngleThreshold_, double dTranslationThreshold_){
	using namespace btl::utility; //for operator <<
	//rotation angle
	cv::Mat_<float> cvmRRef,cvmRCur;
	cvmRRef << pRefFrame_->_eimRw;
	cvmRCur << _eimRw;
	cv::Mat_<float> cvmRVecRef,cvmRVecCur;
	cv::Rodrigues(cvmRRef,cvmRVecRef);
	cv::Rodrigues(cvmRCur,cvmRVecCur);
	cvmRVecCur -= cvmRVecRef;
	//get translation vector
	Eigen::Vector3f eivCRef,eivCCur;
	eivCRef = - pRefFrame_->_eimRw * pRefFrame_->_eivTw;
	eivCCur = -             _eimRw *             _eivTw;
	eivCCur -= eivCRef;
	double dRot = cv::norm( cvmRVecCur, cv::NORM_L2 );
	double dTrn = eivCCur.norm();
	return ( dRot > dRotAngleThreshold_ || dTrn > dTranslationThreshold_);
}

void btl::kinect::CKeyFrame::gpuICP(const CKeyFrame* pPrevFrameWorld_,bool bUsePrevRTAsInitial_){
	// the point cloud in previous frame has been transformed into the world coordinate
	// the current frame is still in camera coordinate

	//define parameters
	const short asICPIterations[] = {10, 5, 4, 4};
	const float fDistThreshold = 0.10f; //meters
	const float fSinAngleThres_ = sin (20.f * 3.14159254f / 180.f);
	//get R,T of reference 
	Eigen::Matrix3f eimrmRwPrev = pPrevFrameWorld_->_eimRw.transpose();//because by default eimrmRwPrev is colume major
	Eigen::Vector3f eivTwPrev = pPrevFrameWorld_->_eivTw;
	pcl::device::Mat33&  devRwPrev = pcl::device::device_cast<pcl::device::Mat33> (eimrmRwPrev);
	float3& devTwPrev = pcl::device::device_cast<float3> (eivTwPrev);
	//get R,T of current frame
	Eigen::Matrix3f eimrmRwCur;
	Eigen::Vector3f eivTwCur;
	if (bUsePrevRTAsInitial_) {
		eimrmRwCur = eimrmRwPrev;   
		eivTwCur = eivTwPrev;
	}//if use referece R and T as inital R and T for ICP, 
	else{
		eimrmRwCur = _eimRw;//.transpose();   //because by default eimrmRwPrev is colume major
		eivTwCur = _eivTw;
	}//other wise just use, the R & T have been updated by calcRT() using appearance-based approach 

	//from low resolution to high
	for (short sPyrLevel = _uPyrHeight-1; sPyrLevel >= 0; sPyrLevel--){
		//	short sPyrLevel = 3;
	    for ( short sIter = 0; sIter < asICPIterations[sPyrLevel]; ++sIter ){
			//	short sIter = 0;
			//get R and T
			pcl::device::Mat33& devRwCurTrans = pcl::device::device_cast<pcl::device::Mat33> (eimrmRwCur);
			float3& devTwCur = pcl::device::device_cast<float3> (eivTwCur);
			cv::gpu::GpuMat cvgmSumBuf;
			//run ICP and reduction
			btl::device::registrationICP( pcl::device::Intr(_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v)(sPyrLevel),fDistThreshold,fSinAngleThres_,
				devRwCurTrans, devTwCur, devRwPrev, devTwPrev,
				*pPrevFrameWorld_->_acvgmShrPtrPyrPts[sPyrLevel],*pPrevFrameWorld_->_acvgmShrPtrPyrNls[sPyrLevel],
				&*_acvgmShrPtrPyrPts[sPyrLevel],&*_acvgmShrPtrPyrNls[sPyrLevel],
				&cvgmSumBuf);
			
			cv::Mat cvmSumBuf;
			cvgmSumBuf.download (cvmSumBuf);
			double* aHostTmp = (double*) cvmSumBuf.data;
			//declare A and b
			Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A;
			Eigen::Matrix<double, 6, 1> b;
			//retrieve A and b from cvmSumBuf
			short sShift = 0;
			for (int i = 0; i < 6; ++i){   // rows
				for (int j = i; j < 7; ++j) { // cols + b
					double value = aHostTmp[sShift++];
					if (j == 6)       // vector b
						b.data()[i] = value;
					else
						A.data()[j * 6 + i] = A.data()[i * 6 + j] = value;
				}//for each col
			}//for each row
			//checking nullspace
			double dDet = A.determinant ();
			if (fabs (dDet) < 1e-15 || dDet != dDet ){
				if (dDet != dDet) std::cout << "qnan" << std::endl;
				//reset ();
				return ;
			}//if dDet is rational
			//float maxc = A.maxCoeff();

			Eigen::Matrix<float, 6, 1> result = A.llt ().solve (b).cast<float>();
			//Eigen::Matrix<float, 6, 1> result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);

			float alpha = result (0);
			float beta  = result (1);
			float gamma = result (2);

			Eigen::Matrix3f Rinc = (Eigen::Matrix3f)Eigen::AngleAxisf (gamma, Eigen::Vector3f::UnitZ ()) * Eigen::AngleAxisf (beta, Eigen::Vector3f::UnitY ()) * Eigen::AngleAxisf (alpha, Eigen::Vector3f::UnitX ());
			Eigen::Vector3f tinc = result.tail<3> ();

			//compose
			//eivTwCur   = Rinc * eivTwCur + tinc;
			//eimrmRwCur = Rinc * eimrmRwCur;
			Eigen::Vector3f eivTinv = - eimrmRwCur.transpose()* eivTwCur;
			Eigen::Matrix3f eimRinv = eimrmRwCur.transpose();
			eivTinv = Rinc * eivTinv + tinc;
			eimRinv = Rinc * eimRinv;
			eivTwCur = - eimRinv.transpose() * eivTinv;
			eimrmRwCur = eimRinv.transpose();
		}//for each iteration
	}//for each pyramid level
	_eimRw = eimrmRwCur;
	_eivTw = eivTwCur;

	return;
}

void btl::kinect::CKeyFrame::constructPyramid(const float fSigmaSpace_, const float fSigmaDisparity_){
	//bilateral filtering in disparity 
	_acvgmShrPtrPyrDisparity[0]->setTo(std::numeric_limits<float>::quiet_NaN());
	btl::device::cudaDepth2Disparity(*_acvgmShrPtrPyrDepths[0], &*_acvgmShrPtrPyr32FC1Tmp[0]);
	btl::device::cudaBilateralFiltering(*_acvgmShrPtrPyr32FC1Tmp[0],fSigmaSpace_,fSigmaDisparity_,&*_acvgmShrPtrPyrDisparity[0]);
	btl::device::cudaDisparity2Depth(*_acvgmShrPtrPyrDisparity[0],&*_acvgmShrPtrPyrDepths[0]);
	//get pts and normals
	_acvgmShrPtrPyrPts[0]->setTo(std::numeric_limits<float>::quiet_NaN());
	btl::device::unprojectRGBCVm(*_acvgmShrPtrPyrDepths[0],_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v, 0,&*_acvgmShrPtrPyrPts[0]);
	_acvgmShrPtrPyrNls[0]->setTo(std::numeric_limits<float>::quiet_NaN());
	btl::device::cudaFastNormalEstimation(*_acvgmShrPtrPyrPts[0],&*_acvgmShrPtrPyrNls[0]);//_vcvgmPyrNls[0]);
	//_acvgmShrPtrPyrRGBs[0]->download(*_acvmShrPtrPyrRGBs[0]);
	//cv::gpu::cvtColor(*_acvgmShrPtrPyrRGBs[0],*_acvgmShrPtrPyrBWs[0],cv::COLOR_RGB2GRAY);
	//_acvgmShrPtrPyrBWs[0]->download(*_acvmShrPtrPyrBWs[0]);
	_acvgmShrPtrPyrPts[0]->download(*_acvmShrPtrPyrPts[0]);
	_acvgmShrPtrPyrNls[0]->download(*_acvmShrPtrPyrNls[0]);
	//down-sampling
	for( unsigned int i=1; i<_uPyrHeight; i++ )	{
		//images
		//_acvgmShrPtrPyrRGBs[i]->setTo(0);
		//cv::gpu::pyrDown(*_acvgmShrPtrPyrRGBs[i-1],*_acvgmShrPtrPyrRGBs[i]);
		//_acvgmShrPtrPyrRGBs[i]->download(*_acvmShrPtrPyrRGBs[i]);
		//cv::gpu::cvtColor(*_acvgmShrPtrPyrRGBs[i],*_acvgmShrPtrPyrBWs[i],cv::COLOR_RGB2GRAY);
		//_acvgmShrPtrPyrBWs[i]->download(*_acvmShrPtrPyrBWs[i]);
		//down-sample disparity
		_acvgmShrPtrPyr32FC1Tmp[i]->setTo(std::numeric_limits<float>::quiet_NaN());
		btl::device::cudaPyrDown( *_acvgmShrPtrPyrDisparity[i-1],fSigmaDisparity_,&*_acvgmShrPtrPyr32FC1Tmp[i]);
		_acvgmShrPtrPyrDisparity[i]->setTo(std::numeric_limits<float>::quiet_NaN());
		btl::device::cudaBilateralFiltering(*_acvgmShrPtrPyr32FC1Tmp[i],fSigmaSpace_,fSigmaDisparity_,&*_acvgmShrPtrPyrDisparity[i]);
		_acvgmShrPtrPyrDepths[i]->setTo(std::numeric_limits<float>::quiet_NaN());
		btl::device::cudaDisparity2Depth(*_acvgmShrPtrPyrDisparity[i],&*_acvgmShrPtrPyrDepths[i]);
		//get pts and normals
		_acvgmShrPtrPyrPts[i]->setTo(std::numeric_limits<float>::quiet_NaN());
		btl::device::unprojectRGBCVm(*_acvgmShrPtrPyrDepths[i],_pRGBCamera->_fFx,_pRGBCamera->_fFy,_pRGBCamera->_u,_pRGBCamera->_v, i,&*_acvgmShrPtrPyrPts[i] );
		_acvgmShrPtrPyrPts[i]->download(*_acvmShrPtrPyrPts[i]);
		_acvgmShrPtrPyrNls[i]->setTo(std::numeric_limits<float>::quiet_NaN());
		btl::device::cudaFastNormalEstimation(*_acvgmShrPtrPyrPts[i],&*_acvgmShrPtrPyrNls[i]);
		_acvgmShrPtrPyrNls[i]->download(*_acvmShrPtrPyrNls[i]);	
	}	
	//transform from local to world
	for( unsigned int i=0; i<_uPyrHeight; i++ )	{
		gpuTransformToWorldCVCV(i);
	}
	return;
}

void btl::kinect::CKeyFrame::applyClassifier(btl::gl_util::CGLUtil::tp_ptr pGL_, float fThreshold_, const unsigned short usLevel_)
{
	//////////////////////////////////
	const float* pNl = (const float*) _acvmShrPtrPyrNls[usLevel_]->data;
	const float* pPt = (const float*) _acvmShrPtrPyrPts[usLevel_]->data;
	const uchar* pRGB = (uchar*)_acvmShrPtrPyrRGBs[usLevel_]->data;
	cv::Mat cvmColor;
	cvmColor.create(btl::kinect::__aKinectH[_uResolution+usLevel_],btl::kinect::__aKinectW[_uResolution+usLevel_],CV_8UC3);
	_acvmShrPtrPyrRGBs[usLevel_]->copyTo(cvmColor);
	uchar* pTmp = (uchar*)cvmColor.data;
	const unsigned char* pColor;
	// Generate the data
	for (unsigned int c = 0; c< btl::kinect::__aKinectW[_uResolution+usLevel_]; c++)
	for (unsigned int r = 0; r< btl::kinect::__aKinectH[_uResolution+usLevel_]; r++){
		if(boost::math::isnan<float>(pPt[0])||c==0||r==0||c==btl::kinect::__aKinectW[_uResolution+usLevel_]-1||r==btl::kinect::__aKinectH[_uResolution+usLevel_]-1)
		{
			pTmp += 3;
			pRGB += 3;
			pPt  += 3;
			pNl  += 3;
			continue;
		}
		int nColor = 1;
		pColor = btl::utility::__aColors[0];
		const float* pLeft = pPt -3;
		const float* pRight= pPt +3;
		const float* pUp   = pPt +btl::kinect::__aKinectW[_uResolution+usLevel_]*3;
		const float* pDown = pPt -btl::kinect::__aKinectW[_uResolution+usLevel_]*3;
		const float* pUpLeft= pUp -3;
		const float* pUpRight= pUp +3;
		const float* pDownLeft= pDown -3;
		const float* pDownRight= pDown +3;
		short sCount=0;
		float fThreshold = fThreshold_;//*(1+usLevel_);
		float fDistance;
		fDistance = sqrt((pUpLeft[0]-pPt[0])*(pUpLeft[0]-pPt[0])+(pUpLeft[1]-pPt[1])*(pUpLeft[1]-pPt[1])+(pUpLeft[2]-pPt[2])*(pUpLeft[2]-pPt[2]));
		if(fDistance>fThreshold||boost::math::isnan<float>(fDistance))
			sCount++;
		fDistance = sqrt((pUpRight[0]-pPt[0])*(pUpRight[0]-pPt[0])+(pUpRight[1]-pPt[1])*(pUpRight[1]-pPt[1])+(pUpRight[2]-pPt[2])*(pUpRight[2]-pPt[2]));
		if(fDistance>fThreshold||boost::math::isnan<float>(fDistance))
			sCount++;
		fDistance = sqrt((pUp[0]-pPt[0])*(pUp[0]-pPt[0])+(pUp[1]-pPt[1])*(pUp[1]-pPt[1])+(pUp[2]-pPt[2])*(pUp[2]-pPt[2]));
		if(fDistance>fThreshold||boost::math::isnan<float>(fDistance))
			sCount++;
		fDistance = sqrt((pDown[0]-pPt[0])*(pDown[0]-pPt[0])+(pDown[1]-pPt[1])*(pDown[1]-pPt[1])+(pDown[2]-pPt[2])*(pDown[2]-pPt[2]));
		if(fDistance>fThreshold||boost::math::isnan<float>(fDistance))
			sCount++;
		fDistance = sqrt((pDownLeft[0]-pPt[0])*(pDownLeft[0]-pPt[0])+(pDownLeft[1]-pPt[1])*(pDownLeft[1]-pPt[1])+(pDownLeft[2]-pPt[2])*(pDownLeft[2]-pPt[2]));
		if(fDistance>fThreshold||boost::math::isnan<float>(fDistance))
			sCount++;
		fDistance = sqrt((pDownRight[0]-pPt[0])*(pDownRight[0]-pPt[0])+(pDownRight[1]-pPt[1])*(pDownRight[1]-pPt[1])+(pDownRight[2]-pPt[2])*(pDownRight[2]-pPt[2]));
		if(fDistance>fThreshold||boost::math::isnan<float>(fDistance))
			sCount++;
		if(sCount>2&&sCount<8){//if it is a border pixel
			pTmp[0] = 0.5*pRGB[0] + 0.5*pColor[0];
			pTmp[1] = 0.5*pRGB[1] + 0.5*pColor[1];
			pTmp[2] = 0.5*pRGB[2] + 0.5*pColor[2];
		}
		else{
			pTmp[0] = pRGB[0];
			pTmp[1] = pRGB[1];
			pTmp[2] = pRGB[2];
		}
		
		pTmp += 3;
		pRGB += 3;
		pPt  += 3;
		pNl  += 3;
	}//for each pixel
	cvmColor.copyTo(*_acvmShrPtrPyrRGBs[usLevel_]);
}

void btl::kinect::CKeyFrame::gpuBoundaryDetector(float fThreshold_, const unsigned short usLevel_)
{
	btl::device::boundaryDetector(fThreshold_,*_acvgmShrPtrPyrPts[usLevel_],*_acvgmShrPtrPyrNls[usLevel_],&*_acvgmShrPtrPyrRGBs[usLevel_]);
	_acvgmShrPtrPyrRGBs[usLevel_]->download(*_acvmShrPtrPyrRGBs[usLevel_]);
}

void btl::kinect::CKeyFrame::setRTTo(const CKeyFrame& cFrame_ ){
	//assign rotation and translation 
	_eimRw = cFrame_._eimRw;
	_eivTw = cFrame_._eivTw;
	updateMVInv();
}

void btl::kinect::CKeyFrame::assignRTfromGL(const btl::gl_util::CGLUtil* pGL_ ){
	if (pGL_){
		//assign rotation and translation 
		pGL_->getRTFromWorld2CamCV(&_eimRw,&_eivTw);
	}
	updateMVInv();
}