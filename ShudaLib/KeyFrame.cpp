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
//stl
#include <vector>
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

btl::utility::SNormalHist btl::kinect::CKeyFrame::_sNormalHist;
btl::utility::SDistanceHist btl::kinect::CKeyFrame::_sDistanceHist;
//btl::utility::tp_plane_obj_list btl::kinect::CKeyFrame::_vPlaneObjsDistanceNormal;
//btl::utility::tp_plane_obj_list btl::kinect::CKeyFrame::_vPlaneObjsNormal;
boost::shared_ptr<cv::Mat> btl::kinect::CKeyFrame::_acvmShrPtrAA[4];
boost::shared_ptr<cv::gpu::GpuMat> btl::kinect::CKeyFrame::_acvgmShrPtrAA[4];//for rendering


btl::kinect::CKeyFrame::CKeyFrame( btl::kinect::SCamera::tp_ptr pRGBCamera_ )
:_pRGBCamera(pRGBCamera_){
	//disparity
	for(int i=0; i<4; i++){
		int nRows = KINECT_HEIGHT>>i; 
		int nCols = KINECT_WIDTH>>i;
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
		_acvgmPyrDepths[i]	  .reset(new cv::gpu::GpuMat(nRows,nCols,CV_32FC1));
		//plane detection
		_acvmShrPtrNormalClusters[i].reset(new cv::Mat(nRows,nCols,CV_16SC1));
		_acvmShrPtrDistanceClusters[i].reset(new cv::Mat(nRows,nCols,CV_16SC1));
	}

	_eConvention = btl::utility::BTL_CV;
	_eimRw.setIdentity();
	Eigen::Vector3d eivC (0.,0.,-1.8); //camera location in the world cv-convention
	_eivTw = -_eimRw.transpose()*eivC;
	_bIsReferenceFrame = false;
	_bRenderPlane = false;
	_bMerge = false;
	_bGPURender = false;
	_pGL = NULL;
	_eClusterType = NORMAL_CLUSTER;//DISTANCE_CLUSTER;
	_nColorIdx = 0;

	//rendering
	glPixelStorei ( GL_UNPACK_ALIGNMENT, 1 );
	glGenTextures ( 1, &_uTexture );
}


void btl::kinect::CKeyFrame::copyTo( CKeyFrame* pKF_, const short sLevel_ ){
	//host
	_acvmShrPtrPyrPts[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrPts[sLevel_]);
	_acvmShrPtrPyrNls[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrNls[sLevel_]);
	_acvmShrPtrPyrRGBs[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrRGBs[sLevel_]);
	_acvmShrPtrPyrBWs[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrBWs[sLevel_]);
	//device
	_acvgmShrPtrPyrPts[sLevel_]->copyTo(*pKF_->_acvgmShrPtrPyrPts[sLevel_]);
	_acvgmShrPtrPyrNls[sLevel_]->copyTo(*pKF_->_acvgmShrPtrPyrNls[sLevel_]);
	_acvgmShrPtrPyrRGBs[sLevel_]->copyTo(*pKF_->_acvgmShrPtrPyrRGBs[sLevel_]);
	_acvgmShrPtrPyrBWs[sLevel_]->copyTo(*pKF_->_acvgmShrPtrPyrBWs[sLevel_]);
	pKF_->_eConvention = _eConvention;
}

void btl::kinect::CKeyFrame::copyTo( CKeyFrame* pKF_ ) {
	for(int i=0; i<4; i++) {
		copyTo(pKF_,i);
		pKF_->_vPlaneObjsDistanceNormal[i] = _vPlaneObjsDistanceNormal[i];

	}
	pKF_->_bIsReferenceFrame = _bIsReferenceFrame;
	pKF_->_eimRw = _eimRw;
	pKF_->_eivTw = _eivTw;
}

void btl::kinect::CKeyFrame::detectConnectionFromCurrentToReference ( CKeyFrame& sReferenceKF_, const short sLevel_ )  {
	boost::shared_ptr<cv::gpu::SURF_GPU> _pSurf(new cv::gpu::SURF_GPU(100));
	(*_pSurf)(*_acvgmShrPtrPyrBWs[sLevel_], cv::gpu::GpuMat(), _cvgmKeyPoints, _cvgmDescriptors);
	_pSurf->downloadKeypoints(_cvgmKeyPoints, _vKeyPoints);
	//from current to reference
	//_cvgmKeyPoints.copyTo(sReferenceKF_._cvgmKeyPoints); _cvgmDescriptors.copyTo(sReferenceKF_._cvgmDescriptors);
	(*_pSurf)(*sReferenceKF_._acvgmShrPtrPyrBWs[sLevel_], cv::gpu::GpuMat(), sReferenceKF_._cvgmKeyPoints, sReferenceKF_._cvgmDescriptors/*,true*/);//make use of provided keypoints
	_pSurf->downloadKeypoints(sReferenceKF_._cvgmKeyPoints, sReferenceKF_._vKeyPoints);
	//from reference to current
	//sReferenceKF_._cvgmKeyPoints.copyTo(_cvgmKeyPoints); sReferenceKF_._cvgmDescriptors.copyTo(_cvgmDescriptors);
	//(*_pSurf)(*_acvgmShrPtrPyrBWs[sLevel_], cv::gpu::GpuMat(), _cvgmKeyPoints, _cvgmDescriptors,true);
	//_pSurf->downloadKeypoints(_cvgmKeyPoints, _vKeyPoints);

	//matching from current to reference
	cv::gpu::BruteForceMatcher_GPU< cv::L2<float> > cBruteMatcher;
	cv::gpu::GpuMat cvgmTrainIdx, cvgmDistance;
	cBruteMatcher.matchSingle( this->_cvgmDescriptors,  sReferenceKF_._cvgmDescriptors, cvgmTrainIdx, cvgmDistance);
	cv::gpu::BruteForceMatcher_GPU< cv::L2<float> >::matchDownload(cvgmTrainIdx, cvgmDistance, _vMatches);
	std::sort( _vMatches.begin(), _vMatches.end() );
	if (_vMatches.size()> 400) { _vMatches.erase( _vMatches.begin()+200, _vMatches.end() ); }
	return;
}

double btl::kinect::CKeyFrame::calcRT ( const CKeyFrame& sReferenceKF_, const unsigned short sLevel_ , unsigned short* pInliers_) {
	CHECK ( !_vMatches.empty(), "SKeyFrame::calcRT() _vMatches should not calculated." );
	//calculate the R and T
	//search for pairs of correspondences with depth data available.
	const float*const  _pCurrentPts = (const float*)              _acvmShrPtrPyrPts[sLevel_]->data;
	const float*const  _pReferencePts = (const float*)sReferenceKF_._acvmShrPtrPyrPts[sLevel_]->data;
	std::vector< int > _vDepthIdxCur, _vDepthIdxRef, _vSelectedPairs;
	for ( std::vector< cv::DMatch >::const_iterator cit = _vMatches.begin(); cit != _vMatches.end(); cit++ ) {
		int nKeyPointIdxCur = cit->queryIdx;
		int nKeyPointIdxRef = cit->trainIdx;

		int nXCur = cvRound ( 			    _vKeyPoints[ nKeyPointIdxCur ].pt.x );
		int nYCur = cvRound ( 			    _vKeyPoints[ nKeyPointIdxCur ].pt.y );
		int nXRef = cvRound ( sReferenceKF_._vKeyPoints[ nKeyPointIdxRef ].pt.x );
		int nYRef = cvRound ( sReferenceKF_._vKeyPoints[ nKeyPointIdxRef ].pt.y );

		int nDepthIdxCur = nYCur * 640 * 3 + nXCur * 3;
		int nDepthIdxRef = nYRef * 640 * 3 + nXRef * 3;

		if ( fabs ( _pCurrentPts[ nDepthIdxCur + 2 ] ) > 0.0001 && fabs (_pReferencePts[ nDepthIdxRef + 2 ] ) > 0.0001 ) {
			_vDepthIdxCur  .push_back ( nDepthIdxCur );
			_vDepthIdxRef  .push_back ( nDepthIdxRef );
			_vSelectedPairs.push_back ( nKeyPointIdxCur );
			_vSelectedPairs.push_back ( nKeyPointIdxRef );
		}
	}

	//PRINT ( _vSelectedPairs.size() );
		/*
        //for visualize the point correspondeneces calculated
        cv::Mat cvmCorr  ( sReferenceKF_._cvmRGB.rows + _cvmRGB.rows, sReferenceKF_._cvmRGB.cols, CV_8UC3 );
        cv::Mat cvmCorr2 ( sReferenceKF_._cvmRGB.rows + _cvmRGB.rows, sReferenceKF_._cvmRGB.cols, CV_8UC3 );
        
        cv::Mat roi1 ( cvmCorr, cv::Rect ( 0, 0, _cvmRGB.cols, _cvmRGB.rows ) );
        cv::Mat roi2 ( cvmCorr, cv::Rect ( 0, _cvmRGB.rows, sReferenceKF_._cvmRGB.cols, sReferenceKF_._cvmRGB.rows ) );
        _cvmRGB.copyTo ( roi1 );
        sReferenceKF_._cvmRGB.copyTo ( roi2 );
        
        static CvScalar colors = {{255, 255, 255}};
        int i = 0;
        int nKey;
        cv::namedWindow ( "myWindow", 1 );
        
        while ( true ) {
            cvmCorr.copyTo ( cvmCorr2 );
            cv::line ( cvmCorr2, _vKeyPoints[ _vSelectedPairs[i] ].pt, cv::Point ( sReferenceKF_._vKeyPoints [ _vSelectedPairs[i+1] ].pt.x, sReferenceKF_._vKeyPoints [ _vSelectedPairs[i+1] ].pt.y + _cvmRGB.rows ), colors );
            cv::imshow ( "myWindow", cvmCorr2 );
            nKey = cv::waitKey ( 30 );
        
            if ( nKey == 32 ){
                i += 2;
                if ( i > _vSelectedPairs.size() ){
                    break;
                }
            }
        
            if ( nKey == 27 ){
                break;
            }
        }*/
                
        int nSize = _vDepthIdxCur.size(); 
		PRINT(nSize);
        Eigen::MatrixXd eimCur ( 3, nSize ), eimRef ( 3, nSize );
        std::vector<  int >::const_iterator cit_Cur = _vDepthIdxCur.begin();
        std::vector<  int >::const_iterator cit_Ref = _vDepthIdxRef.begin();

        for ( int i = 0 ; cit_Cur != _vDepthIdxCur.end(); cit_Cur++, cit_Ref++ ){
            eimCur ( 0, i ) = _pCurrentPts[ *cit_Cur     ];
            eimCur ( 1, i ) = _pCurrentPts[ *cit_Cur + 1 ];
            eimCur ( 2, i ) = _pCurrentPts[ *cit_Cur + 2 ];
            eimRef ( 0, i ) = _pReferencePts[ *cit_Ref     ];
            eimRef ( 1, i ) = _pReferencePts[ *cit_Ref + 1 ];
            eimRef ( 2, i ) = _pReferencePts[ *cit_Ref + 2 ];
            i++;
        }
        double dS2;
        double dErrorBest = btl::utility::absoluteOrientation < double > ( eimRef, eimCur ,  false, &_eimRw, &_eivTw, &dS2 );
		//PRINT ( dErrorBest );
		//PRINT ( _eimR );
		//PRINT ( _eivT );
		double dThreshold = dErrorBest;
        //for ( int i = 0; i < 2; i++ )
        {
            if ( nSize > 10 ) {
                        
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
        
                for ( int n = 0; n < 5000; n++ ) {
                    select5Rand (  eimRef, eimCur, dice, &eimYTmp, &eimXTmp );
                    dError = btl::utility::absoluteOrientation < double > ( eimYTmp, eimXTmp, false, &eimR, &eivT, &dS );
        
                    if ( dError > dThreshold ) {
                        continue;
                    }
        
                    //voting
                    int nVotes = voting ( eimRef, eimCur, eimR, eivT, dThreshold, &vVoterIdx );
                    if ( nVotes > eimCur.cols() *.75 ) {
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
                selectInlier ( eimRef, eimCur, vVoterIdxBest, &eimYInlier, &eimXInlier );
                dErrorBest = btl::utility::absoluteOrientation < double > (  eimYInlier , eimXInlier , false, &_eimRw, &_eivTw, &dS2 );
        
                PRINT ( nMax );
                PRINT ( dErrorBest );
                //PRINT ( _eimR );
                //PRINT ( _eivT );
                *pInliers_ = (unsigned short)nMax;
            }//if
        }//for

    return dErrorBest;
}// calcRT

void btl::kinect::CKeyFrame::renderCameraInGLWorld( bool bRenderCamera_, bool bBW_, bool bRenderDepth_, const double& dSize_,const unsigned short uLevel_ ) {
	Eigen::Matrix4d mGLM1;
	setView( &mGLM1 );
	mGLM1 = mGLM1.inverse().eval();
	glPushMatrix();
	glMultMatrixd ( mGLM1.data() );
	if( _bIsReferenceFrame ){
		glColor3d( 1, 0, 0 );
		glLineWidth(2);
	}
	else{
		glColor3d( 1, 1, 1);
		glLineWidth(1);
	}
	//glColor4d( 1,1,1,0.5 );
	if(bBW_){
		if(bRenderCamera_)	_pRGBCamera->LoadTexture(*_acvmShrPtrPyrBWs[uLevel_],&_uTexture);
		_pRGBCamera->renderCameraInGLLocal ( _uTexture, *_acvmShrPtrPyrBWs[uLevel_], dSize_, bRenderCamera_);
	}else{
		if(bRenderCamera_)	_pRGBCamera->LoadTexture(*_acvmShrPtrPyrRGBs[uLevel_],&_uTexture);
		_pRGBCamera->renderCameraInGLLocal ( _uTexture,*_acvmShrPtrPyrRGBs[uLevel_], dSize_, bRenderCamera_);
	}
	//render dot clouds
	//if(_bRenderPlane) renderPlanesInLocalGL(uLevel_);
	//render3DPtsInLocalGL(uLevel_);//rendering detected plane as well
	if(bRenderDepth_){
		/*//if (_bGPURender) gpuRender3DPtsCVInLocalGL(uLevel_,_bRenderPlane);*/
		//else render3DPtsInLocalGL(uLevel_,_bRenderPlane);
		//if (_bRenderPlaneSeparately) //renderPlanesInLocalGL(2); 
		if(_bRenderPlane) renderPlaneObjsInLocalCVGL(uLevel_);
		else gpuRender3DPtsCVInLocalGL(uLevel_,_bRenderPlane);
	}
	glPopMatrix();
}

void btl::kinect::CKeyFrame::render3DPtsInLocalGL(const unsigned short uLevel_,const bool bRenderPlane_) const {
	//////////////////////////////////
	//for rendering the detected plane
	const unsigned char* pColor;
	const short* pLabel;
	if(bRenderPlane_){
		if(NORMAL_CLUSTER ==_eClusterType){
			pLabel = (const short*)_acvmShrPtrNormalClusters[_pGL->_uLevel]->data;
		}
		else if(DISTANCE_CLUSTER ==_eClusterType){
			pLabel = (const short*)_acvmShrPtrDistanceClusters[_pGL->_uLevel]->data;
		}
	}
	//////////////////////////////////
	float dNx,dNy,dNz;
	float dX, dY, dZ;
	const float* pPt = (const float*) _acvmShrPtrPyrPts[uLevel_]->data;
	const float* pNl = (const float*) _acvmShrPtrPyrNls[uLevel_]->data;
	const unsigned char* pRGB = (const unsigned char*) _acvmShrPtrPyrRGBs[uLevel_]->data;
	// Generate the data
	if( _pGL && _pGL->_bEnableLighting ){glEnable(GL_LIGHTING);}
	else                            	{glDisable(GL_LIGHTING);}
	for( unsigned int i = 0; i < btl::kinect::__aKinectWxH[uLevel_]; i++,pRGB+=3,pNl+=3,pPt+=3){
		//////////////////////////////////
		//for rendering the detected plane
		if(bRenderPlane_ && pLabel[i]>0){
			pColor = btl::utility::__aColors[pLabel[i]/*+_nColorIdx*/%BTL_NUM_COLOR];
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
		if ( _pGL )	{_pGL->renderDisk<float>(dX,dY,dZ,dNx,dNy,dNz,pColor,_pGL->_fSize*(uLevel_+1.f)*.5f,_pGL->_bRenderNormal); }
		else { glColor3ubv ( pColor ); glVertex3f ( dX, dY, dZ );}
	}
	return;
} 

void btl::kinect::CKeyFrame::gpuRender3DPtsCVInLocalGL(const unsigned short uLevel_, const bool bRenderPlane_) const {
	//////////////////////////////////
	//for rendering the detected plane
	const unsigned char* pColor/* = (const unsigned char*)_pVS->_vcvmPyrRGBs[_uPyrHeight-1]->data*/;
	const short* pLabel;
	if(bRenderPlane_){
		if(NORMAL_CLUSTER ==_eClusterType){
			pLabel = (const short*)_acvmShrPtrNormalClusters[_pGL->_uLevel]->data;
		}
		else if(DISTANCE_CLUSTER ==_eClusterType){
			pLabel = (const short*)_acvmShrPtrDistanceClusters[_pGL->_uLevel]->data;
		}
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
	if( _pGL && _pGL->_bEnableLighting ){glEnable(GL_LIGHTING);}
	else                            	{glDisable(GL_LIGHTING);}
	for( unsigned int i = 0; i < btl::kinect::__aKinectWxH[uLevel_]; i++,pRGB+=3,pAA+=3,pPt+=3){
		//////////////////////////////////
		//for rendering the detected plane
		if(bRenderPlane_ && pLabel[i]>0){
			pColor = btl::utility::__aColors[pLabel[i]/*+_nColorIdx*/%BTL_NUM_COLOR];
		}
		else{pColor = pRGB;}
		if(_pGL) _pGL->renderDiskFastGL<float>(pPt[0],-pPt[1],-pPt[2],pAA[2],pAA[0],pAA[1],pColor,_pGL->_fSize*(uLevel_+1.f)*.5f,_pGL->_bRenderNormal);
	}
	return;
} 

void btl::kinect::CKeyFrame::renderPlanesInLocalGL(const unsigned short uLevel_) const
{
	float dNx,dNy,dNz;
	float dX, dY, dZ;
	const float* pPt = (const float*)_acvmShrPtrPyrPts[uLevel_]->data;
	const float* pNl = (const float*)_acvmShrPtrPyrNls[uLevel_]->data;
	const unsigned char* pColor/* = (const unsigned char*)_pVS->_vcvmPyrRGBs[_uPyrHeight-1]->data*/;
	const short* pLabel;
	if(NORMAL_CLUSTER ==_eClusterType){
		//pLabel = (const short*)_pModel->_acvmShrPtrNormalClusters[_pGL->_uLevel]->data;
		pLabel = (const short*)_acvmShrPtrNormalClusters[uLevel_]->data;
	}
	else if(DISTANCE_CLUSTER ==_eClusterType){
		//pLabel = (const short*)_pModel->_acvmShrPtrDistanceClusters[_pGL->_uLevel]->data;
		pLabel = (const short*)_acvmShrPtrDistanceClusters[uLevel_]->data;
	}
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
			if ( _pGL )	{_pGL->renderDisk<float>(dX,dY,dZ,dNx,dNy,dNz,pColor,_pGL->_fSize*(uLevel_+1.f)*.5f,_pGL->_bRenderNormal); }
			else { glColor3ubv ( pColor ); glVertex3f ( dX, dY, dZ );}
		}
	}
	return;
}

void btl::kinect::CKeyFrame::renderPlaneObjsInLocalCVGL(const unsigned short uLevel_) const{
	//////////////////////////////////
	const float* pNl = (const float*) _acvmShrPtrPyrNls[uLevel_]->data;
	const float* pPt = (const float*) _acvmShrPtrPyrPts[uLevel_]->data;
	const unsigned char* pColor; short sColor = 0;
	// Generate the data
	if( _pGL && _pGL->_bEnableLighting ){glEnable(GL_LIGHTING);}
	else                            	{glDisable(GL_LIGHTING);}
	glPointSize(0.1f*(uLevel_+1)*20);
	glBegin(GL_POINTS);
	for(btl::geometry::tp_plane_obj_list::const_iterator citPlaneObj = _vPlaneObjsDistanceNormal[uLevel_].begin(); citPlaneObj!=_vPlaneObjsDistanceNormal[uLevel_].end();citPlaneObj++,sColor++){
		const unsigned char* pColor = btl::utility::__aColors[citPlaneObj->_usIdx+_nColorIdx%BTL_NUM_COLOR];
		for(std::vector<unsigned int>::const_iterator citIdx = citPlaneObj->_vIdx.begin(); citIdx != citPlaneObj->_vIdx.end(); citIdx++ ){
			unsigned int uIdx = *citIdx*3;
			glColor3ubv ( pColor ); 
			glVertex3f ( pPt[uIdx], -pPt[uIdx+1], -pPt[uIdx+2] ); 
			glNormal3f ( pNl[uIdx], -pNl[uIdx+1], -pNl[uIdx+2] );
		}// for each point
	}//for each plane object
	glEnd();
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
void btl::kinect::CKeyFrame::gpuDetectPlane (const short uPyrLevel_){
	//get next frame
	BTL_ASSERT(btl::utility::BTL_CV == _eConvention, "CKeyFrame data convention must be opencv convention");
	//clear previous plane objs
	_vPlaneObjsDistanceNormal[uPyrLevel_].clear();
	//cluster the top pyramid
	_sNormalHist.gpuClusterNormal(*_acvgmShrPtrPyrNls[uPyrLevel_],*_acvmShrPtrPyrNls[uPyrLevel_],uPyrLevel_,&*_acvmShrPtrNormalClusters[uPyrLevel_],&_vPlaneObjsNormal);
	//enforce position continuity
	_sDistanceHist.clusterDistanceHist(*_acvmShrPtrPyrPts[uPyrLevel_],*_acvmShrPtrPyrNls[uPyrLevel_],uPyrLevel_,_vPlaneObjsNormal,&*_acvmShrPtrDistanceClusters[uPyrLevel_],&_vPlaneObjsDistanceNormal[uPyrLevel_]);
	//merge clusters according to avg normal and position.
	btl::geometry::mergePlaneObj(_vPlaneObjsDistanceNormal[uPyrLevel_]);
	//transform the planes into world coordinates
	for (btl::geometry::tp_plane_obj_list::iterator itPlane = _vPlaneObjsDistanceNormal[uPyrLevel_].begin(); itPlane!= _vPlaneObjsDistanceNormal[uPyrLevel_].end(); itPlane++){
		btl::geometry::transformPlaneIntoWorldCVCV(*itPlane,_eimRw,_eivTw);
		//btl::geometry::transformPlaneIntoLocalCVCV(*itPlane,_eimRw,_eivTw);
	}
	return;
}
void btl::kinect::CKeyFrame::associatePlanes(btl::kinect::CKeyFrame& sReferenceFrame_,const ushort usLevel_){
	if( _vPlaneObjsDistanceNormal[usLevel_].empty()||sReferenceFrame_._vPlaneObjsDistanceNormal[usLevel_].empty() ) return;
	for (btl::geometry::tp_plane_obj_list::iterator itRefPlaneObj = sReferenceFrame_._vPlaneObjsDistanceNormal[usLevel_].begin(); itRefPlaneObj!=sReferenceFrame_._vPlaneObjsDistanceNormal[usLevel_].end(); itRefPlaneObj++ ){
		for (btl::geometry::tp_plane_obj_list::iterator itThisPlaneObj = _vPlaneObjsDistanceNormal[usLevel_].begin(); itThisPlaneObj!=_vPlaneObjsDistanceNormal[usLevel_].end(); itThisPlaneObj++ ){
			if( !itThisPlaneObj->_bCorrespondetFound && itThisPlaneObj->identical( *itRefPlaneObj ) ){
				itThisPlaneObj->_usIdx=itRefPlaneObj->_usIdx;
				itThisPlaneObj->_bCorrespondetFound = itRefPlaneObj->_bCorrespondetFound = true;
				break;
			}
		}//for each plane in refererce frame
	}//for each plane in this frame
}



