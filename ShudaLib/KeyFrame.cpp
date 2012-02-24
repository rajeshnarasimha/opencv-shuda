#define INFO
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>

#include <fstream>
#include <list>

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
#include <gl/freeglut.h>
#include "Camera.h"
#include "Kinect.h"
#include "GLUtil.h"
#include "KeyFrame.h"

#include <math.h>
#include "CVUtil.hpp"
#include "Utility.hpp"

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
		_acvmShrPtrPyrBWs[i]  .reset(new cv::Mat(nRows,nCols,CV_8UC1));
		//device
		_acvgmShrPtrPyrPts[i] .reset(new cv::gpu::GpuMat(nRows,nCols,CV_32FC3));
		_acvgmShrPtrPyrNls[i] .reset(new cv::gpu::GpuMat(nRows,nCols,CV_32FC3));
		_acvgmShrPtrPyrRGBs[i].reset(new cv::gpu::GpuMat(nRows,nCols,CV_8UC3));
		_acvgmShrPtrPyrBWs[i] .reset(new cv::gpu::GpuMat(nRows,nCols,CV_8UC1));
		//plane detection
		_acvmShrPtrNormalClusters[i].reset(new cv::Mat(nRows,nCols,CV_16SC1));
		_acvmShrPtrDistanceClusters[i].reset(new cv::Mat(nRows,nCols,CV_16SC1));
		PRINTSTR("construct pyrmide level:");
		PRINT(i);
	}

	_eConvention = btl::utility::BTL_CV;
	_eimR.setIdentity();
	Eigen::Vector3d eivC (0.,0.,-1.8); //camera location in the world
	_eivT = -_eimR.transpose()*eivC;
	_bIsReferenceFrame = false;
	_bRenderPlane = false;
	_pGL = NULL;
}


void btl::kinect::CKeyFrame::copyTo( CKeyFrame* pKF_, const short sLevel_ ){
	//host
	_acvmShrPtrPyrPts[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrPts[sLevel_]);
	_acvmShrPtrPyrNls[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrNls[sLevel_]);
	_acvmShrPtrPyrRGBs[sLevel_]->copyTo(*pKF_->_acvmShrPtrPyrRGBs[sLevel_]);
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
		PRINTSTR("copy pyrmide level:");
		PRINT(i);
	}
	_bIsReferenceFrame = pKF_->_bIsReferenceFrame;
}

void btl::kinect::CKeyFrame::detectConnectionFromCurrentToReference ( CKeyFrame& sReferenceKF_, const short sLevel_ )  {
	boost::shared_ptr<cv::gpu::SURF_GPU> _pSurf(new cv::gpu::SURF_GPU(500));
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

void btl::kinect::CKeyFrame::calcRT ( const CKeyFrame& sReferenceKF_, const unsigned short sLevel_ ) {
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
                
        int nSize = _vDepthIdxCur.size(); PRINT(nSize);
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
        double dErrorBest = btl::utility::absoluteOrientation < double > ( eimRef, eimCur ,  false, &_eimR, &_eivT, &dS2 );
		PRINT ( dErrorBest );
		PRINT ( _eimR );
		PRINT ( _eivT );
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
        
                for ( int n = 0; n < 1000; n++ ) {
                    select5Rand (  eimRef, eimCur, dice, &eimYTmp, &eimXTmp );
                    dError = btl::utility::absoluteOrientation < double > (  eimYTmp, eimXTmp, false, &eimR, &eivT, &dS );
        
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
                    return ;
                }
        
                Eigen::MatrixXd eimXInlier ( 3, vVoterIdxBest.size() );
                Eigen::MatrixXd eimYInlier ( 3, vVoterIdxBest.size() );
                selectInlier ( eimRef, eimCur, vVoterIdxBest, &eimYInlier, &eimXInlier );
                dErrorBest = btl::utility::absoluteOrientation < double > (  eimYInlier , eimXInlier , false, &_eimR, &_eivT, &dS2 );
        
                PRINT ( nMax );
                PRINT ( dErrorBest );
                PRINT ( _eimR );
                PRINT ( _eivT );
                nSize = nMax;
            }//if
        }//for

    return;
}// calcRT

void btl::kinect::CKeyFrame::renderCameraInGLWorld( bool bRenderCamera_,const unsigned short uLevel_/*=0*/ ) const{
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
	_pRGBCamera->LoadTexture(*_acvmShrPtrPyrRGBs[uLevel_]);
	_pRGBCamera->renderCameraInGLLocal ( *_acvmShrPtrPyrRGBs[uLevel_], .2, bRenderCamera_);
	//render dot clouds
	//if(_bRenderPlane) renderPlanesInGLLocal(uLevel_);
	render3DPtsInGLLocal(uLevel_);//rendering detected plane as well
	glPopMatrix();
}

void btl::kinect::CKeyFrame::render3DPtsInGLLocal(const unsigned short _uLevel) const {
	//////////////////////////////////
	//for rendering the detected plane
	const unsigned char* pColor/* = (const unsigned char*)_pVS->_vcvmPyrRGBs[_uPyrHeight-1]->data*/;
	const short* pLabel;
	if(_bRenderPlane){
		if(NORMAL_CLUSTRE ==_eClusterType){
			pLabel = (const short*)_acvmShrPtrNormalClusters[_pGL->_uLevel]->data;
		}
		else if(DISTANCE_CLUSTER ==_eClusterType){
			pLabel = (const short*)_acvmShrPtrDistanceClusters[_pGL->_uLevel]->data;
		}
	}
	//////////////////////////////////
	float dNx,dNy,dNz;
	float dX, dY, dZ;
	const float* pPt = (const float*) _acvmShrPtrPyrPts[_uLevel]->data;
	const float* pNl = (const float*) _acvmShrPtrPyrNls[_uLevel]->data;
	const unsigned char* pRGB = (const unsigned char*) _acvmShrPtrPyrRGBs[_uLevel]->data;
	glPushMatrix();
	// Generate the data
	if( _pGL && _pGL->_bEnableLighting ){glEnable(GL_LIGHTING);}
	else                            	{glDisable(GL_LIGHTING);}
	for( int i = 0; i < _acvmShrPtrPyrPts[_uLevel]->total(); i++,pRGB+=3,pNl+=3,pPt+=3){
		//////////////////////////////////
		//for rendering the detected plane
		if(_bRenderPlane && pLabel[i]>0){
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
		if( fabs(dNx) + fabs(dNy) + fabs(dNz) > 0.000001 ) {
			if ( _pGL )	{_pGL->renderDisk<float>(dX,dY,dZ,dNx,dNy,dNz,pColor,_pGL->_fSize,_pGL->_bRenderNormal); }
			else { glColor3ubv ( pColor ); glVertex3f ( dX, dY, dZ );}
		}
	}
	glPopMatrix();
	return;
} 

void btl::kinect::CKeyFrame::renderPlanesInGLLocal(const unsigned short _uLevel) const
{
	float dNx,dNy,dNz;
	float dX, dY, dZ;
	const float* pPt = (const float*)_acvmShrPtrPyrPts[_pGL->_uLevel]->data;
	const float* pNl = (const float*)_acvmShrPtrPyrNls[_pGL->_uLevel]->data;
	const unsigned char* pColor/* = (const unsigned char*)_pVS->_vcvmPyrRGBs[_uPyrHeight-1]->data*/;
	const short* pLabel;
	if(NORMAL_CLUSTRE ==_eClusterType){
		//pLabel = (const short*)_pModel->_acvmShrPtrNormalClusters[_pGL->_uLevel]->data;
		pLabel = (const short*)_acvmShrPtrNormalClusters[_pGL->_uLevel]->data;
	}
	else if(DISTANCE_CLUSTER ==_eClusterType){
		//pLabel = (const short*)_pModel->_acvmShrPtrDistanceClusters[_pGL->_uLevel]->data;
		pLabel = (const short*)_acvmShrPtrDistanceClusters[_pGL->_uLevel]->data;
	}
	for( int i = 0; i < _acvmShrPtrPyrPts[_uLevel]->total(); i++,pNl+=3,pPt+=3){
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
			if ( _pGL )	{_pGL->renderDisk<float>(dX,dY,dZ,dNx,dNy,dNz,pColor,_pGL->_fSize,_pGL->_bRenderNormal); }
			else { glColor3ubv ( pColor ); glVertex3f ( dX, dY, dZ );}
		}
	}
	return;
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
	Eigen::MatrixXd* eimXTmp_, Eigen::MatrixXd* eimYTmp_, std::vector< int >* pvIdx_/* = NULL */)
{
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

		if ( pvIdx_ ) //PRINT ( * it_Idx );
		{
			//PRINT( *it_Idx );
			pvIdx_->push_back ( *it_Idx );
		}

		lIdx.erase ( it_Idx );
		//PRINT ( lIdx );
	}

	//PRINT ( eimXTmp_ );
	//PRINT ( eimYTmp_ );

	return;
}//end of function


void btl::kinect::CKeyFrame::detectPlane (const short uPyrLevel_){
	//get next frame
#ifdef TIMER	
	// timer on
	_cT0 =  boost::posix_time::microsec_clock::local_time(); 
#endif
	BTL_ASSERT(btl::utility::BTL_CV == _eConvention, "CKeyFrame data convention must be opencv convention");
	//load pyramids
	_usMinArea = btl::kinect::__aKinectWxH[uPyrLevel_]/60;
	//cluster the top pyramid
	clusterNormal(uPyrLevel_,&*_acvmShrPtrNormalClusters[uPyrLevel_],&_vvLabelPointIdx);
	//enforce position continuity
	clusterDistance(uPyrLevel_,_vvLabelPointIdx,&*_acvmShrPtrDistanceClusters[uPyrLevel_]);
	_bRenderPlane = true;
	_eClusterType = NORMAL_CLUSTRE;
#ifdef TIMER
	// timer off
	_cT1 =  boost::posix_time::microsec_clock::local_time(); 
	_cTDAll = _cT1 - _cT0 ;
	_fFPS = 1000.f/_cTDAll.total_milliseconds();
	PRINT( _fFPS );
#endif
	return;
}

void btl::kinect::CKeyFrame::clusterNormal(const unsigned short& uPyrLevel_,cv::Mat* pcvmLabel_,std::vector< std::vector< unsigned int > >* pvvLabelPointIdx_)
{
	//define constants
	const int nSampleElevation = 4;
	const double dCosThreshold = std::cos(M_PI_4/nSampleElevation);
	const cv::Mat& cvmNls = *_acvmShrPtrPyrNls[uPyrLevel_];
	//make a histogram on the top pyramid
	std::vector< tp_normal_hist_bin > vNormalHist;//idx of sampling the unit half sphere of top pyramid
	//_vvIdx is organized as r(elevation)*c(azimuth) and stores the idx of Normals
	normalHistogram(cvmNls,nSampleElevation,&vNormalHist,btl::utility::BTL_CV);
	
	//re-cluster the normals
	pvvLabelPointIdx_->clear();
	pcvmLabel_->setTo(-1);
	short nLabel =0;
	for(unsigned int uIdxBin = 0; uIdxBin < vNormalHist.size(); uIdxBin++){
		if(vNormalHist[uIdxBin].first.size() < _usMinArea ) continue;
		//get neighborhood of a sampling bin
		std::vector<unsigned int> vNeighourhood; 
		btl::utility::getNeighbourIdxCylinder< unsigned int >(nSampleElevation,nSampleElevation*4,uIdxBin,&vNeighourhood);
		//traverse the neighborhood and cluster the 
		std::vector<unsigned int> vLabelNormalIdx;
		for( std::vector<unsigned int>::const_iterator cit_vNeighbourhood=vNeighourhood.begin();
			cit_vNeighbourhood!=vNeighourhood.end();cit_vNeighbourhood++) {
			btl::utility::normalCluster<double>(cvmNls,vNormalHist[*cit_vNeighbourhood].first,vNormalHist[*cit_vNeighbourhood].second,dCosThreshold,nLabel,pcvmLabel_,&vLabelNormalIdx);
		}
		nLabel++;
		pvvLabelPointIdx_->push_back(vLabelNormalIdx);
		//compute average normal
		/*Eigen::Vector3d eivAvgNl;
		btl::utility::avgNormals<double>(cvmNls,vLabelNormalIdx,&eivAvgNl);
		_vLabelAvgNormals.push_back(eivAvgNl);*/
	}
	return;
}
void btl::kinect::CKeyFrame::normalHistogram( const cv::Mat& cvmNls_, int nSamples_, std::vector< tp_normal_hist_bin >* pvNormalHistogram_,btl::utility::tp_coordinate_convention eCon_)
{
	//clear and re-initialize pvvIdx_
	int nSampleAzimuth = nSamples_<<2; //nSamples*4
	pvNormalHistogram_->clear();
	pvNormalHistogram_->resize(nSamples_*nSampleAzimuth,tp_normal_hist_bin(std::vector<unsigned int>(),Eigen::Vector3d(0,0,0)));
	const double dS = M_PI_2/nSamples_;//sampling step
	int r,c,rc;
	const float* pNl = (const float*) cvmNls_.data;
	for(unsigned int i =0; i< cvmNls_.total(); i++, pNl+=3)	{
		if( pNl[2]>0 || fabs(pNl[0])+fabs(pNl[1])+fabs(pNl[2])<0.0001 ) {continue;}
		btl::utility::normalVotes<float>(pNl,dS,&r,&c,eCon_);
		rc = r*nSampleAzimuth+c;
		if(rc<0||rc>pvNormalHistogram_->size()){continue;}
		(*pvNormalHistogram_)[rc].first.push_back(i);
		(*pvNormalHistogram_)[rc].second += Eigen::Vector3d(pNl[0],pNl[1],pNl[2]);
	}
	//average the 
	for(std::vector<tp_normal_hist_bin>::iterator it_vNormalHist = pvNormalHistogram_->begin();
		it_vNormalHist!=pvNormalHistogram_->end(); it_vNormalHist++) {
		if(it_vNormalHist->first.size()>0) {
			it_vNormalHist->second.normalize();
		}
	}

	return;
}
void btl::kinect::CKeyFrame::distanceHistogram( const cv::Mat& cvmNls_, const cv::Mat& cvmPts_, const unsigned int& nSamples, 
	const std::vector< unsigned int >& vIdx_, tp_hist* pvDistHist )
{
	const double dLow  = -3;
	const double dHigh =  3;
	const double dSampleStep = ( dHigh - dLow )/nSamples; 

	pvDistHist->clear();
	pvDistHist->resize(nSamples,tp_pair_hist_bin(std::vector<tp_pair_hist_element>(), 0.) );
	const float*const pPt = (float*) cvmPts_.data;
	const float*const pNl = (float*) cvmNls_.data;
	//collect the distance histogram
	for(std::vector< unsigned int >::const_iterator cit_vPointIdx = vIdx_.begin();
		cit_vPointIdx!=vIdx_.end(); cit_vPointIdx++)
	{
		unsigned int uOffset = (*cit_vPointIdx)*3;
		double dDist = pPt[uOffset]*pNl[uOffset] + pPt[uOffset+1]*pNl[uOffset+1] + pPt[uOffset+2]*pNl[uOffset+2];

		int nBin = floor( (dDist -dLow)/ dSampleStep );
		if( nBin >= 0 && nBin <nSamples)
		{
			(*pvDistHist)[nBin].first.push_back(tp_pair_hist_element(dDist,*cit_vPointIdx));
			(*pvDistHist)[nBin].second += dDist;
		}
	}

	//calc the avg distance for each bin 
	//construct a list for sorting
	for(std::vector< tp_pair_hist_bin >::iterator cit_vDistHist = pvDistHist->begin();
		cit_vDistHist != pvDistHist->end(); cit_vDistHist++ )
	{
		unsigned int uBinSize = cit_vDistHist->first.size();
		if( uBinSize==0 ) continue;

		//calculate avg distance
		cit_vDistHist->second /= uBinSize;
	}
	return;
}
void btl::kinect::CKeyFrame::clusterDistance( const unsigned short uPyrLevel_, const std::vector< std::vector<unsigned int> >& vvNormalClusterPtIdx_, cv::Mat* cvmDistanceClusters_ )
{
	cvmDistanceClusters_->setTo(-1);
	//construct the label mat
	const cv::Mat& cvmPts = *_acvmShrPtrPyrPts[uPyrLevel_];
	const cv::Mat& cvmNls = *_acvmShrPtrPyrNls[uPyrLevel_];
	const double dLow  = -3;
	const double dHigh =  3;
	const int nSamples = 400;
	const double dSampleStep = ( dHigh - dLow )/nSamples; 
	const double dMergeStep = dSampleStep;

	tp_hist	vDistHist; //histogram of distancte vector< vDist, cit_vIdx > 
	short sLabel = 0;
	for(std::vector< std::vector< unsigned int > >::const_iterator cit_vvLabelPointIdx = vvNormalClusterPtIdx_.begin();
		cit_vvLabelPointIdx!=vvNormalClusterPtIdx_.end(); cit_vvLabelPointIdx++){
			//collect 
			distanceHistogram( cvmNls, cvmPts, nSamples, *cit_vvLabelPointIdx, &vDistHist );
			std::vector< tp_flag > vMergeFlags(nSamples, btl::kinect::CKeyFrame::EMPTY); //==0 no merging, ==1 merge with left, ==2 merge with right, ==3 merging with both
			calcMergeFlag( vDistHist, dMergeStep, &vMergeFlags ); // EMPTY/NO_MERGE/MERGE_WITH_LEFT/MERGE_WITH_BOTH/MERGE_WITH_RIGHT 
			//cluster
			mergeDistanceBins( vMergeFlags, vDistHist, *cit_vvLabelPointIdx, &sLabel, &*_acvmShrPtrDistanceClusters[uPyrLevel_] );
			sLabel++;
	}//for each normal label
}
void btl::kinect::CKeyFrame::calcMergeFlag( const tp_hist& vDistHist, const double& dMergeDistance, std::vector< tp_flag >* pvMergeFlags_ ){
	//merge the bins whose distance is similar
	std::vector< tp_flag >::iterator it_vMergeFlags = pvMergeFlags_->begin()+1; 
	std::vector< tp_flag >::iterator it_prev;
	std::vector< tp_pair_hist_bin >::const_iterator cit_prev;
	std::vector< tp_pair_hist_bin >::const_iterator cit_endm1 = vDistHist.end() - 1;

	for(std::vector< tp_pair_hist_bin >::const_iterator cit_vDistHist = vDistHist.begin() + 1;
		cit_vDistHist != cit_endm1; cit_vDistHist++,it_vMergeFlags++ ) {
			unsigned int uBinSize = cit_vDistHist->first.size();
			if(0==uBinSize) continue;
			*it_vMergeFlags = btl::kinect::CKeyFrame::NO_MERGE;
			cit_prev = cit_vDistHist -1;
			it_prev  = it_vMergeFlags-1;
			if( btl::kinect::CKeyFrame::EMPTY == *it_prev ) continue;

			if( fabs(cit_prev->second - cit_vDistHist->second) < dMergeDistance ){ //avg distance smaller than the sample step.
				//previou bin
				if     (btl::kinect::CKeyFrame::NO_MERGE       ==*it_prev){	*it_prev = btl::kinect::CKeyFrame::MERGE_WITH_RIGHT;}
				else if(btl::kinect::CKeyFrame::MERGE_WITH_LEFT==*it_prev){ *it_prev = btl::kinect::CKeyFrame::MERGE_WITH_BOTH; }
				//current bin
				*it_vMergeFlags = btl::kinect::CKeyFrame::MERGE_WITH_LEFT;
			}//if mergable
	}//for each bin
}
void btl::kinect::CKeyFrame::mergeDistanceBins( const std::vector< tp_flag >& vMergeFlags_, const tp_hist& vDistHist_, const std::vector< unsigned int >& vLabelPointIdx_, short* pLabel_, cv::Mat* pcvmLabel_ ){
	std::vector< tp_flag >::const_iterator cit_vMergeFlags = vMergeFlags_.begin();
	std::vector< tp_pair_hist_bin >::const_iterator cit_endm1 = vDistHist_.end() - 1;
	short* pDistanceLabel = (short*) pcvmLabel_->data;
	for(std::vector< tp_pair_hist_bin >::const_iterator cit_vDistHist = vDistHist_.begin() + 1;
		cit_vDistHist != cit_endm1; cit_vDistHist++,cit_vMergeFlags++ )	{
			if(btl::kinect::CKeyFrame::EMPTY==*cit_vMergeFlags) continue;
			if(btl::kinect::CKeyFrame::NO_MERGE==*cit_vMergeFlags||btl::kinect::CKeyFrame::MERGE_WITH_RIGHT==*cit_vMergeFlags||
				btl::kinect::CKeyFrame::MERGE_WITH_BOTH==*cit_vMergeFlags||btl::kinect::CKeyFrame::MERGE_WITH_LEFT==*cit_vMergeFlags){
					if(cit_vDistHist->first.size()>_usMinArea){
						for( std::vector<tp_pair_hist_element>::const_iterator cit_vPair = cit_vDistHist->first.begin();
							cit_vPair != cit_vDistHist->first.end(); cit_vPair++ ){
								pDistanceLabel[cit_vPair->second] = *pLabel_;
						}//for 
					}//if
			}
			if(btl::kinect::CKeyFrame::NO_MERGE==*cit_vMergeFlags||btl::kinect::CKeyFrame::MERGE_WITH_LEFT==*cit_vMergeFlags){
				(*pLabel_)++;
			}
	}//for
}
