#ifndef KEYFRAME
#define KEYFRAME

#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <Converters.hpp>
#include <VideoSourceKinect.hpp>
#include <fstream>
#include <boost/serialization/vector.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <list>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

template<typename T>
struct SKeyFrame {
	typedef boost::shared_ptr<SKeyFrame<T>> tp_shared_ptr;
	boost::shared_ptr<btl::extra::videosource::CKinectView> _pView;
    cv::Mat _cvmRGB;
    cv::Mat _cvmBW;
    T* _pDepth;

    std::vector<cv::KeyPoint> _vKeyPoints;
	std::vector<cv::DMatch> _vMatches;

	cv::gpu::GpuMat _cvgmKeyPoints;
	cv::gpu::GpuMat _cvgmDescriptors;
	cv::gpu::GpuMat _cvgmBW;

	Eigen::Matrix3d _eimR; //R & T is the relative pose w.r.t. the coordinate defined by the previous camera system.
    Eigen::Vector3d _eivT;

	bool _bIsReferenceFrame;

    SKeyFrame( btl::extra::videosource::CCalibrateKinect& cCK_){
		_pView.reset(new btl::extra::videosource::CKinectView(cCK_));
        _cvmRGB.create ( 480, 640, CV_8UC3 );
        _cvmBW .create ( 480, 640, CV_8UC1 );
        _pDepth = new T[921600];
        _eimR.setIdentity();
        Eigen::Vector3d eivC (0.,0.,-1.5);
		_eivT = -_eimR.transpose()*eivC;
		_bIsReferenceFrame = false;
    }

    ~SKeyFrame() {
        delete [] _pDepth;
    }
    void assign ( const cv::Mat& rgb_, const T* pD_ ) {
        rgb_.copyTo ( _cvmRGB );
		_pView->LoadTexture(_cvmRGB);
        // load depth
        memcpy ( _pDepth, pD_, 921600 * sizeof ( T ) );
        // color to grayscale image
        cvtColor ( _cvmRGB, _cvmBW, CV_RGB2GRAY );
		_cvgmBW.upload(_cvmBW);
        // clear corners
		_bIsReferenceFrame = false;
    }
	//detect surf features in the current frame
    void detectCorners() {
		// detecting keypoints & computing descriptors
		boost::shared_ptr<cv::gpu::SURF_GPU> _pSurf(new cv::gpu::SURF_GPU(500));
		(*_pSurf)(_cvgmBW, cv::gpu::GpuMat(), _cvgmKeyPoints, _cvgmDescriptors);
        _pSurf->downloadKeypoints(_cvgmKeyPoints, _vKeyPoints);
        return;
    }
	//detect matches between current frame and reference frame
    void detectCorrespondences ( const SKeyFrame& sReferenceKF_ )  {
		cv::gpu::BruteForceMatcher_GPU< L2<float> > cBruteMatcher;
		cv::gpu::GpuMat cvgmTrainIdx, cvgmDistance;
		cBruteMatcher.matchSingle( this->_cvgmDescriptors,  sReferenceKF_._cvgmDescriptors, cvgmTrainIdx, cvgmDistance);
		cv::gpu::BruteForceMatcher_GPU< L2<float> >::matchDownload(cvgmTrainIdx, cvgmDistance, _vMatches);
		std::sort( _vMatches.begin(), _vMatches.end() );
		if (_vMatches.size()> 200) { _vMatches.erase( _vMatches.begin()+200, _vMatches.end() ); }
		return;
	}
	//
    void calcRT ( const SKeyFrame& sReferenceKF_ ) {
        CHECK ( !_vMatches.empty(), "SKeyFrame::calcRT() _vMatches should not calculated." );
        //calculate the R and T
        vector< int > _vDepthIdxCur, _vDepthIdx1st, _vSelectedPairs;

        for ( std::vector< cv::DMatch >::const_iterator cit = _vMatches.begin(); cit != _vMatches.end(); cit++ ) {
            int nKeyPointIdxCur = cit->queryIdx;
            int nKeyPointIdx1st = cit->trainIdx;

            int nXCur = cvRound ( 			    _vKeyPoints[ nKeyPointIdxCur ].pt.x );
            int nYCur = cvRound ( 			    _vKeyPoints[ nKeyPointIdxCur ].pt.y );
            int nX1st = cvRound ( sReferenceKF_._vKeyPoints[ nKeyPointIdx1st ].pt.x );
            int nY1st = cvRound ( sReferenceKF_._vKeyPoints[ nKeyPointIdx1st ].pt.y );

            int nDepthIdxCur = nYCur * 640 * 3 + nXCur * 3;
            int nDepthIdx1st = nY1st * 640 * 3 + nX1st * 3;

            if ( fabs ( _pDepth[ nDepthIdxCur + 2 ] ) > 0.0001 && fabs ( sReferenceKF_._pDepth[ nDepthIdx1st + 2 ] ) > 0.0001 ) {
                _vDepthIdxCur  .push_back ( nDepthIdxCur );
                _vDepthIdx1st  .push_back ( nDepthIdx1st );
				_vSelectedPairs.push_back ( nKeyPointIdxCur );
				_vSelectedPairs.push_back ( nKeyPointIdx1st );
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
        Eigen::MatrixXd eimCur ( 3, nSize ), eim1st ( 3, nSize );
        vector<  int >::const_iterator cit_Cur = _vDepthIdxCur.begin();
        vector<  int >::const_iterator cit_1st = _vDepthIdx1st.begin();

        for ( int i = 0 ; cit_Cur != _vDepthIdxCur.end(); cit_Cur++, cit_1st++ ){
            eimCur ( 0, i ) = 			    _pDepth[ *cit_Cur     ];
            eimCur ( 1, i ) =  			    _pDepth[ *cit_Cur + 1 ];
            eimCur ( 2, i ) = 			    _pDepth[ *cit_Cur + 2 ];
            eim1st ( 0, i ) = sReferenceKF_._pDepth[ *cit_1st     ];
            eim1st ( 1, i ) = sReferenceKF_._pDepth[ *cit_1st + 1 ];
            eim1st ( 2, i ) = sReferenceKF_._pDepth[ *cit_1st + 2 ];
            i++;
        }
        double dS2;
        double dErrorBest = btl::utility::absoluteOrientation < double > ( eim1st, eimCur ,  false, &_eimR, &_eivT, &dS2 );
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
                vector< int > vVoterIdx;
                Eigen::Matrix3d eimRBest;
                Eigen::Vector3d eivTBest;
                vector< int > vVoterIdxBest;
                int nMax = 0;
                vector < int > vRndIdx;
                Eigen::MatrixXd eimXTmp ( 3, 5 ), eimYTmp ( 3, 5 );
        
                for ( int n = 0; n < 1000; n++ ) {
                    select5Rand (  eim1st, eimCur, dice, &eimYTmp, &eimXTmp );
                    dError = btl::utility::absoluteOrientation < double > (  eimYTmp, eimXTmp, false, &eimR, &eivT, &dS );
        
                    if ( dError > dThreshold ) {
                        continue;
                    }
        
                    //voting
                    int nVotes = voting ( eim1st, eimCur, eimR, eivT, dThreshold, &vVoterIdx );
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
                selectInlier ( eim1st, eimCur, vVoterIdxBest, &eimYInlier, &eimXInlier );
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

	void applyRelativePose( const SKeyFrame& sReferenceKF_ ) {
		_eimR = _eimR*sReferenceKF_._eimR;
		_eivT = _eimR*sReferenceKF_._eivT + _eivT;
	}

	void renderCamera( bool bRenderCamera_ ) const{
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
	    _pView->renderCamera ( btl::extra::videosource::CCalibrateKinect::RGB_CAMERA, _cvmRGB, btl::extra::videosource::CKinectView::ALL_CAMERA, .2, bRenderCamera_);
		//render dot clouds
		renderDepth();
	    glPopMatrix();
	}

	void setView(Eigen::Matrix4d* pModelViewGL) const {
    	*pModelViewGL = btl::utility::setModelViewGLfromRTCV ( _eimR, _eivT );
		return;
	}

    void renderDepth() const {
		const unsigned char* pColor = _cvmRGB.data;
		const T* pDepth = _pDepth;
		glPushMatrix();
		glPointSize ( 1. );
		glBegin ( GL_POINTS );
		for ( int i = 0; i < 307200; i++ ) {
			T dX = *pDepth++;
			T dY = *pDepth++;
			T dZ = *pDepth++;

			if ( fabs ( dZ ) > 0.001 ) {
				glColor3ubv ( pColor );
				glVertex3f ( dX, -dY, -dZ );
			}

			pColor += 3;
		}
		glEnd();
		glPopMatrix();
    }

private:
    void selectInlier ( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, const std::vector< int >& vVoterIdx_, Eigen::MatrixXd* peimXInlier_, Eigen::MatrixXd* peimYInlier_ ) {
        CHECK ( vVoterIdx_.size() == peimXInlier_->cols(), " vVoterIdx_.size() must be equal to peimXInlier->cols(). " );
        CHECK ( vVoterIdx_.size() == peimYInlier_->cols(), " vVoterIdx_.size() must be equal to peimYInlier->cols(). " );
        std::vector< int >::const_iterator cit = vVoterIdx_.begin();

        for ( int i = 0; cit != vVoterIdx_.end(); cit++, i++ ) {
            ( *peimXInlier_ ).col ( i ) = eimX_.col ( *cit );
            ( *peimYInlier_ ).col ( i ) = eimY_.col ( *cit );
        }
        return;
    }

    int voting ( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, const Eigen::Matrix3d& eimR_, const Eigen::Vector3d& eivV_, const double& dThreshold, std::vector< int >* pvVoterIdx_ ) {
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

    void select5Rand ( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, boost::variate_generator< boost::mt19937&, boost::uniform_real<> >& dice_, 
						Eigen::MatrixXd* eimXTmp_, Eigen::MatrixXd* eimYTmp_, std::vector< int >* pvIdx_ = NULL )
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
};//end of struct

#endif
