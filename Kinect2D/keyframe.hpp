#ifndef KEYFRAME
#define KEYFRAME

#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <btl/Utility/Converters.hpp>
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


using namespace btl; //for "<<" operator
using namespace utility;
//using namespace extra;
//using namespace videosource;
using namespace Eigen;
using namespace cv;

bool sort_pred ( const pair<double, cv::Point2f>& left, const pair<double, cv::Point2f>& right );

bool sort_pred ( const pair<double, cv::Point2f>& left, const pair<double, cv::Point2f>& right )
{
    return left.first < right.first;
}

struct SKeyFrame
{
    cv::Mat _cvmRGB;
    cv::Mat _cvmBW;
    cv::Mat _cvmBWOrigin;
    double* _pDepth;

    cv::Mat _cvmDescriptors;
    cv::flann::Index* _pKDTree; // this is just an index of _cvmDescriptors;

    vector<cv::KeyPoint> _vKeyPoints;
    vector<float>        _vDescriptors;

    vector<int> _vPtPairs; // correspondences odd: input KF even: current KF

    Eigen::Matrix3d _eimR; //R & T is the relative pose w.r.t. the coordinate defined by the previous camera system.
    Eigen::Vector3d _eivT;

    SKeyFrame()
    {
        _cvmRGB.create ( 480, 640, CV_8UC3 );
        _cvmBW .create ( 480, 640, CV_8UC1 );
        _pDepth = new double[921600];
        _pKDTree = NULL;
    }

    ~SKeyFrame()
    {
        delete [] _pDepth;
        delete _pKDTree;
    }

    inline SKeyFrame& operator= ( const SKeyFrame& sKF_ )
    {
        sKF_._cvmRGB.copyTo ( _cvmRGB );
        sKF_._cvmBW .copyTo ( _cvmBW  );
        memcpy ( _pDepth, sKF_._pDepth, 921600 * sizeof ( double ) );
    }

    void save2XML ( const string& strN_ )
    {
        cv::Mat cvmBGR;
        cvtColor ( _cvmRGB, cvmBGR, CV_RGB2BGR );

        cv::imwrite ( "rgb" + strN_ + ".bmp", cvmBGR );
        // create and open a character archive for input
        string strDepth = "depth" + strN_ + ".xml";
        std::ofstream ofs ( strDepth.c_str() );
        boost::archive::xml_oarchive oa ( ofs );
        double* pM = _pDepth;
        vector< double > vDepth;
        vDepth.resize ( 921600 );

        for ( vector< double >::iterator it = vDepth.begin(); it != vDepth.end(); it++ )
        {
            *it = *pM++;
        }

        oa << BOOST_SERIALIZATION_NVP ( vDepth );
    }

    void loadfXML ( const string& strN_ )
    {
        _cvmRGB = cv::imread ( "rgb" + strN_ + ".bmp" );
        // color to grayscale image
        cvtColor ( _cvmRGB, _cvmBWOrigin, CV_RGB2GRAY );
        GaussianBlur ( _cvmBWOrigin, _cvmBW, Size ( 11, 11 ), 5 );
        // create and open a character archive for output
        string strDepth = "depth" + strN_ + ".xml";
        std::ifstream ifs ( strDepth.c_str() );
        boost::archive::xml_iarchive ia ( ifs );
        vector< double > vDepth;
        vDepth.resize ( 921600 );
        ia >> BOOST_SERIALIZATION_NVP ( vDepth );
        double* pM = _pDepth;

        for ( vector< double >::iterator it = vDepth.begin(); it != vDepth.end(); it++ )
        {
            *pM++ = *it;
        }
    }

    void assign ( const cv::Mat& rgb_, const double* pD_ )
    {
        rgb_.copyTo ( _cvmRGB );
        // load depth
        memcpy ( _pDepth, pD_, 921600 * sizeof ( double ) );
        // color to grayscale image
        cvtColor ( _cvmRGB, _cvmBW, CV_RGB2GRAY );
        // clear corners
		clear();
    }

    void detectCorners()
    {
        cv::Mat cvmMask;

        _vKeyPoints.clear();
        _vDescriptors.clear();
        cv::SURF cSurf ( 500, 4, 2, true );
        cSurf ( _cvmBW, cvmMask, _vKeyPoints, _vDescriptors );
        cout << "cSurf() Object:" << _vKeyPoints.size() << endl;
        convert2CVM ( &_cvmDescriptors );

        /*
        		static CvScalar colors = {{0,0,255}};
        		cv::namedWindow ( "myObj", 1 );
        		while ( true )
        	    {
        			for(int i = 0; i < _vKeyPoints.size(); i++ )
        		    {
        		        int radius = cvRound(_vKeyPoints[i].size*1.2/9.*2);
        				cv::circle( _cvmRGB, _vKeyPoints[i].pt, radius, colors, 1, 8, 0 );
        		    }

        		    cv::imshow ( "myObj", _cvmRGB );
        			int nKey = cv::waitKey ( 30 );
        			if ( nKey == 27 )
        			{
        				break;
        			}
        	    }
        */
        return;
    }

    void constructKDTree()
    {
        _pKDTree = new cv::flann::Index ( _cvmDescriptors, cv::flann::KDTreeIndexParams ( 4 ) ); // using 4 randomized kdtrees
        return;
    }

    void detectCorrespondences ( const SKeyFrame& sKF_ )
    {
		CHECK( _pKDTree!=NULL, "SKeyFrame::detectCorrespondences(): KDTree not intialized." );

        _vPtPairs.clear();
        // find nearest neighbors using FLANN
        const int nSizeObject = sKF_._cvmDescriptors.rows;
        cv::Mat cvmIndices ( nSizeObject, 2, CV_32S );
        cv::Mat cvmDists   ( nSizeObject, 2, CV_32F );
        _pKDTree->knnSearch( sKF_._cvmDescriptors, cvmIndices, cvmDists, 2, cv::flann::SearchParams ( 128 ) ); // maximum number of leafs checked
        int* pIndices = cvmIndices.ptr<int> ( 0 );
        float* pDists = cvmDists.ptr<float> ( 0 );
        for ( int i = 0; i < nSizeObject; ++i )
        {
            if ( pDists[2*i] < 0.6 * pDists[2*i+1] )
            {
                _vPtPairs.push_back ( i );
                _vPtPairs.push_back ( pIndices[2*i] );
            }
        }

        return;
    }

    void calcRT ( const SKeyFrame& sCurrentKF_ )
    {
		CHECK( !_vPtPairs.empty(), "SKeyFrame::calcRT() _vPtPairs should not calculated." );
		//PRINT( _vPtPairs.size() );
        //calculate the R and T
        vector< int > _vDepthIdxCur, _vDepthIdx1st, _vSelectedPairs;
        for ( vector< int >::const_iterator cit = _vPtPairs.begin(); cit != _vPtPairs.end(); )
        {
            int nKeyPointIdxCur = *cit++;
            int nKeyPointIdx1st = *cit++;

            int nXCur = int ( sCurrentKF_._vKeyPoints[ nKeyPointIdxCur ].pt.x + .5 );
            int nYCur = int ( sCurrentKF_._vKeyPoints[ nKeyPointIdxCur ].pt.y + .5 );
            int nX1st = int (	  		  _vKeyPoints[ nKeyPointIdx1st ].pt.x + .5 );
            int nY1st = int (			  _vKeyPoints[ nKeyPointIdx1st ].pt.y + .5 );

            int nDepthIdxCur = nYCur * 640 * 3 + nXCur * 3;
            int nDepthIdx1st = nY1st * 640 * 3 + nX1st * 3;

            if ( abs ( sCurrentKF_._pDepth[ nDepthIdxCur + 2 ] ) > 0.0001 && abs ( _pDepth[ nDepthIdx1st + 2 ] ) > 0.0001 )
            {
                _vDepthIdxCur  .push_back ( nDepthIdxCur );
                _vDepthIdx1st  .push_back ( nDepthIdx1st );
                _vSelectedPairs.push_back ( nKeyPointIdxCur );
                _vSelectedPairs.push_back ( nKeyPointIdx1st );
            }
        }
        PRINT ( _vSelectedPairs.size() );

/*
        //for display
        cv::Mat cvmCorr  ( _cvmRGB.rows + sCurrentKF_._cvmRGB.rows, _cvmRGB.cols, CV_8UC3 );
        cv::Mat cvmCorr2 ( _cvmRGB.rows + sCurrentKF_._cvmRGB.rows, _cvmRGB.cols, CV_8UC3 );

        cv::Mat roi1 ( cvmCorr, cv::Rect ( 0, 0, sCurrentKF_._cvmRGB.cols, sCurrentKF_._cvmRGB.rows ) );
        cv::Mat roi2 ( cvmCorr, cv::Rect ( 0, sCurrentKF_._cvmRGB.rows, _cvmRGB.cols, _cvmRGB.rows ) );
        sCurrentKF_._cvmRGB.copyTo ( roi1 );
        _cvmRGB.copyTo ( roi2 );

        static CvScalar colors = {{255, 255, 255}};
        int i = 0;
        int nKey;
        cv::namedWindow ( "myWindow", 1 );

        while ( true )
        {
            cvmCorr.copyTo ( cvmCorr2 );
            cv::line ( cvmCorr2, sCurrentKF_._vKeyPoints[ _vSelectedPairs[i] ].pt, cv::Point ( _vKeyPoints [ _vSelectedPairs[i+1] ].pt.x, _vKeyPoints [ _vSelectedPairs[i+1] ].pt.y + sCurrentKF_._cvmRGB.rows ), colors );
            cv::imshow ( "myWindow", cvmCorr2 );
            nKey = cv::waitKey ( 30 );

            if ( nKey == 32 )
            {
                i += 2;

                if ( i > _vSelectedPairs.size() )
                {
                    break;
                }
            }

            if ( nKey == 27 )
            {
                break;
            }
        }
*/

        int nSize = _vDepthIdxCur.size();
        Eigen::MatrixXd eimCur ( 3, nSize ), eim1st ( 3, nSize );
        vector<  int >::const_iterator cit_Cur = _vDepthIdxCur.begin();
        vector<  int >::const_iterator cit_1st = _vDepthIdx1st.begin();

        for ( int i = 0 ; cit_Cur != _vDepthIdxCur.end(); cit_Cur++, cit_1st++ )
        {
            eimCur ( 0, i ) = sCurrentKF_._pDepth[ *cit_Cur     ];
            eimCur ( 1, i ) = sCurrentKF_._pDepth[ *cit_Cur + 1 ];
            eimCur ( 2, i ) = sCurrentKF_._pDepth[ *cit_Cur + 2 ];

            eim1st ( 0, i ) = 		      _pDepth[ *cit_1st     ];
            eim1st ( 1, i ) = 			  _pDepth[ *cit_1st + 1 ];
            eim1st ( 2, i ) = 		      _pDepth[ *cit_1st + 2 ];
            i++;
        }

        double dS2;
        double dErrorBest = absoluteOrientation < double > ( eimCur , eim1st , false, &_eimR, &_eivT, &dS2 );
        PRINT ( dErrorBest );
        PRINT ( _eimR );
        PRINT ( _eivT );

        for ( int i = 0; i < 2; i++ )
        {
            if ( nSize > 10 )
            {
                double dThreshold = dErrorBest * 0.75;
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

                for ( int n = 0; n < 4000; n++ )
                {
                    select5Rand ( eimCur, eim1st, dice, &eimXTmp, &eimYTmp );
                    dError = absoluteOrientation < double > (  eimXTmp, eimYTmp, false, &eimR, &eivT, &dS );

                    if ( dError > dThreshold )
                    {
                        continue;
                    }

                    //voting
                    int nVotes = voting ( eimCur, eim1st, eimR, eivT, dThreshold, &vVoterIdx );

                    if ( nVotes > eimCur.cols() *.75 )
                    {
                        nMax = nVotes;
                        eimRBest = eimR;
                        eivTBest = eivT;
                        vVoterIdxBest = vVoterIdx;
                        break;
                    }

                    if ( nVotes > nMax );

                    {
                        nMax = nVotes;
                        eimRBest = eimR;
                        eivTBest = eivT;
                        vVoterIdxBest = vVoterIdx;
                    }
                }

                if ( nMax <= 6 )
                {
                    cout << "try increase the threshould" << endl;
                    return ;
                }

                Eigen::MatrixXd eimXInlier ( 3, vVoterIdxBest.size() );
                Eigen::MatrixXd eimYInlier ( 3, vVoterIdxBest.size() );
                selectInlier ( eimCur, eim1st, vVoterIdxBest, &eimXInlier, &eimYInlier );
                dErrorBest = absoluteOrientation < double > ( eimXInlier , eimYInlier , false, &_eimR, &_eivT, &dS2 );

                PRINT ( nMax );
                PRINT ( dErrorBest );
                PRINT ( _eimR );
                PRINT ( _eivT );
                nSize = nMax;

            }//if
        }//for
        return;
    }// calcRT

private:

	void clear() 
	{
    	_vKeyPoints.clear();
    	_vDescriptors.clear();
	    _vPtPairs.clear(); // correspondences odd: input KF even: current KF
	}

    void convert2CVM ( cv::Mat* pcvmDescriptors_ ) const
    {
        int nSizeImage = _vKeyPoints.size();
        int nLengthDescriptorImage = _vDescriptors.size() / nSizeImage;
        pcvmDescriptors_->create ( nSizeImage, nLengthDescriptorImage, CV_32F );
        // copy descriptors
        float* pImage = pcvmDescriptors_->ptr<float> ( 0 );

        for ( vector< float >::const_iterator cit_Descriptor = _vDescriptors.begin(); cit_Descriptor != _vDescriptors.end(); cit_Descriptor++ )
        {
            *pImage++ = *cit_Descriptor;
        }
    }
    void selectInlier ( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, const vector< int >& vVoterIdx_, Eigen::MatrixXd* peimXInlier_, Eigen::MatrixXd* peimYInlier_ )
    {
        CHECK ( vVoterIdx_.size() == peimXInlier_->cols(), " vVoterIdx_.size() must be equal to peimXInlier->cols(). " );
        CHECK ( vVoterIdx_.size() == peimYInlier_->cols(), " vVoterIdx_.size() must be equal to peimYInlier->cols(). " );
        vector< int >::const_iterator cit = vVoterIdx_.begin();

        for ( int i = 0; cit != vVoterIdx_.end(); cit++, i++ )
        {
            ( *peimXInlier_ ).col ( i ) = eimX_.col ( *cit );
            ( *peimYInlier_ ).col ( i ) = eimY_.col ( *cit );
        }

        return;
    }

    int voting ( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, const Eigen::Matrix3d& eimR_, const Eigen::Vector3d& eivV_, const double& dThreshold, vector< int >* pvVoterIdx_ )
    {
        int nV = 0;
        pvVoterIdx_->clear();

        for ( int i = 0; i < eimX_.cols(); i++ )
        {
            Eigen::Vector3d vX = eimX_.col ( i );
            Eigen::Vector3d vY = eimY_.col ( i );
            Eigen::Vector3d vN = vY - eimR_ * vX - eivV_;

            if ( dThreshold > vN.norm() )
            {
                pvVoterIdx_->push_back ( i );
                nV++;
            }
        }

        return nV;
    }// end of function voting

    void select5Rand ( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, boost::variate_generator< boost::mt19937&, boost::uniform_real<> >& dice_, Eigen::MatrixXd* eimXTmp_, Eigen::MatrixXd* eimYTmp_, vector< int >* pvIdx_ = NULL )
    {
        CHECK ( eimX_.rows() == 3, "select5Rnd() eimX_ must have 3 rows" );
        CHECK ( eimY_.rows() == 3, "select5Rnd() eimY_ must have 3 rows" );
        CHECK ( eimX_.cols() == eimY_.cols(), "select5Rnd() eimX_ and eimY_ must contain the same # of cols" );
        CHECK ( eimXTmp_->rows() == 3, "select5Rnd() eimXTmp_ must have 3 rows" );
        CHECK ( eimYTmp_->cols() == 5, "select5Rnd() eimYTmp_ must have 5 cols" );

        if ( eimX_.cols() < 6 )
        {
            return;
        }

        if ( pvIdx_ )
        {
            pvIdx_->clear();
        }

        // generate 5 non-repeat random index
        list< int > lIdx;

        for ( int i = 0; i < eimX_.cols(); i++ )
        {
            lIdx.push_back ( i );
        }

        //PRINT ( lIdx );
        list< int >::iterator it_Idx;
        double dRand;
        int nIdx;

        for ( int i = 0; i < 5; i++ )
        {
            // generate 5 non-repeat random index
            dRand = dice_();
            nIdx = int ( dRand * lIdx.size() - 1 + .5 );
            //locate inside the list
            it_Idx = lIdx.begin();

            while ( nIdx-- > 0 )
            {
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
