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
    vector< cv::Point2f > _vCorners;
    Eigen::Matrix3d _eimR; //R & T is the relative pose w.r.t. the coordinate defined by the previous camera system.
    Eigen::Vector3d _eivT;
    vector<bool> _vEffective; // this hold the vectors whose depth is effective

    SKeyFrame()
    {
        _cvmRGB.create ( 480, 640, CV_8UC3 );
        _cvmBW .create ( 480, 640, CV_8UC1 );
        _pDepth = new double[921600];
    }

    ~SKeyFrame()
    {
        delete [] _pDepth;
    }

    inline SKeyFrame& operator= ( const SKeyFrame& sKF_ )
    {
        sKF_._cvmRGB.copyTo ( _cvmRGB );
        sKF_._cvmBW .copyTo ( _cvmBW  );
        memcpy ( _pDepth, sKF_._pDepth, 921600 * sizeof ( double ) );
        _vCorners = sKF_._vCorners;
    }

	void save2XML(const string& strN_)
	{
		cv::imwrite( "rgb"+ strN_ + ".bmp", _cvmRGB );
		// create and open a character archive for input
		string strDepth = "depth"+ strN_ + ".xml";
	    std::ofstream ofs ( strDepth.c_str() );
    	boost::archive::xml_oarchive oa ( ofs );
		double *pM = _pDepth;
		vector< double > vDepth; vDepth.resize( 921600 );
		for( vector< double >::iterator it = vDepth.begin(); it != vDepth.end(); it++ )
		{
			*it = *pM++; 
		}
		oa << BOOST_SERIALIZATION_NVP ( vDepth );
	}

	void loadfXML(const string& strN_)
	{
		_cvmRGB = cv::imread("rgb"+ strN_ + ".bmp");
		// color to grayscale image
        cvtColor ( _cvmRGB, _cvmBWOrigin, CV_RGB2GRAY );
		GaussianBlur( _cvmBWOrigin, _cvmBW, Size( 11, 11), 5 );
		// create and open a character archive for output
	    string strDepth = "depth"+ strN_ + ".xml";
	    std::ifstream ifs ( strDepth.c_str() );
    	boost::archive::xml_iarchive ia ( ifs );
		vector< double > vDepth; vDepth.resize( 921600 );
		ia >> BOOST_SERIALIZATION_NVP ( vDepth );
		double* pM = _pDepth;
		for( vector< double >::iterator it = vDepth.begin(); it != vDepth.end(); it++ )
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
        cvtColor ( _cvmRGB, _cvmBWOrigin, CV_RGB2GRAY );
		GaussianBlur( _cvmBWOrigin, _cvmBW, Size( 7, 7), 3 );
        // clear corners
        _vCorners.clear();
    }

    void detectCorners()
    {
        vector< cv::KeyPoint > vKs2;
        cv::FAST ( _cvmBW, vKs2, 10, true );
        //FileStorage fs("KeyPoint", FileStorage::WRITE);
        //write(fs, "KeyPoint", vKs);
        vector< cv::Point2f > vPtAll;
        KeyPoint::convert ( vKs2, vPtAll );
        // compute the Shi-Tomasi score for all FAST corners
        vector<pair<double, cv::Point2f> > vCornersAndSTScores;

        for ( vector< cv::Point2f >::const_iterator cit = vPtAll.begin(); cit != vPtAll.end(); cit ++ )
        {
            if ( cit->x < 20 || cit->x > 620 || cit->y < 20 || cit->y > 460 ) //remove border points
            {
                continue;
            }

            if ( abs ( _pDepth[ ( int ( cit->y ) * 640 + int ( cit->x ) ) * 3 + 2] ) < 0.0001 ) //remove the corner whose depth is empty
            {
                continue;
            }

            double dST = FindShiTomasiScoreAtPoint< double > ( _cvmBW, 5 , cit->x, cit->y ); // threshould choose the same as used in PTAM
            vCornersAndSTScores.push_back ( pair< double, cv::Point2f > ( -1.0 * dST, *cit ) );
        }

        // Sort according to Shi-Tomasi score
        int nToAdd = 1000;
        _vCorners.clear();
        std::sort ( vCornersAndSTScores.begin(), vCornersAndSTScores.end(), sort_pred );
		//PRINT ( vCornersAndSTScores.size() );
        for ( unsigned int i = 0; i < vCornersAndSTScores.size() && i < nToAdd; i++ )
        {
            _vCorners.push_back (  vCornersAndSTScores[i].second );
        }

        return;
    }

    bool detectOpticFlowAndRT ( const SKeyFrame& sPrevKF_ )
    {

        if ( sPrevKF_._vCorners.empty() )
        {
            return false;
        }

        // get optical flow lines
        vector<unsigned char> vStatus;
        vector<float> vErr;
        calcOpticalFlowPyrLK ( sPrevKF_._cvmBW, _cvmBW, sPrevKF_._vCorners, _vCorners, vStatus, vErr, cv::Size( 15, 15 ), 5, cv::TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 10, .05 ) );

        int nSize = 0;        _vEffective.clear();
        for ( vector< Point2f >::const_iterator cit_Curr = _vCorners.begin(); cit_Curr != _vCorners.end(); cit_Curr++ )
        {
            int nX = int ( cit_Curr->x + .5 );
            int nY = int ( cit_Curr->y + .5 );

            if ( abs ( _pDepth[ nY*640*3+nX*3+2 ] ) < 0.0001 )
            {
                _vEffective.push_back ( false );
            }
            else
            {
                _vEffective.push_back ( true );
                nSize++;
            }
        }
		PRINT ( _vCorners.size() );	
        PRINT ( nSize );
        Eigen::MatrixXd eimX ( 3, nSize ), eimY ( 3, nSize );
        vector< Point2f >::const_iterator cit_Curr = _vCorners.begin();
        vector< Point2f >::const_iterator cit_Prev = sPrevKF_._vCorners.begin();
        vector< bool >::const_iterator it3 = _vEffective.begin();
        int nXCurr, nYCurr, nIdxCurr, nXPrev, nYPrev, nIdxPrev;
		int nC = 0;
        for ( int i = 0; cit_Curr != _vCorners.end(); cit_Curr++, cit_Prev++, it3++ )
        {
//			PRINT( nC++ );
            if ( *it3 )// if the current point is effective containing both the depth and 2D corners
            {
                nXPrev   = int ( cit_Prev->x + .5 );
                nYPrev 	 = int ( cit_Prev->y + .5 );
                nIdxPrev = nYPrev * 640 * 3 + nXPrev * 3;
                eimX ( 0, i ) = sPrevKF_._pDepth[ nIdxPrev   ];
                eimX ( 1, i ) = sPrevKF_._pDepth[ nIdxPrev+1 ];
                eimX ( 2, i ) = sPrevKF_._pDepth[ nIdxPrev+2 ];
                nXCurr   = int ( cit_Curr->x + .5 );
                nYCurr   = int ( cit_Curr->y + .5 );
                nIdxCurr = nYCurr * 640 * 3 + nXCurr * 3;
                eimY ( 0, i ) = _pDepth[ nIdxCurr   ];
                eimY ( 1, i ) = _pDepth[ nIdxCurr+1 ];
                eimY ( 2, i ) = _pDepth[ nIdxCurr+2 ];
//				PRINT( eimX.col(i) );
//				PRINT( eimY.col(i) );
//				PRINT( i );
                i++;
            }
        }
		//Eigen::MatrixXd eimN = eimX - eimY;
		//PRINT( eimN );
//RANSAC
        if ( nSize > 10 )
        {
			double dThreshold = 0.01;
            Eigen::MatrixXd eimXTmp ( 3, 5 ), eimYTmp ( 3, 5 );
            // random generator
            boost::mt19937 rng;
            boost::uniform_real<> gen ( 0, 1 );
            boost::variate_generator< boost::mt19937&, boost::uniform_real<> > dice ( rng, gen );
			double dError; Eigen::Matrix3d eimR; Eigen::Vector3d eivT; double dS; vector< int > vVoterIdx;
			Eigen::Matrix3d eimRBest; Eigen::Vector3d eivTBest; vector< int > vVoterIdxBest;
			int nMax = 0;	vector < int > vRndIdx;
			for( int n = 0; n<2000; n++ )
			{
//				PRINT( n );
	 			select5Rand( eimX, eimY, dice, &eimXTmp, &eimYTmp );
				dError = absoluteOrientation < double > (  eimXTmp, eimYTmp, false, &eimR, &eivT, &dS );
//				PRINT( dError );
//				PRINT( eimR );
//				PRINT( eivT );
				if( dError > dThreshold )
					continue;
				//voting
				int nVotes = voting( eimX, eimY, eimR, eivT, dThreshold, &vVoterIdx );
//				PRINT( nVotes );
				if( nVotes > eimX.cols()/3 )
				{
					nMax = nVotes;
					eimRBest = eimR;
					eivTBest = eivT;
					vVoterIdxBest = vVoterIdx;
					break;
				}

				if( nVotes > nMax );
				{
					nMax = nVotes;
					eimRBest = eimR;
					eivTBest = eivT;
					vVoterIdxBest = vVoterIdx;
				}
			}
			PRINT( nMax );

			if( nMax <= 10 )
			{
				_eimR.setIdentity();
				_eivT.setZero();
				cout << "try increase the threshould"<< endl;
				return false;
			}
			Eigen::MatrixXd eimXInlier( 3, vVoterIdxBest.size() );
			Eigen::MatrixXd eimYInlier( 3, vVoterIdxBest.size() );
			selectInlier( eimX, eimY, vVoterIdxBest, &eimXInlier, &eimYInlier );
//			Eigen::MatrixXd mX(eimX.leftCols<100>());
//			Eigen::MatrixXd mY( eimY.leftCols<100>());
			double dS2;
			double dErrorBest = absoluteOrientation < double > ( eimXInlier , eimYInlier , false, &_eimR, &_eivT, &dS2 );
	        PRINT ( dErrorBest );
    	    PRINT ( _eimR );
        	PRINT ( _eivT );
        }// end if( nSize > 6 );
//		else
//		{
//			_eimR.setIdentity();
//			_eivT.setZero();
//		}
		return true;
    }//end function detectOpticFlowAndRT()

private:
	void selectInlier( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, const vector< int >& vVoterIdx_, Eigen::MatrixXd* peimXInlier_, Eigen::MatrixXd* peimYInlier_ )
	{
		CHECK( vVoterIdx_.size() == peimXInlier_->cols(), " vVoterIdx_.size() must be equal to peimXInlier->cols(). " );
		CHECK( vVoterIdx_.size() == peimYInlier_->cols(), " vVoterIdx_.size() must be equal to peimYInlier->cols(). " );
		vector< int >::const_iterator cit = vVoterIdx_.begin();
		for( int i = 0; cit != vVoterIdx_.end(); cit++, i++ )
		{
			(*peimXInlier_).col( i ) = eimX_.col( *cit );
			(*peimYInlier_).col( i ) = eimY_.col( *cit );
		}
		return;
	}

	int voting( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, const Eigen::Matrix3d& eimR_, const Eigen::Vector3d& eivV_, const double& dThreshold, vector< int >* pvVoterIdx_ )
	{
		int nV = 0;
		pvVoterIdx_->clear(); 
		for( int i = 0; i < eimX_.cols(); i++ )
		{
			Eigen::Vector3d vX = eimX_.col( i );
			Eigen::Vector3d vY = eimY_.col( i );
			Eigen::Vector3d vN = vY - eimR_ * vX - eivV_;
			if( dThreshold > vN.norm())
			{
				pvVoterIdx_->push_back( i );
				nV++;
			}
		}
		return nV;
	}// end of function voting

	void select5Rand( const Eigen::MatrixXd& eimX_, const Eigen::MatrixXd& eimY_, boost::variate_generator< boost::mt19937&, boost::uniform_real<> >& dice_, Eigen::MatrixXd* eimXTmp_, Eigen::MatrixXd* eimYTmp_, vector< int >* pvIdx_ =NULL)
	{
		CHECK( eimX_.rows()== 3, "select5Rnd() eimX_ must have 3 rows" );
		CHECK( eimY_.rows()== 3, "select5Rnd() eimY_ must have 3 rows" );
		CHECK( eimX_.cols()== eimY_.cols(), "select5Rnd() eimX_ and eimY_ must contain the same # of cols" );
		CHECK( eimXTmp_->rows()==3, "select5Rnd() eimXTmp_ must have 3 rows" );
		CHECK( eimYTmp_->cols()==5, "select5Rnd() eimYTmp_ must have 5 cols" );
		if( eimX_.cols() < 6 )
			return;
		if( pvIdx_ )
			pvIdx_->clear();
        // generate 5 non-repeat random index
        list< int > lIdx;
        for ( int i = 0; i < eimX_.cols(); i++ )
        {
            lIdx.push_back ( i );
        }
        //PRINT ( lIdx );
        list< int >::iterator it_Idx;
		double dRand; int nIdx; 
        for ( int i = 0; i < 5; i++ )
        {
            // generate 5 non-repeat random index
	        dRand = dice_();
            nIdx = int ( dRand * lIdx.size()-1 + .5 );
            //locate inside the list
            it_Idx = lIdx.begin();
            while ( nIdx-- > 0 )
            {
                it_Idx++;
            }

            (*eimXTmp_) ( 0, i ) = eimX_ ( 0, * it_Idx );
            (*eimXTmp_) ( 1, i ) = eimX_ ( 1, * it_Idx );
            (*eimXTmp_) ( 2, i ) = eimX_ ( 2, * it_Idx );

            (*eimYTmp_) ( 0, i ) = eimY_ ( 0, * it_Idx );
            (*eimYTmp_) ( 1, i ) = eimY_ ( 1, * it_Idx );
            (*eimYTmp_) ( 2, i ) = eimY_ ( 2, * it_Idx );

            if(pvIdx_)//PRINT ( * it_Idx );
			{
				//PRINT( *it_Idx );
				pvIdx_->push_back( *it_Idx );
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
