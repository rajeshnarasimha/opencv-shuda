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

void findPairs( const vector< cv::KeyPoint >& vObjectKeyPoints_, const vector< float >& vObjectDescriptors_ , const vector< cv::KeyPoint >& vImageKeyPoints_, const vector< float >& vImageDescriptors_, vector< int >* pvPtPairs_ )
{
	int nSizeObject = vObjectKeyPoints_.size();
	int nLengthDescriptorObject = vObjectDescriptors_.size() / nSizeObject;
	cv::Mat cvmObject( nSizeObject, nLengthDescriptorObject, CV_32F );
	// copy descriptors
	float* pObject = cvmObject.ptr<float>(0);
    for(vector< float >::const_iterator cit_ObjectDescriptor = vObjectDescriptors_.begin(); cit_ObjectDescriptor!=vObjectDescriptors_.end(); cit_ObjectDescriptor++)
    {
		*pObject++ = *cit_ObjectDescriptor;
    }

	int nSizeImage = vImageKeyPoints_.size();
	int nLengthDescriptorImage = vImageDescriptors_.size() / nSizeImage;
	cv::Mat cvmImage( nSizeImage, nLengthDescriptorImage, CV_32F );
	// copy descriptors
	float* pImage = cvmImage.ptr<float>(0);
    for(vector< float >::const_iterator cit_ImageDescriptor = vImageDescriptors_.begin(); cit_ImageDescriptor!=vImageDescriptors_.end(); cit_ImageDescriptor++)
    {
		*pImage++ = *cit_ImageDescriptor;
    }

	// find nearest neighbors using FLANN
    cv::Mat cvmIndices(nSizeObject, 2, CV_32S);
    cv::Mat cvmDists  (nSizeObject, 2, CV_32F);
    cv::flann::Index cFlannIndex(cvmImage, cv::flann::KDTreeIndexParams(4));  // using 4 randomized kdtrees
    cFlannIndex.knnSearch(cvmObject, cvmIndices, cvmDists, 2, cv::flann::SearchParams(64) ); // maximum number of leafs checked
	cout << " new " << cvmIndices.rows << endl;
    int* pIndices = cvmIndices.ptr<int>(0);
    float* pDists = cvmDists.ptr<float>(0);
    for (int i=0;i<cvmIndices.rows;++i) {
    	if (pDists[2*i]<0.6*pDists[2*i+1]) {
    		pvPtPairs_->push_back(i);
    		pvPtPairs_->push_back(pIndices[2*i]);
    	}
    }
	return;
}


struct SKeyFrame
{
    cv::Mat _cvmRGB;
    cv::Mat _cvmBW;
	cv::Mat _cvmBWOrigin;
    double* _pDepth;
	cv::flann::Index* _pKDTree;

	vector<cv::KeyPoint> _vObjectKeyPoints;
	vector<float>        _vObjectDescriptors;


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
		cv::Mat cvmBGR;
		cvtColor ( _cvmRGB, cvmBGR, CV_RGB2BGR );

		cv::imwrite( "rgb"+ strN_ + ".bmp", cvmBGR );
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
        cvtColor ( _cvmRGB, _cvmBW, CV_RGB2GRAY );
		//GaussianBlur( _cvmBWOrigin, _cvmBW, Size( 7, 7), 3 );
        // clear corners
        _vCorners.clear();
    }

    void detectCorners()
    {
		cv::Mat cvmMask;

		cv::SURF cSurf( 500, 4, 2, true );
    	cSurf( _cvmBW, cvmMask, _vObjectKeyPoints, _vObjectDescriptors );
		cout << "cSurf() Object:" << _vObjectKeyPoints.size() << endl;
		cout << "cSurf() Descriptor:" << _vObjectDescriptors.size() << endl;

        return;
    }

	void constructKDTree()
	{
		int nSizeImage = _vObjectKeyPoints.size();
		int nLengthDescriptorImage = _vObjectDescriptors.size() / nSizeImage;
		cv::Mat cvmImage( nSizeImage, nLengthDescriptorImage, CV_32F );
		PRINT( nSizeImage );
		// copy descriptors
		float* pImage = cvmImage.ptr<float>(0);
	    for(vector< float >::const_iterator cit_ImageDescriptor = _vObjectDescriptors.begin(); cit_ImageDescriptor!=_vObjectDescriptors.end(); cit_ImageDescriptor++)
    	{
			*pImage++ = *cit_ImageDescriptor;
	    }
		PRINT( *cvmImage.ptr<float>(0) );
		PRINT( *cvmImage.ptr<float>(100) );
		PRINT( *cvmImage.ptr<float>(1000) );

    	_pKDTree = new 	cv::flann::Index(cvmImage, cv::flann::KDTreeIndexParams(4));  // using 4 randomized kdtrees
		return;
	}

	void match(const SKeyFrame& sKF_ ,vector<int>* pvPtPairs_)
	{
		const vector< cv::KeyPoint >& vObjectKeyPoints_ = sKF_._vObjectKeyPoints;
		const vector< float >& vObjectDescriptors_      = sKF_._vObjectDescriptors;
		int nSizeObject = vObjectKeyPoints_.size();
		int nLengthDescriptorObject = vObjectDescriptors_.size() / nSizeObject;
		cv::Mat cvmObject( nSizeObject, nLengthDescriptorObject, CV_32F );
		// copy descriptors
		float* pObject = cvmObject.ptr<float>(0);
	    for(vector< float >::const_iterator cit_ObjectDescriptor = vObjectDescriptors_.begin(); cit_ObjectDescriptor!=vObjectDescriptors_.end(); cit_ObjectDescriptor++)
    	{
			*pObject++ = *cit_ObjectDescriptor;
	    }
		PRINT( nSizeObject );
		// find nearest neighbors using FLANN
    	cv::Mat cvmIndices(nSizeObject, 2, CV_32S);
	    cv::Mat cvmDists  (nSizeObject, 2, CV_32F);
    	_pKDTree->knnSearch(cvmObject, cvmIndices, cvmDists, 2, cv::flann::SearchParams(64) ); // maximum number of leafs checked

    	int* pIndices = cvmIndices.ptr<int>(0);
	    float* pDists = cvmDists.ptr<float>(0);
    	for (int i=0;i<cvmIndices.rows;++i) {
    		if (pDists[2*i]<0.6*pDists[2*i+1]) {
    			pvPtPairs_->push_back(i);
	    		pvPtPairs_->push_back(pIndices[2*i]);
    		}
	    }
		return;
	}
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
