#ifndef BTL_UTILITY_HELPER
#define BTL_UTILITY_HELPER
/**
* @file helper.hpp
* @brief helpers developed consistent with btl2 format, it contains a group of useful when developing btl together with
* opencv and Eigen.
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.0
* 1. << and >> converter from standard vector to cv::Mat and Eigen::Matrix<T, ROW, COL>
* 2. PRINT() for debugging
* 3. << to output vector using std::cout
* 4. exception handling and exception related macro including CHECK( condition, "error message") and THROW ("error message")
* @date 2011-03-15
*/
#define CV_SSE2 1

#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <complex>
#include <string>
#include <boost/exception/all.hpp>
#include <Eigen/Dense>
#include <boost/preprocessor/stringize.hpp>


namespace btl
{
namespace utility
{
using namespace cv;
using namespace std;
using namespace Eigen;

// based on boost stringize.hpp
#define PRINT( a ) std::cout << BOOST_PP_STRINGIZE( a ) << " = " << std::endl << a << std::flush << std::endl;

//exception based on boost
typedef boost::error_info<struct tag_my_info, std::string> CErrorInfo;
struct CError: virtual boost::exception, virtual std::exception { };
#define CHECK( condition, what) \
	if ((condition) != true)\
	{\
        CError cE;\
        cE << CErrorInfo ( what );\
        throw cE;\
	}
#define THROW(what)\
	{\
        CError cE;\
        cE << CErrorInfo ( what );\
        throw cE;\
	}\
 
template< class T >
cv::Mat_< T > convertMat ( CvMat* pcM_ );


template< class T >
Eigen::Matrix< T , 4, 4 > setOpenGLModelViewMatrix ( const Eigen::Matrix< T, 3, 3 >& mR_, const Eigen::Matrix< T, 3, 1 >& vT_ );
//converters ////////////////////////////////////////////////////////////////////
// operator >>
// other -> vector
// 1.1 Point3_ -> vector
template <class T>
vector< T >& operator << ( vector< T >& vVec_, const Point3_< T >& cvPt_ );

// 1.2 vector < Point3_ > -> vector< < > >
template <class T>
vector< vector< T > >& operator << ( vector< vector< T > >& vvVec_, const vector< Point3_ < T > >& cvPt3_ );

// 1.3 vector < vector < Point3_ > > -> vector< < < > > >
template <class T>
vector< vector< vector< T > > >& operator << ( vector< vector< vector< T > > >& vvvVec_, const vector< vector< Point3_ < T > > >& vvPt3_ );

// 2.1 Point_ -> vector
template <class T>
vector< T >& operator << ( vector< T >& vVec_, const Point_< T >& cvPt_ );

// 2.2 vector < Point_ > -> vector< < > >
template <class T>
vector< vector< T > >& operator << ( vector< vector< T > >& vvVec_, const vector< Point_< T > >& cvPt_ );

// 2.3 vector < vector < Point_ > > -> vector< < < > > >
template <class T>
vector< vector< vector< T > > >& operator << ( vector< vector< vector< T > > >& vvvVec_, const vector< vector< Point_< T > > >& vPt_ );

// 3.  Static Matrix -> vector < < > >
template < class T , int ROW, int COL >
vector< vector< T > >& operator << ( vector< vector< T > >& vvVec_,  const Eigen::Matrix< T, ROW, COL >& eiMat_ );

// 4.1 Mat_ -> vector
template < class T >
vector< vector< T > >& operator << ( vector< vector< T > >& vvVec_,  const Mat_< T >& cvMat_ );

// 4.2 vector< Mat_<> > -> vector
template < class T >
vector< vector< vector< T > > >& operator << ( vector< vector< vector< T > > >& vvvVec_,  const vector< Mat_< T > >& vmMat_ );

// 5.1 vector< Mat > -> vector
template < class T >
vector< vector< vector< T > > >& operator << ( vector< vector< vector< T > > >& vvvVec_,  const vector< Mat >& vmMat_ );



// operator <<
// vector -> other
// 1.1 vector -> Point3_
template <class T>
Point3_< T >& operator << ( Point3_< T >& cvPt_, const vector< T >& vVec_ );

// 1.2 vector < < > > -> vector< Point3_ >
template <class T>
vector< Point3_< T > >& operator << ( vector< Point3_< T > >& cvPt_, const vector< vector< T > >& vvVec_ );

// 1.3 vector < < < > > > -> vector < < Point3_ > >
template <class T>
vector< vector< Point3_< T > > >& operator << ( vector< vector< Point3_< T > > >& vvPt_, const vector< vector< vector< T > > >& vvvVec_ );

// 2.1 vector -> Point_
template <class T>
Point_< T >& operator << ( Point_< T >& cvPt_, const vector< T >& vVec_ );

// 2.2 vector < < > > -> vector< Point_ >
template <class T>
vector< Point_< T > >& operator << ( vector< Point_< T > >& cvPt_, const vector< vector< T > >& vvVec_ );

// 2.3 vector < < < > > > -> vector< < Point_ > >
template <class T>
vector< vector< Point_< T > > >& operator << ( vector< vector< Point_< T > > >& vvPt_, const vector< vector< vector< T > > >& vvvVec_ );

// 3.1 vector < < > > -> Eigen::Dynamic, Matrix
template < class T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& operator << ( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& eiMat_, const vector< vector< T > >& vvVec_ );

// 3.2 vector < < > > -> Static, Matrix
template < class T , int ROW, int COL>
Eigen::Matrix< T, ROW, COL >& operator << ( Eigen::Matrix< T, ROW, COL >& eiMat_, const vector< vector< T > >& vvVec_ );

// 4.1 vector -> Mat_ -> vector
template < class T >
vector< vector< T > >& operator << ( vector< vector< T > >& vvVec_, const Mat_< T >& cvMat_ );

// 4.2 vector< < < > > > -> vector< Mat_<> >
template < class T >
vector< Mat_< T > >& operator << ( vector< Mat_< T > >& vmMat_ ,  const vector< vector< vector< T > > >& vvvVec_ );

// 5.1 vector< < < > > > -> vector< Mat >
template < class T >
vector< Mat >& operator << ( vector< Mat >& vmMat_ ,  const vector< vector< vector< T > > >& vvvVec_ );



// operator <<
// other -> other
// 1.1 Mat -> Static Matrix
template < class T, int ROW, int COL  >
Eigen::Matrix< T, ROW, COL >& operator << ( Eigen::Matrix< T, ROW, COL >& eiVec_, const Mat_< T >& cvVec_ );
// 1.2 Static Matrix -> Mat
template < class T, int ROW, int COL  >
Mat_< T >& operator << ( Mat_< T >& cvVec_, const Eigen::Matrix< T, ROW, COL >& eiVec_ );
// 1.3 Static Matrix -> Mat_;
template < class T, int ROW, int COL  >
Mat& operator << ( Mat& cvVec_, const Eigen::Matrix< T, ROW, COL >& eiVec_ );

// 2.1 Point3_ -> Vector
template < class T >
Eigen::Matrix<T, 3, 1> & operator << ( Eigen::Matrix< T, 3, 1 >& eiVec_, const Point3_< T >& cvVec_ );

// 3.1 CvMat -> cv::Mat_
template < class T >
cv::Mat_< T >& operator << ( cv::Mat_< T >& cppM_, const CvMat& cM_ );
// 3.2 cv::Mat_ -> CvMat
template < class T >
CvMat& operator << ( CvMat& cM_, const cv::Mat_< T >& cppM_ );



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// operator >>
// other -> vector
// 1.1 Point3_ -> vector
template <class T>
const Point3_< T >& operator >> ( const Point3_< T >& cvPt_, vector< T >& vVec_ );

// 1.2 vector < Point3_ > -> vector< < > >
template <class T>
const vector< Point3_ < T > >& operator >> ( const vector< Point3_ < T > >& vPt3_, vector< vector< T > >& vvVec_ );

// 1.3 vector < vector < Point3_ > > -> vector< < < > > >
template <class T>
const vector< vector< Point3_ < T > > >& operator >> ( const vector< vector< Point3_ < T > > >& vvPt3_, vector< vector< vector< T > > >& vvvVec_ );

// 2.1 Point_ -> vector
template <class T>
const Point_< T >& operator >> ( const Point_< T >& cvPt_, vector< T >& vVec_ );

// 2.2 vector < Point_ > -> vector< < > >
template <class T>
const vector< Point_< T > >& operator >> ( const vector< Point_< T > >& vPt_, vector< vector< T > >& vvVec_ );

// 2.3 vector < vector < Point_ > > -> vector< < < > > >
template <class T>
const vector< Point_< T > >&  operator >> ( const vector< Point_< T > >& vPt_, vector< vector< vector< T > > >& vvvVec_ );

// 3.  Static Matrix -> vector < < > >
template < class T , int ROW, int COL >
const Eigen::Matrix< T, ROW, COL >& operator >> ( const Eigen::Matrix< T, ROW, COL >& eiMat_, vector< vector< T > >& vvVec_ );

// 4.1 Mat_ -> vector
template < class T >
const Mat_< T >& operator >> ( const Mat_< T >& cvMat_, vector< vector< T > >& vvVec_ );

// 4.2 vector< Mat_<> > -> vector
template < class T >
const vector< Mat_< T > >& operator >> ( const vector< Mat_< T > >& vmMat_, vector< vector< vector< T > > >& vvvVec_ );

// 5.1 vector< Mat > -> vector
template < class T >
const vector< Mat >& operator >> ( const vector< Mat >& vmMat_, vector< vector< vector< T > > >& vvvVec_ );





// vector -> other
// 1.1 vector -> Point3_
template <class T>
const vector< T >& operator >> ( const vector< T >& vVec_, Point3_< T >& cvPt_ );

// 1.2 vector < < > > -> vector< Point3_ >
template <class T>
const vector< vector< T > >& operator >> ( const vector< vector< T > >& vvVec_, vector< Point3_< T > >& cvPt_ );

// 1.3 vector < < < > > > -> vector < < Point3_ > >
template <class T>
const vector< vector< vector< T > > >& operator >> ( const vector< vector< vector< T > > >& vvvVec_, vector< vector< Point3_< T > > >& vvPt_ );

// 2.1 vector -> Point_
template <class T>
const vector< T >& operator >> ( const vector< T >& vVec_, Point_< T >& cvPt_ );

// 2.2 vector < < > > -> vector< Point_ >
template <class T>
const vector< vector< T > >& operator >> (  const vector< vector< T > >& vvVec_, vector< Point_< T > >& cvPt_ );

// 2.3 vector < < < > > > -> vector< < Point_ > >
template <class T>
const vector< vector< vector< T > > >& operator >> ( const vector< vector< vector< T > > >& vvvVec_, vector< vector< Point_< T > > >& vvPt_ );

// 3.1 vector < < > > -> Eigen::Dynamic, Matrix
template < class T >
const vector< vector< T > >& operator >> ( const vector< vector< T > >& vvVec_, Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& eiMat_ );

// 3.2 vector < < > > -> Static, Matrix
template < class T , int ROW, int COL>
const vector< vector< T > >& operator >> ( const vector< vector< T > >& vvVec_, Eigen::Matrix< T, ROW, COL >& eiMat_ );

// 4.1 vector -> Mat_ -> vector
template < class T >
const Mat_< T >& operator >> ( const Mat_< T >& cvMat_, vector< vector< T > >& vvVec_ );

// 4.2 vector< < < > > > -> vector< Mat_<> >
template < class T >
const vector< vector< vector< T > > >& operator >> ( const vector< vector< vector< T > > >& vvvVec_, vector< Mat_< T > >& vmMat_ );

// 5.1 vector< < < > > > -> vector< Mat >
template < class T >
const vector< vector< vector< T > > >& operator >> ( const vector< vector< vector< T > > >& vvvVec_, vector< Mat >& vmMat_ );








//not implemented
//template < class T, int ROW, int COL  >
//Eigen::Matrix< T, ROW, COL >& operator << ( Eigen::Matrix< T, ROW, COL >& eiVec_, const MatExpr& cvVec_);

//print vector
template <class T>
std::ostream& operator << ( std::ostream& os, const vector< T > & v );

}//utility
}//btl


// ====================================================================
// === Implementation
namespace btl
{
namespace utility
{

template < class T, int ROW >
vector< T >& operator << ( vector< T >& vVec_, const Eigen::Matrix< T, ROW, 1 >& eiVec_ )
{
    vVec_.clear();

    for ( int r = 0; r < ROW; r++ )
    {
        vVec_.push_back ( eiVec_ ( r, 0 ) );
    }

    return vVec_;
}

template < class T, int ROW >
const Eigen::Matrix< T, ROW, 1 >&  operator >> ( const Eigen::Matrix< T, ROW, 1 >& eiVec_, vector< T >& vVec_ )
{
    vVec_ << eiVec_;
}

template < class T >
Eigen::Matrix< T, Eigen::Dynamic, 1 >& operator << ( Eigen::Matrix< T, Eigen::Dynamic, 1 >& eiVec_, const vector< T >& vVec_ )
{
    if ( vVec_.empty() )
    {
        eiVec_.resize ( 0, 0 );
    }
    else
    {
        eiVec_.resize ( vVec_.size(), 1 );
    }

    for ( int r = 0; r < vVec_.size(); r++ )
    {
        eiVec_ ( r, 0 ) = vVec_[r];
    }

    return eiVec_;
}

template < class T >
const vector< T >&  operator >> ( const vector< T >& vVec_, Eigen::Matrix< T, Eigen::Dynamic, 1 >& eiVec_ )
{
    eiVec_ << vVec_;
}

template < class T, int ROW >
Eigen::Matrix< T, ROW, 1 >& operator << ( Eigen::Matrix< T, ROW, 1 >& eiVec_, const vector< T >& vVec_ )
{
    CHECK ( eiVec_.rows() == vVec_.size(), "Eigen::Vector << vector wrong!" );

    for ( int r = 0; r < vVec_.size(); r++ )
    {
        eiVec_ ( r, 0 ) = vVec_[r];
    }

    return eiVec_;
}

template < class T, int ROW >
const vector< T >&  operator >> ( const vector< T >& vVec_, Eigen::Matrix< T, ROW, 1 >& eiVec_ )
{
    eiVec_ << vVec_;
}


/*template < class T>
std::*/

template < class T, int ROW, int COL  >
vector< vector< T > >& operator << ( vector< vector< T > >& vvVec_, const Eigen::Matrix< T, ROW, COL >& eiMat_ )
{
    vvVec_.clear();

    for ( int r = 0; r < ROW; r++ )
    {
        vector< T > v;

        for ( int c = 0; c < COL; c++ )
        {
            v.push_back ( eiMat_ ( r, c ) );
        }

        vvVec_.push_back ( v );
    }

    return vvVec_;
}

template < class T >
Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& operator << ( Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& eiMat_, const vector< vector< T > >& vvVec_ )
{
    if ( vvVec_.empty() )
    {
        eiMat_.resize ( 0, 0 );
    }
    else
    {
        eiMat_.resize ( vvVec_.size(), vvVec_[0].size() );
    }

    for ( int r = 0; r < vvVec_.size(); r++ )
        for ( int c = 0; c < vvVec_[r].size(); c++ )
        {
            eiMat_ ( r, c ) = vvVec_[r][c];
        }

    return eiMat_;
}

template < class T , int ROW, int COL>
Eigen::Matrix< T, ROW, COL >& operator << ( Eigen::Matrix< T, ROW, COL >& eiMat_, const vector< vector< T > >& vvVec_ )
{
    if ( ROW != vvVec_.size() )
    {
        CError cE;
        cE << CErrorInfo ( " vector< vector<> > is inconsistent with ROW of Matrix. \n" );
        throw cE;
    }
    else if ( COL != vvVec_[0].size() )
    {
        CError cE;
        cE << CErrorInfo ( " vector<> is inconsistent with COL of Matrix. \n" );
        throw cE;
    }

    for ( int r = 0; r < vvVec_.size(); r++ )
        for ( int c = 0; c < vvVec_[r].size(); c++ )
        {
            eiMat_ ( r, c ) = vvVec_[r][c];
        }

    return eiMat_;
}

template <class T>
vector< T >& operator << ( vector< T >& vVec_, const Point3_< T >& cvPt_ )
{
    vVec_.clear();
    vVec_.push_back ( cvPt_.x );
    vVec_.push_back ( cvPt_.y );
    vVec_.push_back ( cvPt_.z );
    return vVec_;
}

template <class T>
Point3_< T >& operator << ( Point3_< T >& cvPt_, const vector< T >& vVec_ )
{
    if ( vVec_.empty() )
    {
        cvPt_.x = cvPt_.y = cvPt_.z = 0;
    }
    else if ( 3 != vVec_.size() )
    {
        CError cE;
        cE << CErrorInfo ( " vector<> is inconsistent with Point3_. \n" );
        throw cE;
    }
    else
    {
        cvPt_.x = vVec_[0];
        cvPt_.y = vVec_[1];
        cvPt_.z = vVec_[2];
    }

    return cvPt_;
}
//vector -> Point_
template <class T>
Point_< T >& operator << ( Point_< T >& cvPt_, const vector< T >& vVec_ )
{
    if ( vVec_.empty() )
    {
        cvPt_.x = cvPt_.y = 0;
    }
    else if ( 2 != vVec_.size() )
    {
        CError cE;
        cE << CErrorInfo ( " vector<> is inconsistent with Point3_. \n" );
        throw cE;
    }
    else
    {
        cvPt_.x = vVec_[0];
        cvPt_.y = vVec_[1];
    }

    return cvPt_;
}
//Point_ -> vector
template <class T>
vector< T >& operator << ( vector< T >& vVec_, const Point_< T >& cvPt_ )
{
    vVec_.clear();
    vVec_.push_back ( cvPt_.x );
    vVec_.push_back ( cvPt_.y );
    return vVec_;
}



template < class T, int ROW, int COL  >
Eigen::Matrix< T, ROW, COL >& operator << ( Eigen::Matrix< T, ROW, COL >& eiVec_, const Mat_< T >& cvVec_ )
{
    if ( ROW != cvVec_.rows || COL != cvVec_.cols )
    {
        CError cE;
        cE << CErrorInfo ( " Mat dimension is inconsistent with Vector3d . \n" );
		//PRINT( cvVec_.cols ); PRINT( cvVec_.rows );
        throw cE;
    }

    for ( int r = 0; r < ROW; r++ )
        for ( int c = 0; c < COL; c++ )
        {
            eiVec_ ( r, c ) = cvVec_.template at< T > ( r, c );
        }

    return eiVec_;
}
// 4.1 vector -> Mat_ -> vector
template < class T >
vector< vector< T > >& operator << ( vector< vector< T > >& vvVec_,  const Mat_< T >& cvMat_ )
{
    vvVec_.clear();

    for ( int r = 0; r < cvMat_.rows; r++ )
    {
        vector< T > v;

        for ( int c = 0; c < cvMat_.cols; c++ )
        {
            v.push_back ( cvMat_.template at< T > ( r, c ) );
        }

        vvVec_.push_back ( v );
    }

    return vvVec_;
}

template < class T >
Mat_< T >& operator << ( Mat_< T >& cvMat_,  const  vector< vector< T > >& vvVec_ )
{

    if ( vvVec_.empty() || vvVec_[0].empty() )
    {
        CError cE;
        cE << CErrorInfo ( " the input vector<> cannot be empty.\n" );
        throw cE;
    }

    cvMat_.create ( ( int ) vvVec_.size(), ( int ) vvVec_[0].size() );

    for ( int r = 0; r < ( int ) vvVec_.size(); r++ )
    {
        for ( int c = 0; c < vvVec_[r].size(); c++ )
        {
            cvMat_.template at< T > ( r, c ) = vvVec_[r][c];
        }
    }

    return cvMat_;
}

template < class T >
vector< vector< vector< T > > >& operator << ( vector< vector< vector< T > > >& vvvVec_,  const vector< Mat_< T > >& vmMat_ )
{
    vvvVec_.clear();
    typename vector< Mat_< T > >::const_iterator constItr = vmMat_.begin();

    for ( ; constItr != vmMat_.end(); ++constItr )
    {
        vector< vector< T > > vv;
        vv << ( *constItr );
        vvvVec_.push_back ( vv );
    }

    return vvvVec_;
}

template < class T >
vector< vector< vector< T > > >& operator << ( vector< vector< vector< T > > >& vvvVec_,  const vector< Mat >& vmMat_ )
{
    vector< Mat_< T > > vmTmp;

    typename vector< Mat >::const_iterator constItr = vmMat_.begin();

    for ( ; constItr != vmMat_.end(); ++constItr )
    {
        vmTmp.push_back ( Mat_< T > ( *constItr ) );
    }

    vvvVec_ << vmTmp;
    return vvvVec_;
}


template < class T >
vector< Mat_< T > >& operator << ( vector< Mat_< T > >& vmMat_ ,  const vector< vector< vector< T > > >& vvvVec_ )
{
    vmMat_.clear();
    typename vector< vector< vector< T > > >::const_iterator constVectorVectorItr = vvvVec_.begin();

    for ( ; constVectorVectorItr != vvvVec_.end(); ++ constVectorVectorItr )
    {
        Mat_< T > mMat;
        mMat << ( *constVectorVectorItr );
        vmMat_.push_back ( mMat );
    }

    return vmMat_;
}

template < class T >
vector< Mat >& operator << ( vector< Mat >& vmMat_ ,  const vector< vector< vector< T > > >& vvvVec_ )
{
    vector< Mat_< T > > vmTmp;
    vmTmp << vvvVec_;
    typename vector< Mat_< T > >::const_iterator constItr = vmTmp.begin();

    for ( ; constItr != vmTmp.end(); ++constItr )
    {
        vmMat_.push_back ( Mat ( *constItr ) );
    }

    return vmMat_;
}

//vector< Point_ > -> vector< < > >
template <class T>
vector< vector< T > >& operator << ( vector< vector< T > >& vvVec_, const vector< Point_ < T > >& cvPt_ )
{
    using namespace std;
    using namespace cv;

    vvVec_.clear();
    typename vector< Point_< T > >::const_iterator constItr = cvPt_.begin();

    for ( ; constItr != cvPt_.end(); ++constItr )
    {
        vector < T > v;
        v << *constItr;
        vvVec_.push_back ( v );
    }

    return vvVec_;
}

//vector< Point3_ > -> vector< < > >
template <class T>
vector< vector< T > >& operator << ( vector< vector< T > >& vvVec_, const vector< Point3_ < T > >& cvPt3_ )
{
    using namespace std;
    using namespace cv;

    vvVec_.clear();
    typename vector< Point3_< T > >::const_iterator constItr = cvPt3_.begin();

    for ( ; constItr != cvPt3_.end(); ++constItr )
    {
        vector < T > v;
        v << *constItr;
        vvVec_.push_back ( v );
    }

    return vvVec_;
}


//vector < <> > -> vector< Point_ >
template <class T>
vector< Point_< T > >& operator << ( vector< Point_< T > >& cvPt_, const vector< vector< T > >& vvVec_ )
{
    cvPt_.clear();
    typename vector< vector< T > >::const_iterator constItr = vvVec_.begin();

    for ( ; constItr != vvVec_.end(); ++constItr )
    {
        Point_< T > Pt;
        Pt << *constItr;
        cvPt_.push_back ( Pt );
    }

    return cvPt_;
}

// 2.3 vector < < < > > > -> vector< < Point_ > >
template <class T>
vector< vector< Point_< T > > >& operator << ( vector< vector< Point_< T > > >& vvPt_, const vector< vector< vector< T > > >& vvvVec_ )
{
    vvPt_.clear();
    typename vector< vector< vector< T > > >::const_iterator constItr = vvvVec_.begin();

    for ( ; constItr != vvvVec_.end(); ++constItr )
    {
        vector< Point_< T > > vPt;
        vPt << *constItr;
        vvPt_.push_back ( vPt );
    }

    return vvPt_;
}


//vector < <> > -> vector< Point3_ >
template <class T>
vector< Point3_< T > >& operator << ( vector< Point3_< T > >& cvPt_, const vector< vector< T > >& vvVec_ )
{
    cvPt_.clear();
    typename vector< vector< T > >::const_iterator constItr = vvVec_.begin();

    for ( ; constItr != vvVec_.end(); ++constItr )
    {
        Point3_< T > Pt3;
        Pt3 << *constItr;
        cvPt_.push_back ( Pt3 );
    }

    return cvPt_;
}

// 1.3 vector < < < > > > -> vector < < Point3_ > >
template <class T>
vector< vector< Point3_< T > > >& operator << ( vector< vector< Point3_< T > > >& vvPt_, const vector< vector< vector< T > > >& vvvVec_ )
{
    vvPt_.clear();
    typename vector< vector< vector< T > > >::const_iterator constItr = vvvVec_.begin();

    for ( ; constItr != vvvVec_.end(); ++constItr )
    {
        vector< Point3_< T > > vPt3;
        vPt3 << *constItr;
        vvPt_.push_back ( vPt3 );
    }

    return vvPt_;
}



//vector< vector< Point3_ > > -> vector< < < > > >
template <class T>
vector< vector< vector< T > > >& operator << ( vector< vector< vector< T > > >& vvvVec_, const vector< vector< Point3_ < T > > >& vvPt3_ )
{
    typename vector< vector< Point3_ < T > > >::const_iterator constItr = vvPt3_.begin();

    for ( ; constItr != vvPt3_.end(); ++ constItr )
    {
        vector< vector < T > > vv;
        vv << *constItr;
        vvvVec_.push_back ( vv );
    }

    return vvvVec_;
}

template <class T>
vector< vector< vector< T > > >& operator << ( vector< vector< vector< T > > >& vvvVec_, const vector< vector< Point_< T > > >& vvPt_ )
{
    typename vector< vector< Point_ < T > > >::const_iterator constItr = vvPt_.begin();

    for ( ; constItr != vvPt_.end(); ++ constItr )
    {
        vector< vector < T > > vv;
        vv << *constItr;
        vvvVec_.push_back ( vv );
    }

    return vvvVec_;
}

// 2.1 Point3_ -> Vector
template < class T >
Eigen::Matrix<T, 3, 1> & operator << ( Eigen::Matrix< T, 3, 1 >& eiVec_, const Point3_< T >& cvVec_ )
{
    eiVec_ ( 0 ) = cvVec_.x;
    eiVec_ ( 1 ) = cvVec_.y;
    eiVec_ ( 2 ) = cvVec_.z;
}

// other -> other
// 1.2 Static Matrix -> Mat_ < >
template < class T, int ROW, int COL  >
Mat_< T >& operator << ( Mat_< T >& cvVec_, const Eigen::Matrix< T, ROW, COL >& eiVec_ )
{
    cvVec_.create ( ROW, COL );

    for ( int r = 0; r < ROW; r++ )
        for ( int c = 0; c < COL; c++ )
        {
            cvVec_.template at<T> ( r, c ) = eiVec_ ( r, c );
        }

    return cvVec_;
}
// other -> other
// 1.2 Static Matrix -> Mat
template < class T, int ROW, int COL  >
Mat& operator << ( Mat& cvVec_, const Eigen::Matrix< T, ROW, COL >& eiVec_ )
{
    cvVec_ = Mat_< T > ( ROW, COL );

    for ( int r = 0; r < ROW; r++ )
        for ( int c = 0; c < COL; c++ )
        {
            cvVec_.template at<T> ( r, c ) = eiVec_ ( r, c );
        }

    return cvVec_;
}

//CvMat -> cv::Mat_
template < class T >
cv::Mat_< T >& operator << ( cv::Mat_< T >& cppM_, const CvMat& cM_ )
{
    CHECK ( cppM_.type() == CV_MAT_TYPE ( cM_.type ) , "operator CvMat << Mat_: the type of cv::Mat_ and CvMat is inconsistent. \n" );
    CHECK ( CV_IS_MAT ( &cM_ ),                       "operator CvMat << Mat_: the data of CvMat must be pre-allocated. \n" );

    cppM_.create ( cM_.rows, cM_.cols );

    for ( int r = 0; r < cM_.rows; r++ )
        for ( int c = 0; c < cM_.cols; c++ )
        {
            cppM_.template at< T > ( r, c ) = CV_MAT_ELEM ( cM_, T, r, c );
        }

    return cppM_;
}

//cv::Mat_ -> CvMat
template < class T >
CvMat& operator << ( CvMat& cM_, const cv::Mat_< T >& cppM_ )
{
    CHECK ( cppM_.type() == CV_MAT_TYPE ( cM_.type ) , "operator CvMat << Mat_: the type of cv::Mat_ and CvMat is inconsistent. \n" );
    CHECK ( CV_IS_MAT ( &cM_ ),                       "operator CvMat << Mat_: the data of CvMat is not allocated. \n" );
    CHECK ( cppM_.rows == cM_.rows ,                 "operator CvMat << Mat_: the # of rows of cv::Mat_ and CvMat is inconsistent. \n" );
    CHECK ( cppM_.cols == cM_.cols,                  "operator CvMat << Mat_: the # of cols of cv::Mat_ and CvMat is inconsistent. \n" );


    for ( int r = 0; r < cppM_.rows; r++ )
        for ( int c = 0; c < cppM_.cols; c++ )
        {
            CV_MAT_ELEM ( cM_, T, r, c ) = cppM_.template at< T > ( r, c );
        }

    return cM_;
}

//cv::Mat_ -> CvMat
template < class T >
void assignPtr ( cv::Mat_< T >* cppM_, CvMat* pcM_ )
{
    for ( int r = 0; r < cppM_->rows; r++ )
        for ( int c = 0; c < cppM_->cols; c++ )
        {
            CV_MAT_ELEM ( *pcM_, T, r, c ) = cppM_->template at< T > ( r, c );
        }

}
//cv::CvMat -> Mat_
template < class T >
void assignPtr ( CvMat* pcM_,  cv::Mat_< T >* cppM_ )
{
    for ( int r = 0; r < pcM_->rows; r++ )
        for ( int c = 0; c < pcM_->cols; c++ )
        {
            cppM_->template at< T > ( r, c ) = CV_MAT_ELEM ( *pcM_, T, r, c );
        }
}



// 1.1 Point3_ -> vector
template <class T>
const Point3_< T >& operator >> ( const Point3_< T >& cvPt_, vector< T >& vVec_ )
{
    vVec_ << cvPt_;
}

// 1.2 vector < Point3_ > -> vector< < > >
template <class T>
const vector< Point3_ < T > >& operator >> ( const vector< Point3_ < T > >& vPt3_, vector< vector< T > >& vvVec_ )
{
    vvVec_ << vPt3_;
}

// 1.3 vector < vector < Point3_ > > -> vector< < < > > >
template <class T>
const vector< vector< Point3_ < T > > >& operator >> ( const vector< vector< Point3_ < T > > >& vvPt3_, vector< vector< vector< T > > >& vvvVec_ )
{
    vvvVec_ << vvPt3_;
}

// 2.1 Point_ -> vector
template <class T>
const Point_< T >& operator >> ( const Point_< T >& cvPt_, vector< T >& vVec_ )
{
    vVec_ << cvPt_;
}

// 2.2 vector < Point_ > -> vector< < > >
template <class T>
const vector< Point_< T > >& operator >> ( const vector< Point_< T > >& vPt_, vector< vector< T > >& vvVec_ )
{
    vvVec_ << vPt_;
}

// 2.3 vector < vector < Point_ > > -> vector< < < > > >
template <class T>
const vector< Point_< T > >&  operator >> ( const vector< vector< Point_< T > > >& vvPt_, vector< vector< vector< T > > >& vvvVec_ )
{
    vvvVec_ << vvPt_;
}

// 3.  Static Matrix -> vector < < > >
template < class T , int ROW, int COL >
const Eigen::Matrix< T, ROW, COL >& operator >> ( const Eigen::Matrix< T, ROW, COL >& eiMat_, vector< vector< T > >& vvVec_ )
{
    vvVec_ << eiMat_;
}

// 4.1 Mat_ -> vector
template < class T >
const Mat_< T >& operator >> ( const Mat_< T >& cvMat_, vector< vector< T > >& vvVec_ )
{
    vvVec_ << cvMat_;
}

// 4.2 vector< Mat_<> > -> vector
template < class T >
const vector< Mat_< T > >& operator >> ( const vector< Mat_< T > >& vmMat_, vector< vector< vector< T > > >& vvvVec_ )
{
    vvvVec_ << vmMat_;
}

// 5.1 vector< Mat > -> vector
template < class T >
const vector< Mat >& operator >> ( const vector< Mat >& vmMat_, vector< vector< vector< T > > >& vvvVec_ )
{
    vvvVec_ << vmMat_;
	return vmMat_;
}

// operator >>
// vector -> other
// 1.1 vector -> Point3_Eigen::Matrix<short int, 2, 1,
template <class T>
const vector< T >& operator >> ( const vector< T >& vVec_, Point3_< T >& cvPt_ )
{
    cvPt_ << vVec_;
}

// 1.2 vector < < > > -> vector< Point3_ >
template <class T>
const vector< vector< T > >& operator >> ( const vector< vector< T > >& vvVec_ , vector< Point3_< T > >& cvPt_ )
{
    cvPt_ << vvVec_;
}

// 1.3 vector < < < > > > -> vector < < Point3_ > >
template <class T>
const vector< vector< vector< T > > >& operator >> ( const vector< vector< vector< T > > >& vvvVec_, vector< vector< Point3_< T > > >& vvPt_ )
{
    vvPt_ << vvvVec_;
}

// 2.1 vector -> Point_
template <class T>
const vector< T >& operator >> ( const vector< T >& vVec_, Point_< T >& cvPt_ )
{
    cvPt_ << vVec_;
}

// 2.2 vector < < > > -> vector< Point_ >
template <class T>
const vector< vector< T > >& operator >> ( const vector< vector< T > >& vvVec_, vector< Point_< T > >& cvPt_ )
{
    cvPt_ << vvVec_;
}

// 2.3 vector < < < > > > -> vector< < Point_ > >
template <class T>
const vector< vector< vector< T > > >& operator >> ( const vector< vector< vector< T > > >& vvvVec_, vector< vector< Point_< T > > >& vvPt_ )
{
    vvPt_ << vvvVec_;
}

// 3.1 vector < < > > -> Eigen::Dynamic, Matrix
template < class T >
const vector< vector< T > >& operator >> (  const vector< vector< T > >& vvVec_, Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >& eiMat_ )
{
    eiMat_ << vvVec_;
}

// 3.2 vector < < > > -> Static, Matrix
template < class T , int ROW, int COL>
const vector< vector< T > >& operator >> ( const vector< vector< T > >& vvVec_, Eigen::Matrix< T, ROW, COL >& eiMat_ )
{
    eiMat_ << vvVec_;
}

// 4.1 vector -> Mat_
template < class T >
const vector< vector< T > >& operator >> ( const vector< vector< T > >& vvVec_,  Mat_< T >& cvMat_ )
{
    cvMat_ << vvVec_;
}

// 4.2 vector< < < > > > -> vector< Mat_<> >
template < class T >
const vector< vector< vector< T > > >& operator >> ( const vector< vector< vector< T > > >& vvvVec_, vector< Mat_< T > >& vmMat_ )
{
    vmMat_ << vvvVec_;
}

// 5.1 vector< < < > > > -> vector< Mat >
template < class T >
const vector< vector< vector< T > > >& operator >> ( const vector< vector< vector< T > > >& vvvVec_, vector< Mat >& vmMat_ )
{
    vmMat_ << vvvVec_;
}

template< class T >
Eigen::Matrix< T , 4, 4 > setOpenGLModelViewMatrix ( const Eigen::Matrix< T, 3, 3 >& mR_, const Eigen::Matrix< T, 3, 1 >& vT_ )
{
    // column first for pGLMat_[16];
    // row first for Matrix3d;
    // pGLMat_[ 0] =  mR_(0,0); pGLMat_[ 4] =  mR_(0,1); pGLMat_[ 8] =  mR_(0,2); pGLMat_[12] =  vT_(0);
    // pGLMat_[ 1] = -mR_(1,0); pGLMat_[ 5] = -mR_(1,1); pGLMat_[ 9] = -mR_(1,2); pGLMat_[13] = -vT_(1);
    // pGLMat_[ 2] = -mR_(2,0); pGLMat_[ 6] = -mR_(2,1); pGLMat_[10] = -mR_(2,2); pGLMat_[14] = -vT_(2);
    // pGLMat_[ 3] =  0;        pGLMat_[ 7] =  0;        pGLMat_[11] =  0;        pGLMat_[15] = 1;
    /*
        Eigen::Matrix< T , 4, 4 > mMat;
        mMat(0, 0) =  mR_(0,0); mMat(1,0) =  mR_(0,1); mMat(2,0) =  mR_(0,2); mMat(3,0) =  vT_(0);
        mMat(0, 1) = -mR_(1,0); mMat(1,1) = -mR_(1,1); mMat(2,1) = -mR_(1,2); mMat(3,1) = -vT_(1);
        mMat(0, 2) = -mR_(2,0); mMat(1,2) = -mR_(2,1); mMat(2,2) = -mR_(2,2); mMat(3,2) = -vT_(2);
        mMat(0, 3) =  0;        mMat(1,3) =  0;        mMat(2,3) =  0;        mMat(3,3) = 1;
        mMat.transposeInPlace();
    */

    Eigen::Matrix< T , 4, 4 > mMat;
    mMat ( 0, 0 ) =  mR_ ( 0, 0 );
    mMat ( 1, 0 ) = -mR_ ( 1, 0 );
    mMat ( 2, 0 ) = -mR_ ( 2, 0 );
    mMat ( 3, 0 ) =  0;
    mMat ( 0, 1 ) =  mR_ ( 0, 1 );
    mMat ( 1, 1 ) = -mR_ ( 1, 1 );
    mMat ( 2, 1 ) = -mR_ ( 2, 1 );
    mMat ( 3, 1 ) =  0;
    mMat ( 0, 2 ) =  mR_ ( 0, 2 );
    mMat ( 1, 2 ) = -mR_ ( 1, 2 );
    mMat ( 2, 2 ) = -mR_ ( 2, 2 );
    mMat ( 3, 2 ) =  0;
    mMat ( 0, 3 ) =  vT_ ( 0 );
    mMat ( 1, 3 ) = -vT_ ( 1 );
    mMat ( 2, 3 ) = -vT_ ( 2 );
    mMat ( 3, 3 ) =  1;

    return mMat;
}

// for print
template <class T>
std::ostream& operator << ( std::ostream& os, const vector< T > & v )
{
    os << "[";

    for ( typename vector< T >::const_iterator constItr = v.begin(); constItr != v.end(); ++constItr )
    {
        os << " " << ( *constItr ) << " ";
    }

    os << "]";
    return os;
}

// for print
template <class T1, class T2>
std::ostream& operator << ( std::ostream& os, const map< T1, T2 > & mp )
{
    os << "[";

    for ( typename map< T1, T2 >::const_iterator constItr = mp.begin(); constItr != mp.end(); ++constItr )
    {
        os << " " << ( *constItr ).first << ": " << ( *constItr ).second << " ";
    }

    os << "]";
    return os;
}

template <class T>
std::ostream& operator << ( std::ostream& os, const Size_< T >& s )
{
    os << "[ " << s.width << ", " << s.height << " ]";
    return os;
}

template <class T>
std::ostream& operator << ( std::ostream& os, const list< T >& l_ )
{
	os << "[";
	for ( typename list< T >::const_iterator cit_List = l_.begin(); cit_List != l_.end(); cit_List++ )
	{
		os << " " << *cit_List << " ";
	}
	os << "]";
	return os;
}

//used by freenect depth images
template <class T>
T rawDepthToMetersLinear ( int nRawDepth_, const Mat_< T >& mPara_ = Mat_< T >() )
{
    double k1 = -0.002788688001059727;
    double k2 = 3.330949940125644;

    if ( !mPara_.empty() )
    {
        k1 = mPara_.template at< T > ( 0, 0 );
        k2 = mPara_.template at< T > ( 1, 0 );
    }

    if ( nRawDepth_ < 2047 )
    {
        T tDepth = T ( 1.0 / ( T ( nRawDepth_ ) * k1 + k2 ) );
        tDepth = tDepth > 0 ? tDepth : 0;
        return tDepth;
    }

    return 0;
}
//used by freenect depth images
template <class T>
T rawDepthToMetersTanh ( int nRawDepth_, const Mat_< T >& mPara_ = Mat_< T >() )
{
    double k1 = 1.1863;
    double k2 = 2842.5;
    double k3 = 0.1236;

    if ( !mPara_.empty() )
    {
        k1 = mPara_.template at< T > ( 0, 0 );
        k2 = mPara_.template at< T > ( 1, 0 );
        k3 = mPara_.template at< T > ( 2, 0 );
    }

    //PRINT( nRawDepth_ );

    double depth = nRawDepth_;

    if ( depth < 5047 )
    {
        depth = k3 * tan ( depth / k2 + k1 );
        //PRINT( depth );
    }
    else
    {
        depth = 0;
    }

    return T ( depth );
}

//used by freenect depth images
template< class T >
T rawDepth ( int nX_, int nY_, const Mat& cvmDepth_ )
{
    unsigned char* pDepth = ( unsigned char* ) cvmDepth_.data;
    pDepth += ( nY_ * cvmDepth_.cols + nX_ ) * 3;
    int nR = * ( pDepth );
    int nG = * ( pDepth + 1 );
    int nB = * ( pDepth + 2 );
    T nRawDepth = nR * 256 + nG;
    /*
    	PRINT( nR );
    	PRINT( nG );
    	PRINT( nB );
    	PRINT( nRawDepth );
    */
    return nRawDepth;
}

template< class T >
Mat_< T > getColor ( int nX_, int nY_, const Mat& cvmImg_ )
{
    unsigned char* pDepth = cvmImg_.data;
    pDepth += ( nY_ * cvmImg_.cols + nX_ ) * 3;
    T nR = * ( pDepth );
    T nG = * ( pDepth + 1 );
    T nB = * ( pDepth + 2 );

    Mat_< T > rgb = ( Mat_< T > ( 3, 1 ) << nR, nG, nB );

    /*
    	PRINT( nR );
    	PRINT( nG );
    	PRINT( nB );
    	PRINT( rgb );
    */
    return rgb;
}

template< class T >
T* getColorPtr ( const short& nX_, const short& nY_, const Mat& cvmImg_ )
{
    if ( nX_ < 0 || nX_ >= ( short ) cvmImg_.cols || nY_ < 0 || nY_ >= ( short ) cvmImg_.rows )
    {
        return ( T* ) NULL;
    }

    unsigned char* pDepth = cvmImg_.data  + ( nY_ * cvmImg_.cols + nX_ ) * 3;
    return ( T* ) pDepth;
}

template< class T1, class T2 >
void unprojectCamera2World ( const int& nX_, const int& nY_, const unsigned short& nD_, const Eigen::Matrix< T1, 3, 3 >& mK_, Eigen::Matrix< T2, 3, 1 >* pVec_ )
{
//the pixel coordinate is defined w.r.t. camera reference, which is defined as x-lef, y-downward and z-foward. It's
//a right hand system.
//when rendering the point using opengl's camera reference which is defined as x-left, y-upward and z-backward. the
//	glVertex3d ( Pt(0), -Pt(1), -Pt(2) );
    if ( nD_ > 400 )
    {
        T2 dZ = nD_ / 1000.; //convert to meter
        T2 dX = ( nX_ - mK_ ( 0, 2 ) ) / mK_ ( 0, 0 ) * dZ;
        T2 dY = ( nY_ - mK_ ( 1, 2 ) ) / mK_ ( 1, 1 ) * dZ;
        ( *pVec_ ) << dX + 0.0025, dY, dZ + 0.00499814; // the value is esimated using CCalibrateKinectExtrinsics::calibDepth()
        // 0.0025 by experience.
    }
    else
    {
        ( *pVec_ ) << 0, 0, 0;
    }
}

template< class T >
void projectWorld2Camera ( const Eigen::Matrix< T, 3, 1 >& vPt_, const Eigen::Matrix3d& mK_, Eigen::Matrix< short, 2, 1>* pVec_  )
{
// this is much faster than the function
// eiv2DPt = mK * vPt; eiv2DPt /= eiv2DPt(2);
    ( *pVec_ ) ( 0 ) = short ( mK_ ( 0, 0 ) * vPt_ ( 0 ) / vPt_ ( 2 ) + mK_ ( 0, 2 ) + 0.5 );
    ( *pVec_ ) ( 1 ) = short ( mK_ ( 1, 1 ) * vPt_ ( 1 ) / vPt_ ( 2 ) + mK_ ( 1, 2 ) + 0.5 );
}

//used by freenect depth images
template < class T >
T depthInMeters ( int nX_, int nY_, const Mat& cvmDepth_, const Mat_< T >& mPara_ = Mat_< T >(), const int nMethodType_ = 0 )
{
    int nRawDepth = rawDepth <int> ( nX_, nY_, cvmDepth_ );
    T tDepth;

    switch ( nMethodType_ )
    {
    case 0:
        tDepth = T ( rawDepthToMetersLinear< T > ( nRawDepth, mPara_ ) );
        break;
    case 1:
        tDepth = T ( rawDepthToMetersTanh< T > ( nRawDepth, mPara_ ) );
        break;
    default:
        tDepth = T ( rawDepthToMetersLinear< T > ( nRawDepth, mPara_ ) );
    }

    return tDepth;
}

template < class T >
Eigen::Matrix< T, 2, 1 > distortPoint ( const Eigen::Matrix< T, 2, 1 >& eivUndistorted_, const Mat_< T >& cvmK_, const Mat_< T >& cvmInvK_, const Mat_< T >& cvmDistCoeffs_ )
{
    double xu = eivUndistorted_ ( 0 );
    double yu = eivUndistorted_ ( 1 );
    double xun = cvmInvK_ ( 0, 0 ) * xu + cvmInvK_ ( 0, 1 ) * yu + cvmInvK_ ( 0, 2 );
    double yun = cvmInvK_ ( 1, 0 ) * xu + cvmInvK_ ( 1, 1 ) * yu + cvmInvK_ ( 1, 2 );
    double x2 = xun * xun;
    double y2 = yun * yun;
    double xy = xun * yun;
    double r2 = x2 + y2;
    double r4 = r2 * r2;
    double r6 = r4 * r2;
    double k1 = cvmDistCoeffs_ ( 0 );
    double k2 = cvmDistCoeffs_ ( 1 );
    double k3 = cvmDistCoeffs_ ( 2 );
    double k4 = cvmDistCoeffs_ ( 3 );
    double k5 = cvmDistCoeffs_ ( 4 );
    double dRadialDistortion ( 1.0 + k1 * r2 + k2 * r4 + k5 * r6 );
    double dTangentialDistortionX = ( 2 * k3 * xy ) + ( k4 * ( r2 + 2 * x2 ) );
    double dTangentialDistortionY = ( k3 * ( r2 + 2 * y2 ) ) + ( 2 * k4 * xy );
    double xdn = ( xun * dRadialDistortion ) + dTangentialDistortionX;
    double ydn = ( yun * dRadialDistortion ) + dTangentialDistortionY;
    double xd = cvmK_ ( 0, 0 ) * xdn + cvmK_ ( 0, 1 ) * ydn + cvmK_ ( 0, 2 );
    double yd = cvmK_ ( 1, 0 ) * xdn + cvmK_ ( 1, 1 ) * ydn + cvmK_ ( 1, 2 );
    Vector2d distorted ( xd, yd );
    return distorted;
}

template < class T >
void map4UndistortImage ( const Vector2i& eivImageSize_, const Mat_< T >& cvmK_, const Mat_< T >& cvmInvK_, const Mat_< T >& cvmDistCoeffs_, Mat* pMapXY )
{
    pMapXY->create ( eivImageSize_ ( 1 ), eivImageSize_ ( 0 ), CV_16SC2 );
    short* pData = ( short* ) pMapXY->data;
    Mat_<short> mapX, mapY;
//    mapX = Mat_<float> ( cvmImage_.size() );
//    mapY = Mat_<float> ( cvmImage_.size() );
    int nIdx = 0;

    for ( int y = 0; y < eivImageSize_ ( 1 ); ++y )
    {
        for ( int x = 0; x < eivImageSize_ ( 0 ); ++x )
        {
            Eigen::Matrix< T, 2, 1> undistorted ( x, y );
            Eigen::Matrix< T, 2, 1> distorted = distortPoint< T > ( undistorted, cvmK_, cvmInvK_, cvmDistCoeffs_ );
            pData [nIdx  ] = short ( distorted ( 0 ) + 0.5 );
            pData [nIdx+1] = short ( distorted ( 1 ) + 0.5 );
            nIdx += 2;
            //mapX[y][x] = ( float ) distorted ( 0 );
            //mapY[y][x] = ( float ) distorted ( 1 );
        }
    }

    return;
}

template < class T >
void undistortImage ( const Mat& cvmImage_,  const Mat_< T >& cvmK_, const Mat_< T >& cvmInvK_, const Mat_< T >& cvmDistCoeffs_, Mat* pUndistorted_ )
{
    //CHECK( cvmImage_.size() == pUndistorted_->size(), "the size of all images must be the same. \n" );
    Mat mapXY ( cvmImage_.size(), CV_16SC2 );
    short* pData = ( short* ) mapXY.data;
//    Mat_<float> mapX, mapY;
//    mapX = Mat_<float> ( cvmImage_.size() );
//    mapY = Mat_<float> ( cvmImage_.size() );
    int nIdx = 0;

    for ( int y = 0; y < cvmImage_.rows; ++y )
    {
        for ( int x = 0; x < cvmImage_.cols; ++x )
        {
            Eigen::Matrix< T, 2, 1> undistorted ( x, y );
            Eigen::Matrix< T, 2, 1> distorted = distortPoint< T > ( undistorted, cvmK_, cvmInvK_, cvmDistCoeffs_ );
            pData [nIdx  ] = short ( distorted ( 0 ) + 0.5 );
            pData [nIdx+1] = short ( distorted ( 1 ) + 0.5 );
            nIdx += 2;
            //mapX[y][x] = ( float ) distorted ( 0 );
            //mapY[y][x] = ( float ) distorted ( 1 );
        }
    }

//    cout << " undistortImage() " << endl << flush;
    cv::remap ( cvmImage_, *pUndistorted_, mapXY, Mat(), cv::INTER_NEAREST, cv::BORDER_CONSTANT );
//	cout << " after undistortImage() " << endl << flush;
    return;
}

template< class T >
T absoluteOrientation ( Eigen::MatrixXd& A_, Eigen::MatrixXd&  B_, bool bEstimateScale_, Eigen::Matrix< T, 3, 3>* pR_, Eigen::Matrix< T , 3, 1 >* pT_, double* pdScale_ )
{
// main references: http://www.mathworks.com/matlabcentral/fileexchange/22422-absolute-orientation
    CHECK ( 	A_.rows() == 3, " absoluteOrientation() requires the input matrix A_ is a 3 x N matrix. " );
    CHECK ( 	B_.rows() == 3, " absoluteOrientation() requires the input matrix B_ is a 3 x N matrix. " );
    CHECK ( 	A_.cols() == B_.cols(), " absoluteOrientation() requires the columns of input matrix A_ and B_ are equal. " );


    //Compute the centroid of each point set
    Vector3d eivCentroidA, eivCentroidB;

    for ( int nC = 0; nC < A_.cols(); nC++ )
    {
        eivCentroidA += A_.col ( nC );
        eivCentroidB += B_.col ( nC );
    }

    eivCentroidA /= A_.cols();
    eivCentroidB /= A_.cols();
    //PRINT( eivCentroidA );
    //PRINT( eivCentroidB );

    //Remove the centroid
    Eigen::MatrixXd An ( 3, A_.cols() ), Bn ( 3, A_.cols() );

    for ( int nC = 0; nC < A_.cols(); nC++ )
    {
        An.col ( nC ) = A_.col ( nC ) - eivCentroidA;
        Bn.col ( nC ) = B_.col ( nC ) - eivCentroidB;
    }

    //PRINT( An );
    //PRINT( Bn );

    //Compute the quaternions
    Eigen::Matrix4d M, Ma, Mb;

    for ( int nC = 0; nC < A_.cols(); nC++ )
    {
        //pure imaginary Shortcuts
        Vector4d a, b;
        a ( 1 ) = An ( 0, nC );
        a ( 2 ) = An ( 1, nC );
        a ( 3 ) = An ( 2, nC );
        b ( 1 ) = Bn ( 0, nC );
        b ( 2 ) = Bn ( 1, nC );
        b ( 3 ) = Bn ( 2, nC );
        //cross products
        Ma << a ( 0 ), -a ( 1 ), -a ( 2 ), -a ( 3 ),
           a ( 1 ),  a ( 0 ),  a ( 3 ), -a ( 2 ),
           a ( 2 ), -a ( 3 ),  a ( 0 ),  a ( 1 ),
           a ( 3 ),  a ( 2 ), -a ( 1 ),  a ( 0 );
        Mb << b ( 0 ), -b ( 1 ), -b ( 2 ), -b ( 3 ),
           b ( 1 ),  b ( 0 ), -b ( 3 ),  b ( 2 ),
           b ( 2 ),  b ( 3 ),  b ( 0 ), -b ( 1 ),
           b ( 3 ), -b ( 2 ),  b ( 1 ),  b ( 0 );
        //Add up
        M += Ma.transpose() * Mb;
    }

    Eigen::EigenSolver <Matrix4d> eigensolver ( M );

    Eigen::Matrix< std::complex< double >, 4, 1 > v = eigensolver.eigenvalues();

    //find the largest eigenvalue;
    double dLargest = -1000000;
    int n;

    for ( int i = 0; i < 4; i++ )
    {
        if ( dLargest < v ( i ).real() )
        {
            dLargest = v ( i ).real();
            n = i;
        }
    }

    //PRINT( dLargest );
    //PRINT( n );

    Vector4d e;
    e << eigensolver.eigenvectors().col ( n ) ( 0 ).real(),
      eigensolver.eigenvectors().col ( n ) ( 1 ).real(),
      eigensolver.eigenvectors().col ( n ) ( 2 ).real(),
      eigensolver.eigenvectors().col ( n ) ( 3 ).real();

    //PRINT( e );

    Eigen::Matrix4d M1, M2, R;
    //Compute the rotation matrix
    M1 <<  e ( 0 ), -e ( 1 ), -e ( 2 ), -e ( 3 ),
       e ( 1 ), e ( 0 ), e ( 3 ), -e ( 2 ),
       e ( 2 ), -e ( 3 ), e ( 0 ), e ( 1 ),
       e ( 3 ), e ( 2 ), -e ( 1 ), e ( 0 );
    M2 <<  e ( 0 ), -e ( 1 ), -e ( 2 ), -e ( 3 ),
       e ( 1 ), e ( 0 ), -e ( 3 ), e ( 2 ),
       e ( 2 ), e ( 3 ), e ( 0 ), -e ( 1 ),
       e ( 3 ), -e ( 2 ), e ( 1 ), e ( 0 );
    R = M1.transpose() * M2;
    ( *pR_ ) = R.block ( 1, 1, 3, 3 );

    //Compute the scale factor if necessary
    if ( bEstimateScale_ )
    {
        double a = 0, b = 0;

        for ( int nC = 0; nC < A_.cols(); nC++ )
        {
            a += Bn.col ( nC ).transpose() * ( *pR_ ) * An.col ( nC );
            b += Bn.col ( nC ).transpose() * Bn.col ( nC );
        }

        //PRINT( a );
        //PRINT( b );
        ( *pdScale_ ) = b / a;
    }
    else
    {
        ( *pdScale_ ) = 1;
    }


    //Compute the final translation
    ( *pT_ ) = eivCentroidB - ( *pdScale_ ) * ( *pR_ ) * eivCentroidA;

    //Compute the residual error
    double dE = 0;
    Vector3d eivE;

    for ( int nC = 0; nC < A_.cols(); nC++ )
    {
        eivE = B_.col ( nC ) - ( ( *pdScale_ ) * ( *pR_ ) * A_.col ( nC ) + ( *pT_ ) );
        dE += eivE.norm();
    }

    return dE / A_.cols();
}

template< class T >
void filterDepth ( const double& dThreshould_, const Mat_ < T >& cvmDepth_, Mat_< T >* pcvmDepthNew_ )
{
	//PRINT( dThreshould_ );
    pcvmDepthNew_->create ( cvmDepth_.size() );

    for ( int y = 0; y < cvmDepth_.rows; y++ )
        for ( int x = 0; x < cvmDepth_.cols; x++ )
        {
            pcvmDepthNew_->template at< T > ( y, x ) = 0;

            if ( x == 0 || x == cvmDepth_.cols - 1 || y == 0 || y == cvmDepth_.rows - 1 )
            {
                continue;
            }

            T c = cvmDepth_.template at< T > ( y, x   );
            T cl = cvmDepth_.template at< T > ( y, x - 1 );

            if ( fabs ( c - cl ) < dThreshould_ )
            {
                //PRINT( fabs( c-cl ) );
                T cr = cvmDepth_.template at< T > ( y, x + 1 );

                if ( fabs ( c - cr ) < dThreshould_ )
                {
                    T cu = cvmDepth_.template at< T > ( y - 1, x );

                    if ( fabs ( c - cu ) < dThreshould_ )
                    {
                        T cb = cvmDepth_.template at< T > ( y + 1, x );

                        if ( fabs ( c - cb ) < dThreshould_ )
                        {
                            T cul = cvmDepth_.template at< T > ( y - 1, x - 1 );

                            if ( fabs ( c - cul ) < dThreshould_ )
                            {
                                T cur = cvmDepth_.template at< T > ( y - 1, x + 1 );

                                if ( fabs ( c - cur ) < dThreshould_ )
                                {
                                    T cbl = cvmDepth_.template at< T > ( y + 1, x - 1 );

                                    if ( fabs ( c - cbl ) < dThreshould_ )
                                    {
                                        T cbr = cvmDepth_.template at< T > ( y + 1, x + 1 );

                                        if ( fabs ( c - cbr ) < dThreshould_ )
                                        {
                                            pcvmDepthNew_ ->template at< T > ( y, x ) = c;
                                            //PRINT( y );
                                            //PRINT( x );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    return;
}

template< class T >
T FindShiTomasiScoreAtPoint ( cv::Mat& img_, const int& nHalfBoxSize_ , const int& nX_, const int& nY_ )
{
    T dXX = 0;
    T dYY = 0;
    T dXY = 0;

    int nStartX = nX_ - nHalfBoxSize_;
    int nEndX   = nX_ + nHalfBoxSize_;
    int nStartY = nY_ - nHalfBoxSize_;
    int nEndY   = nY_ + nHalfBoxSize_;

    for ( int r = nStartY; r <= nEndY; r++ )
        for ( int c = nStartX; c <= nEndX; c++ )
        {
            T dx = img_.at< unsigned char > ( r, c + 1 ) - img_.at< unsigned char > ( r, c - 1 );
            T dy = img_.at< unsigned char > ( r + 1, c ) - img_.at< unsigned char > ( r - 1, c );
            dXX += dx * dx;
            dYY += dy * dy;
            dXY += dx * dy;
        }

    int nPixels = ( 2 * nHalfBoxSize_ + 1 ) * ( 2 * nHalfBoxSize_ + 1 );
    dXX = dXX / ( 2.0 * nPixels );
    dYY = dYY / ( 2.0 * nPixels );
    dXY = dXY / ( 2.0 * nPixels );
    // Find and return smaller eigenvalue:
    return 0.5 * ( dXX + dYY - sqrt ( ( dXX + dYY ) * ( dXX + dYY ) - 4 * ( dXX * dYY - dXY * dXY ) ) );
};

template< class T1, class T2>
void convert2DisparityDomain(const cv::Mat_<T1>& cvDepth_, cv::Mat_<T2>* pcvDisparity_)
{
    const T1* pInputDepth = (T1*)cvDepth_.data;
    T2* pOutputDisparity = (T2*)pcvDisparity_->data;
    for ( unsigned int y = 0; y < cvDepth_.rows; y++ )
    {
        for ( unsigned int x = 0; x < cvDepth_.cols; x++ )
        {
            *pOutputDisparity++ = 1./(*pInputDepth++);
        }
    }
    return;
}

template< class T1, class T2>
void convert2DepthDomain(const cv::Mat_<T1>& cvDepth_, cv::Mat_<T2>* pcvDisparity_)
{
    const T1* pInputDepth = (T1*)cvDepth_.data;
    T2* pOutputDisparity = (T2*)pcvDisparity_->data;
    for ( unsigned int y = 0; y < cvDepth_.rows; y++ )
    {
        for ( unsigned int x = 0; x < cvDepth_.cols; x++ )
        {
            *pOutputDisparity++ = (T2)(1./(*pInputDepth++)+.5);
        }
    }
    return;
}

template< class T > 
void bilateralFiltering( const cv::Mat_<T>& cvmSrc_, double dSigmaSpace_, double dSigmaRange_, cv::Mat_<T>* pcvmDst_)
{
    unsigned int uSize = (unsigned int)(dSigmaSpace_+.5)*2;
    cv::Mat_<T> cmSpaceKernel(uSize,uSize);
    
    
    return;
}



template< class T >
void gaussianKernel( double dSigmaSpace, unsigned int& uSize_, cv::Mat_<T>* pcvmKernel_ )
{
    
}


//template< class T >
//Matrix< T, 3, 3 > skewSymmetric( const Matrix< T, 3, 1>& eivVec_ )
//{
//	Matrix< T, 3, 3 > eimMat;
//	/*0*/                       eimMat(0,1) =  -eivVec_(2); eimMat(0,2) =  eivVec_(1);
//	eimMat(1,0) =   eivVec_(2); /*0*/                       eimMat(1,2) = -eivVec_(0);
//	eimMat(2,0) =  -eivVec_(1); eimMat(2,1) =   eivVec_(0); /*0*/
//	return eimMat;
//}
/*
template< class T >
Matrix< T, 3,3> fundamental(const Matrix< T, 3, 3 >& eimK1_, const Matrix< T, 3, 3 >& eimK2_, const Matrix< T, 3,3>& eimR_, const Matrix< T, 3,1 >& eivT_, Mat_< T >* pcvmDepthNew_ )
{
// compute fundamental matrix that the first camera is on classic pose and the second is on R and T pose, the internal
// parameters of first camera is K1, and the second is K2
// reference Multiple view geometry on computer vision page 244.
//  F = K2^{-T}RK^{T} * skew( K R^{T} t );
	Matrix< T, 3, 3> eimF = eimK2_.inverse().eval().transpose() * eimR_ * eimK1_.transpose() * skewSymmetric( eimK1_ * eimR_.transpose() * eivT_ );
	return eimF;
}
*/
}//utility
}//btl

#endif
