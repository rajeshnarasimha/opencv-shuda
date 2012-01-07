#ifndef BTL_OTHER_UTILITY_HELPER
#define BTL_OTHER_UTILITY_HELPER

//helpers based-on stl and boost

#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <complex>
#include <string>
#include <boost/exception/all.hpp>
#include <boost/preprocessor/stringize.hpp>


namespace btl
{
namespace utility
{

#define SMALL 1e-50 // a small value
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
 

// for print
template <class T>
std::ostream& operator << ( std::ostream& os, const std::vector< T > & v )
{
	os << "[";

	for ( typename std::vector< T >::const_iterator constItr = v.begin(); constItr != v.end(); ++constItr )
	{
		os << " " << ( *constItr ) << " ";
	}

	os << "]";
	return os;
}

template <class T1, class T2>
std::ostream& operator << ( std::ostream& os, const std::map< T1, T2 > & mp )
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
std::ostream& operator << ( std::ostream& os, const std::list< T >& l_ )
{
    os << "[";
    for ( typename std::list< T >::const_iterator cit_List = l_.begin(); cit_List != l_.end(); cit_List++ )
    {
        os << " " << *cit_List << " ";
    }
    os << "]";
    return os;
}

//calculate vector<> difference for testing
template< class T>
T matNormL1 ( const std::vector< T >& vMat1_, const std::vector< T >& vMat2_ )
{
	T tAccumDiff = 0;
	for(unsigned int i=0; i < vMat1_.size(); i++ )
	{
		T tDiff = vMat1_[i] - vMat2_[i];
		tDiff = tDiff >= 0? tDiff:-tDiff;
		tAccumDiff += tDiff;
	}
	return tAccumDiff;
}

}
}

#endif
