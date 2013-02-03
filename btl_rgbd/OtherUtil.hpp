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



namespace btl{ namespace other{
	template <class T>
	void increase(const T nCycle_, T* pnIdx_ ){
		++*pnIdx_;
		*pnIdx_ = *pnIdx_ < nCycle_? *pnIdx_: *pnIdx_-nCycle_;
	}
	template <class T>
	void decrease(const T nCycle_, T* pnIdx_ ){
		--*pnIdx_;
		*pnIdx_ = *pnIdx_ < 0?       *pnIdx_+nCycle_: *pnIdx_;
	}
}//other
}//btl


namespace btl
{
namespace utility
{


#ifdef  INFO
	// based on boost stringize.hpp
	#define PRINT( a ) std::cout << BOOST_PP_STRINGIZE( a ) << " = " << std::endl << (a) << std::flush << std::endl;
	#define PRINTSTR( a ) std::cout << a << std::endl << std::flush;
#else
	#define PRINT( a ) 
	#define PRINTSTR( a ) 
#endif//INFO

#define SMALL 1e-50 // a small value
#define BTL_DOUBLE_MAX 10e20
	enum tp_coordinate_convention { BTL_GL, BTL_CV };
	//exception based on boost
	typedef boost::error_info<struct tag_my_info, std::string> CErrorInfo;
	struct CError: virtual boost::exception, virtual std::exception { };
#define THROW(what)\
	{\
	btl::utility::CError cE;\
	cE << btl::utility::CErrorInfo ( what );\
	throw cE;\
	}
	//exception from btl2
	struct CException : public std::runtime_error
	{
		CException(const std::string& str) : std::runtime_error(str) {}
	};
#define BTL_THROW(what) {throw btl::utility::CException(what);}
	//ASSERT condition to be true; other wise throw
#define CHECK( AssertCondition_, Otherwise_) \
	if ((AssertCondition_) != true)\
	BTL_THROW( Otherwise_ );
	//THROW( Otherwise_ );
	//if condition happen then throw
#define BTL_ERROR( ErrorCondition_, ErrorMessage_ ) CHECK( !(ErrorCondition_), ErrorMessage_) 
#define BTL_ASSERT CHECK
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

    for ( typename std::map< T1, T2 >::const_iterator constItr = mp.begin(); constItr != mp.end(); ++constItr )
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


template< class T >
void getNeighbourIdxCylinder(const unsigned short& usRows, const unsigned short& usCols, const T& i, std::vector< T >* pNeighbours_ )
{
	// get the neighbor 1d index in a cylindrical coordinate system
	int a = usRows*usCols;
	BTL_ASSERT(i>=0 && i<a,"btl::utility::getNeighbourIdx() i is out of range");

	pNeighbours_->clear();
	pNeighbours_->push_back(i);
	T r = i/usCols;
	T c = i%usCols;
	T nL= c==0?        i-1 +usCols : i-1;	
	T nR= c==usCols-1? i+1 -usCols : i+1;
	pNeighbours_->push_back(nL);
	pNeighbours_->push_back(nR);

	if(r>0)//get up
	{
		T nU= i-usCols;
		pNeighbours_->push_back(nU);
		T nUL= nU%usCols == 0? nU-1 +usCols: nU-1;
		pNeighbours_->push_back(nUL);
		T nUR= nU%usCols == usCols-1? nU+1 -usCols : nU+1;
		pNeighbours_->push_back(nUR);
	}
	else if(r==usRows-1)//correspond to polar region
	{
		T t = r*usCols;
		for( T n=0; n<usCols; n++)
			pNeighbours_->push_back(t+n);
	}
	if(r<usRows-1)//get down
	{
		T nD= i+usCols;
		pNeighbours_->push_back(nD);
		T nDL= nD%usCols == 0? nD-1 +usCols: nD-1;
		pNeighbours_->push_back(nDL);
		T nDR= nD%usCols == usCols-1? nD+1 -usCols : nD+1;
		pNeighbours_->push_back(nDR);
	}

	return;
}

}//utility
}//btl

#endif
