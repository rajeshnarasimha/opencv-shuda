#include "optimdepth.hpp"
#include <btl/Utility/Converters.hpp>

using namespace btl::utility;

bool COptimDepth::isOK()
{
	cout << "COptimDepth::isOK() ";

	CHECK( !_vRealDepth.empty(), "COptimDepth:: RealDepth can not be empty." );
	CHECK( _vRealDepth.size() == _vRawDepth.size(), "COptimDepth:: must have equal # of elements" );
/*
    // for linear model
	m_X.create(2,1);
	m_X.at<double>(0,0) = -0.0030711016;
	m_X.at<double>(1,0) =  3.3309495161;
 	
	m_vDelta.create(m_X.size());
	m_vDelta.at<double>(0,0) = .00001;
	m_vDelta.at<double>(1,0) = .00001;
*/
	m_X.create(3,1);
	m_X.at<double>(0,0) = 1.1863;
	m_X.at<double>(1,0) = 2842.5;
	m_X.at<double>(2,0) = 0.1236;

 	
	m_vDelta.create(m_X.size());
	m_vDelta.at<double>(0,0) = .0001;
	m_vDelta.at<double>(1,0) = 1.0;
	m_vDelta.at<double>(2,0) = .00001;

	m_vdCosts.clear();
	m_vXs.clear();
	m_vGs.clear();

	return true;
}

double COptimDepth::Func(const Mat_<double>& X)
{
	//cout << "COptimDepth::Func() ";

	//CHECK( X.rows == 2, "COptimDepth::Func() X must have 2 rows\n" );
	//return COptim::Func( X );
	
	double dE = 0.0;
	for(unsigned int i = 0; i < _vRealDepth.size(); i++ )
	{
		//dE += abs( _vRealDepth[i] - 1.0/( _vRawDepth[i]*X.at<double>(0,0) + X.at<double>(1,0) ) ); 
		dE += abs( _vRealDepth[i] - rawDepthToMetersTanh< double >( _vRawDepth[i], X ) );
	}
	return dE;
}
/*
bool COptimDepthOpenNI::isOK()
{
	cout << "COptimDepthOpenNI::isOK() ";

	CHECK( !_vRealDepth.empty(), "COptimDepthOpenNI:: RealDepth can not be empty." );
	CHECK( _vRealDepth.size() == _vRawDepth.size(), "COptimDepthOpenNI:: must have equal # of elements" );
	

 	
	m_vDelta.create(m_X.size());
	m_vDelta.at<double>(0,0) = .0001;
	m_vDelta.at<double>(1,0) = .0001;
	m_vDelta.at<double>(2,0) = .0001;

	m_vdCosts.clear();
	m_vXs.clear();
	m_vGs.clear();

	return true;
}

double COptimDepthOpenNI::Func(const Mat_<double>& X)
{
	//cout << "COptimDepth::Func() ";

	//CHECK( X.rows == 2, "COptimDepth::Func() X must have 2 rows\n" );
	//return COptim::Func( X );
	
	double dE = 0.0;
	for(unsigned int i = 0; i < _vRealDepth.size(); i++ )
	{
		dE += _vRealDepth[i].dot( _vRawDepth[i] );
	}
	return dE;
}
*/
