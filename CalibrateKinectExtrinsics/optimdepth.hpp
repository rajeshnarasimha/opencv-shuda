#ifndef OPTIMDEPTH_SHUDA 
#define OPTIMDEPTH_SHUDA
#include <btl/extra/VideoSource/optim.hpp>


class COptimDepth: public shuda::COptim
{
public:
	void set( const vector<double>& vRealDepth_, const vector<int>& vRawDepth_ )
	{
		_vRealDepth = vRealDepth_;
		_vRawDepth  = vRawDepth_;
	}

	// cost functions to be mininized
	// override this by the actual cost function,
	// default is the Rosenbrock's Function
	virtual double Func(const Mat_<double>& X);
	virtual bool   isOK();

private:
	vector<double> _vRealDepth;
	vector<int>    _vRawDepth;
};
/*
// to optimize the depth computation using NI
class COptimDepthOpenNI: public shuda::COptim
{
public:
	void set( const vector< Vector4d >& vRealDepth_, const vector< Vector4d >& vRawDepth_ )
	{
		_vRealDepth = vRealDepth_;
		_vRawDepth  = vRawDepth_;
	}

	// cost functions to be mininized
	// override this by the actual cost function,
	// default is the Rosenbrock's Function
	virtual double Func(const Mat_<double>& X);
	virtual bool   isOK();

private:
	vector< Vector4d > _vRealDepth;
	vector< Vector4d > _vRawDepth;
};
*/

#endif
