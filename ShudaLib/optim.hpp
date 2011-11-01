#ifndef OPTIMIZATION_SHUDA 
#define OPTIMIZATION_SHUDA
/**
* @file optim.hpp
* @brief Adapted from KennethLib developed by Dr. Kenneth Wong
* This class is mainly designed for minumizing a predefined function 
* Prerequirement:
* 	- the search space must containing only one single global
* Dependency:
*   - OpenCV 2.1 C++
* @author Shuda Li<lishuda1980@gmail.com>
* @version 1.0
* @date 2011-04-04
*/


#include <Converters.hpp>
#include <math.h>
#include <vector>
using namespace std;
using namespace cv;
using namespace btl::utility;
#define OPTIM_TINY	(1.0e-20)
#define OPTIM_ZEPS	(1.0e-13)
#define OPTIM_GOLD	(0.3819660)		// golden ratio

// for bracketing a minimum
#define OPTIM_SRAT	(1.618034)		// ratio of magnification in successive interval
#define OPTIM_SLTD	(100.0)			// maximum magnification allowed

namespace shuda
{

class COptim
{

public:
	// line search algorithms
	enum COPTIM_LNALG
	{
		GOLDEN,	// golden		: golden section search with parabolic interpolation
		MAXLNALG
	};	 

	// multi-dimensional search algorithms
	enum COPTIM_MDALG
	{
		CONJUGATE,		// conjugate	: conjugate gradient search
		DIRECTIONSETS,	// powell		: direction set (powell's) method, no derivatives
		GRADIENTDESCENDENT,// gradient   : gradient descendent search
		MAXMDALG
	};

	// constructors
	COptim();

	// destructors
	virtual ~COptim();

	// for multi-dimensional search
	// set the algorithm to be used
	void SetMdAlg(int mdAlg) {m_nMdAlg = mdAlg;}
	// set the max no. of iterations
	void SetMaxMdIter(int nMax) {m_nMaxMdIter = (nMax>0 ? nMax : 0);}
	// set the fractional tolerance
	virtual void SetMdTol(double ftol) {m_MdTol = (fabs(ftol)>1.0e-20 ? fabs(ftol) : 1.0e-20);}
	// set the min. magnitude of the gradient
	virtual void SetMinMdGrad(double minGrad) {m_MinMdGrad = fabs(minGrad);}

	// for line search
	// set the algorithm to be used
	void SetLnAlg(int nLnAlg_) {m_nLnAlg = nLnAlg_;}
	// set the max no. of iterantions 
	virtual void SetMaxLnIter(int nMax) {m_nMaxLnIter = (nMax>0 ? nMax : 0);}
	// set the fractional tolerance
	virtual void SetLnTol(double ftol) {m_LnTol = (fabs(ftol)>1.0e-20 ? fabs(ftol) : 1.0e-20);}

	// for gradient calculation using finite difference
	virtual void SetDelta(int nIdx_,double dx_) 
	{
		CHECK( (nIdx_>=(int)m_vDelta.rows*m_vDelta.cols||nIdx_<0), "m_vDelta is wrong.");  
		m_vDelta.at<double>( nIdx_, 0 ) = (fabs(dx_)>1.0e-20 ? fabs(dx_) : 1.0e-20);
	}

	// retrievers
	inline const string& GetMdAlgName(int idx) const {return m_MdAlg_List[idx];}
	inline const string& GetLnAlgName(int idx) const {return m_LnAlg_List[idx];}
	inline const int& GetMdAlg() const {return m_nMdAlg;}
	inline const int& GetLnAlg() const {return m_nLnAlg;}
	inline const int& GetMaxMdIter() const {return m_nMaxMdIter;}
	inline const int& GetMaxLnIter() const {return m_nMaxLnIter;}
	inline const double& GetMdTol() const {return m_MdTol;}
	inline const double& GetLnTol() const {return m_LnTol;}
	inline const double& GetMinMdGrad() const {return m_MinMdGrad;}
	inline double GetDelta(int nIdx_) const 
	{
		CHECK( (nIdx_>=(int)m_vDelta.rows*m_vDelta.cols||nIdx_<0), "m_vDelta is wrong." );   
		return m_vDelta.at<double>( nIdx_, 0 );
	}
	inline const int& Iter() const {return m_nIter;}
	inline const double& Cost() const {return m_Cost;}
	virtual const Mat_<double>& GetX() const {return m_X;}
	virtual Mat_<double>& GetX() {return m_X;}

	// launcher
	// begin the optimization which calls isOK()
	bool Go();
	// override this to check if everything has been initialized and ready to start
	virtual bool isOK(); 
	// override this to display the current information after each iteration
	virtual void Display(); 

	// cost functions to be mininized
	// override this by the actual cost function,
	// default is the Rosenbrock's Function
	virtual double Func(const Mat_<double>& X);
	// override this by the gradient of the cost function
	// default is by finite-difference
	virtual void dFunc(const Mat_<double>& X, Mat_<double>& G);

	// serializations

protected:
	// multi-dimensional search algorithms
	virtual bool ConjugateGradient(Mat_<double>& X);
	virtual bool DirectionSets(Mat_<double>& X);
	virtual bool GradientDescendent(Mat_<double>& X);
/**
* @brief search for the local minimum along the direction D
*
* @param X is the initial location and also be the location after line searching
* @param D is the searching direction, it is constant
* @param lambda is the distance along the searching direction
*
* @return true if it converges.
*/
	virtual bool LineSearch(Mat_<double>& X, const Mat_<double>& D, double& lambda);

	// line search algorithms
	virtual bool GoldenSection(Mat_<double>& X, const Mat_<double>& D, double& lambda);

	// function for bracketing a minimum in a line search
	virtual void MnBrak(double& a, double& b, double& c, double& fa, double& fb, double& fc, const Mat_<double>& X, const Mat_<double>& D);

protected:
	// for multi-dimensional search
	string m_MdAlg_List[MAXMDALG];
	int m_nMdAlg;		// algorithm to be used
	int	m_nMaxMdIter;	// maximium number of iterations
	double m_MdTol;		// fractional tolerance
	double m_MinMdGrad;	// minimum magitude of the gradient

	// for line search
	string m_LnAlg_List[MAXLNALG];
	int m_nLnAlg;		// algorithm to be used
	int m_nMaxLnIter;	// max no. of iterations
	double m_LnTol;		// fractional tolerance

	// for gradient calculation using finite difference
	Mat_<double> m_vDelta;

	// output
	int m_nIter;		// current number of iterations
	double m_Cost;		// current cost

	// parameters
	Mat_<double> m_X;
	vector<double> m_vdCosts;
	vector< Mat_<double> > m_vXs;
	vector< Mat_<double> > m_vGs;
};

}//namespace shuda
#endif
