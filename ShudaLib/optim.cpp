#include "optim.hpp"
#include <algorithm>
#include <math.h>
//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////
namespace shuda
{

COptim::COptim()
{
	cout << "COptim() ";
	// for multi-dimensional search
	m_nMdAlg = GRADIENTDESCENDENT;
	m_nMaxMdIter = 200;
	m_MdTol = 1.0e-8;
	m_MinMdGrad = 0.0;

	// for line search
	m_nLnAlg = GOLDEN;
	m_nMaxLnIter = 50;
	m_LnTol = 1.0e-8;

	// for gradient calculation using finite difference
	m_nIter = 0;
	m_Cost = 0;

	m_LnAlg_List[0] = "Golden Section";

	m_MdAlg_List[0] = "Conjugate Gradient";
	m_MdAlg_List[1] = "Direction Sets";
	m_MdAlg_List[2] = "Gradient Descendent";
}

COptim::~COptim()
{

}

bool COptim::Go()
{
	cout << "Go() ";
	if (isOK())
	{
		
		switch(m_nMdAlg)
		{
		case (CONJUGATE):
			return ConjugateGradient(m_X);

		case (DIRECTIONSETS):
			return DirectionSets(m_X);

		case (GRADIENTDESCENDENT):
			return GradientDescendent(m_X);
	
		default:
			return GradientDescendent(m_X);
		}
	}
	else
	{
		return false;
	}
}
bool COptim::isOK()
{
	cout << "isOK() ";
	m_X.create(2,1);
	m_X.at<double>(0,0) = 100;
	m_X.at<double>(1,0) = 7;
 	
	m_vDelta.create(m_X.size());
	m_vDelta.at<double>(0,0) = .0001;
	m_vDelta.at<double>(1,0) = .0001;

	m_vdCosts.clear();
	m_vXs.clear();
	m_vGs.clear();

	return true;
}

void COptim::Display()
{
	if( m_nIter > 0 )
	{
	PRINT( m_nIter );
	PRINT( m_vdCosts[ m_nIter-1 ] );
	PRINT( m_vXs[ m_nIter-1 ] );
	PRINT( m_vGs[ m_nIter-1 ] );
	}
}

// defining the function to be minimized.
double COptim::Func(const Mat_<double>& X)
{
	CHECK( 1 == X.cols, "Optim::Func() X must be a column vector.\n" ); 
	// Rosenbrock's Function
	// f(X) = X0^2 + X1^2	
	double X0 = X.at<double>(0,0)-12.;
	double X1 = X.at<double>(1,0)-24.;
	double X02= X0 * X0;
	double X12= X1 * X1;

	return ( X02 + X12/2. );
}

void COptim::dFunc(const Mat_<double>& X, Mat_<double>& G)
{
	CHECK( 1 == X.cols, "Optim::Func() X must be a column vector.\n" ); 
	CHECK( 1 == G.cols, "Optim::Func() G must be a column vector.\n" ); 

	Mat_<double> X0 = X;

	if (X0.rows == 0)
	{
		G.release();
		return;
	}	

	CHECK( X.size() == m_vDelta.size(),  "m_vDelta is set incorrectly." );

	// calculate the gradient using finite difference
	// G(X) = F(X+dX) - F(X-dX)
	for(unsigned int i=0; i<X0.rows; i++) //X0 is column vector
	{

		X0.at<double>(i,0) = X.at<double>(i,0) + m_vDelta.at<double>(i, 0);
		G.at<double>(i,0)  = Func(X0);

		X0.at<double>(i,0) = X.at<double>(i,0) - m_vDelta.at<double>(i, 0);
		G.at<double>(i,0) -= Func(X0);

		X0.at<double>(i,0) = X.at<double>(i,0);
	}
}

bool COptim::ConjugateGradient(Mat_<double>& X)
{
	cout << "COptim::ConjugateGradient() ";
	CHECK( 1 == X.cols, "Optim::Func() X must be a column vector.\n" ); 
    double lastCost;	// cost of last optimal point
	double mag0, mag1;	// squares of magnitudes of last and current gradients
	Mat_<double> Grad0, Grad1, D;
	int i;
	double lambda;

	Grad0.create( X.size() );
	Grad1.create( X.size() );
	D.create( X.size() );
	D.zeros( X.size() );

	// initialize at the starting point
	m_nIter = 0;
	m_Cost = Func(X);

	// initialize the search direction
	dFunc(X, Grad1);

	for(m_nIter=1; m_nIter<=m_nMaxMdIter; m_nIter++)
	{
		Grad0 = -Grad1;		// downhill direction
		D += Grad0;
		lastCost = m_Cost;

		m_vdCosts.push_back( m_Cost );
		m_vXs.push_back( X );
		m_vGs.push_back( D );
		Display();

		// get the square of magnitude of grad0
		mag0 = Grad0.dot(Grad0);
	     
		// if the magnitude of the gradient vanishes, 
		// the current point is a (local) minimum
		if(mag0<=m_MinMdGrad)
		{
			return true;
		}

		LineSearch(X, D, lambda);//update m_Cost;

		// check if terminating condition has been met
		if (2.0*fabs(m_Cost-lastCost)<=m_MdTol*(fabs(m_Cost)+fabs(lastCost)+OPTIM_TINY))
		{
			return true;			

		}

		// get the gradient at the new point
		dFunc(X, Grad1);

		mag1 = Grad1.dot( Grad1 ) + Grad0.dot( Grad1 );// Polak-Ribiere
//        mag1 = Grad1.dot( Grad1 );// Fletcher-Reeves

		// update the distance in search direction
		D *= (mag1/mag0);	// memory of the previous directions
	}
	return false;
}

bool COptim::GradientDescendent(Mat_<double>& X)
{
	cout << "COptim::GradientDescendent() ";
	double lastCost;	// cost of last optimal point
	double mag;	// squares of magnitudes of last and current gradients
	Mat_<double> Grad;
	int i;
	double lambda;

	Grad.create( X.size() );
	// initialize at the starting point
	m_nIter = 0;
	m_Cost = Func(X);

	// initialize the search direction
	for(m_nIter=1; m_nIter<=m_nMaxMdIter; m_nIter++)
	{
        dFunc(X, Grad);
		lastCost = m_Cost;

		m_vdCosts.push_back( m_Cost );
		m_vXs.push_back( X );
		m_vGs.push_back( Grad );

		Display();

		// get the square of magnitude of grad0
		mag = Grad.dot(Grad);

		// if the magnitude of the gradient vanishes, 
		// the current point is a (local) minimum
		if(mag<=m_MinMdGrad)
		{
			return true;
		}

		LineSearch(X, Grad, lambda);//update m_Cost;

		// check if terminating condition has been met
		if (2.0*fabs(m_Cost-lastCost)<=m_MdTol*(fabs(m_Cost)+fabs(lastCost)+OPTIM_TINY)) //ZEPS
		{
			return true;
		}
	}
	return false;
}

bool COptim::DirectionSets(Mat_<double>& X)
{
	cout << "COptim::DirectionSets() ";

	int ibig;
	double del, Cost0, Cost1, lambda;

	Mat_<double> X0, X1, Dn, D;

	X0.create( X.size() );
	X1.create( X.size() );
	Dn.create( X.rows, X.rows ); //Dn is a square matrix;
	D .create( X.size() );

	// initialize at the starting point
	Dn = Mat_<double>::eye(  X.rows, X.rows );// set to identity matrix

	m_nIter = 0;
	m_Cost = Func(X);

	// save the initial point
	X0 = X.clone();
	for(m_nIter=1; m_nIter<=m_nMaxMdIter; m_nIter++)
	{
		Cost0 = m_Cost;

		ibig = 0;
		del = 0.0;

		m_vdCosts.push_back( m_Cost );
		m_vXs.push_back( X );
		m_vGs.push_back( D );
		PRINT( Dn );
		Display();
		// loop over each direction
		for(int i=0;i<Dn.cols;i++)
		{
			D = Dn.col(i).clone();
			Cost1 = m_Cost;
			LineSearch(X, D, lambda);

			if (fabs(Cost1-m_Cost)>del)//find the biggest cost falling int direction sets
			{
				del = fabs(Cost1-m_Cost);
				ibig=i;
			}
		}

		// check if terminating condition has been met
		if (2.0*fabs(Cost0-m_Cost) <= m_MdTol*(fabs(Cost0)+fabs(m_Cost)+OPTIM_TINY))
		{
			cout << "converges. \n";
			return true;
		}

		X1 = (X*2)-X0;	// extrapolated point
		D = X -X0;		// average direction moved
		X0 = X;			// old start point

		Cost1 = Func(X1);

		if (Cost1<Cost0) 
		{
			if (2.0*(Cost0-2.0*m_Cost+Cost1)*(Cost0-m_Cost-del)*(Cost0-m_Cost-del)<del*(Cost0-Cost1)*(Cost0-Cost1))
			{
				LineSearch(X, D, lambda);
				//set the corresponding column as D;
				Dn.at<double>( 0, ibig) = D.at<double>( 0, 0 ) *lambda ;
				Dn.at<double>( 1, ibig) = D.at<double>( 1, 0 ) *lambda ;
				Dn.at<double>( 2, ibig) = D.at<double>( 2, 0 ) *lambda ;
			}
		}
	}
	return false;
}

bool COptim::LineSearch(Mat_<double>& X, const Mat_<double>& D, double& lambda)
{
	// perform the line search
	switch(m_nLnAlg)
	{
	case (GOLDEN):
		return GoldenSection(X, D, lambda);

	default:
		return GoldenSection(X, D, lambda);
	}
}

bool COptim::GoldenSection(Mat_<double>& X, const Mat_<double>& D, double& lambda)
{
	double a, b, c, fa, fb, fc;
	double u, v, w, x, fu, fv, fw, fx;
	double p, q, r, tol1, tol2, d, e, t, xm;

	// bracket the minimum b between a and c
	a = 0.0;
	b = 5e-1;
	MnBrak(a, b, c, fa, fb, fc, X, D);

	// make sure a, b and c are in ascending order
	if (a>c)
	{
		// swap a and c
		t = a; a = c; c = t;
	}

	// initialize the movements of the last 2 steps to zeros
	e = d = 0.0;

	// initialize v, w and x to b
	v = w = x = b;
	fv = fw = fx = fb;

	for(int i=0; i<m_nMaxLnIter; i++)
	{
		// find the mid point xm between a and c
		xm = 0.5*(a+c);

		// find the fractional tolerance
		tol2 = 2.0*(tol1 = m_LnTol*fabs(x)+OPTIM_ZEPS);

		// check if terminating condition has been met
		if (fabs(x-xm)<=(tol2-0.5*(c-a)))
		{
			X += (D*x);
			m_Cost = fx;
			lambda = x;

			return true;
		}

		if (fabs(e)>tol1)
		{
			// construct a trial parabolic fit
			r = (x-w)*(fx-fv);
			q = (x-v)*(fx-fw);
			p = (x-v)*q-(x-w)*r;
			q = 2.0*(q-r);
			if (q>0.0)
			{
				p = -p;
			}
			q = fabs(q);

			t = e; e = d;

			// ensure the change is less than half the step before last,
			// and the new point lies in (a,c)
			if (fabs(p)>=fabs(0.5*q*t) || p<=q*(a-x) || p>=q*(c-x))
			{
				// reject the parabolic fit and use golden section step instead
				d = OPTIM_GOLD*(e = (x>=xm ? a-x : c-x));
			}
			else
			{
				// take the parabolic step
				d = p/q;
				u = x+d;

				if (u-a<tol2 || c-u<tol2)
				{
					// too close to the bracket end-points,
#ifdef __linux__
					d = copysign(tol1, xm-x);
#else if _WIN32 || _WIN64
					d = _copysign(tol1, xm-x);
#endif
				}
			}
		}
		else
		{
			// take the golden section step
			d = OPTIM_GOLD*(e = (x>=xm ? a-x: c-x));
		}

		// ensure the new point is not too close to the current point
#ifdef __linux__
		u = (fabs(d)>=tol1 ? x+d : x + copysign(tol1, d));
#else if _WIN32 || _WIN64
		u = (fabs(d)>=tol1 ? x+d : x + _copysign(tol1, d));
#endif
		
		fu = Func(X+(D*u));
		if (fu<=fx)
		{
			if (u>=x)
			{
				a = x;
			}
			else
			{
				c = x;
			}

			// v <= w <= x <= u
			v = w; w = x; x = u;
			fv = fw; fw = fx; fx = fu;
		}
		else
		{
			if (u<x)
			{
				a = u;
			}
			else
			{
				c = u;
			}

			if (fu<=fw || w==x)
			{
				// v <= w <= u
				v = w; w = u;
				fv = fw; fw = fu;
			}
			else if (fu<=fv || v==x || v==w)
			{
				// v <= u
				v = u;
				fv = fu;
			}
		}
	}

	// max no. of iterations has been reached
	X += (D*x);
	m_Cost = fx;
	lambda = x;

	return false;
}

void COptim::MnBrak(double& a, double& b, double& c, double& fa, double& fb, double& fc, const Mat_<double>& X, const Mat_<double>& D)
{
	double r, q, u, ulim, fu;
	int count = 10;

	fa = m_Cost;//Func(X+(D*a));
	fb = Func(X+(D*b));

	// make sure f(a) >= f(b)
	while(fb>fa && count--)
	{
		b = a + 0.5*(b-a);
		fb = Func(X+(D*b));
	}

	if (fb>fa)
	{
		// swap the roles of a and b
		c = a; a = b; b = c;
		fc = fa; fa = fb; fb = fc;
	}

	// form the first guess of c
	c = b + OPTIM_SRAT*(b-a);
	fc = Func(X+(D*c));

	while(fb>fc)
	{
		// find the minimum at u by parabolic interpolation
		r = (b-a)*(fb-fc);
		q = (b-c)*(fb-fa);
#ifdef __linux__
		u = b - ((b-c)*q-(b-a)*r)/(2.0* copysign(std::max(fabs(q-r), OPTIM_TINY), q-r));
#else if _WIN32 || _WIN64
		u = b - ((b-c)*q-(b-a)*r)/(2.0* _copysign(std::max(fabs(q-r), OPTIM_TINY), q-r));
#endif
		

		ulim = b + OPTIM_SLTD*(c-b);

		if ((b-u)*(u-c)>0.0)	
		{
			// u is between b and c
			fu = Func(X+(D*u));

			if (fu<fc)
			{
				// got a minumim between b and c
				a = b; b = u;
				fa = fb; fb = fu;
				return;
			}
			else if (fu>fb)
			{
				// got a minimum between a and u
				c = u;
				fc = fu;
				return;
			}

			// parabolic fit doesn't help,
			// used default magnification
			u = c + OPTIM_SRAT*(c-b);
			fu = Func(X+(D*u));
		}
		else if ((c-u)*(u-ulim)>0.0)
		{
			// u is between c and its allowed limit
			fu = Func(X+(D*u));

			if (fu<fc)
			{
				// (a,b,c) <= (c,u,u+OPTIM_SRAT*(u-c))
				a = c;
				b = u;
				c = u + OPTIM_SRAT*(u-c);

				fa = fc;
				fb = fu;
				fc = Func(X+(D*c));

				continue;
			}
		}
		else if ((u-ulim)*(ulim-c)>=0.0)
		{
			// u is outside its allowed limit,
			// set u to its limit instead
			u = ulim;
			fu = Func(X+(D*u));
		}
		else
		{
			// use default magnification
			u = c + OPTIM_SRAT*(c-b);
			fu = Func(X+(D*u));
		}

		// (a,b,c) <= (b,c,u)
		a = b; b = c; c = u;
		fa = fb; fb = fc; fc = fu;
	}
}

}//namespace shuda
