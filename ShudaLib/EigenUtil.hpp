#ifndef BTL_Eigen_UTILITY_HEADER
#define BTL_Eigen_UTILITY_HEADER

//eigen-based helpers
#include "OtherUtil.hpp"
#include <Eigen/Dense>

namespace btl
{
namespace utility
{

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

template< class T >
T absoluteOrientation ( Eigen::MatrixXd& A_, Eigen::MatrixXd&  B_, bool bEstimateScale_, Eigen::Matrix< T, 3, 3>* pR_, Eigen::Matrix< T , 3, 1 >* pT_, double* pdScale_ )
{
	// main references: http://www.mathworks.com/matlabcentral/fileexchange/22422-absolute-orientation
	CHECK ( 	A_.rows() == 3, " absoluteOrientation() requires the input matrix A_ is a 3 x N matrix. " );
	CHECK ( 	B_.rows() == 3, " absoluteOrientation() requires the input matrix B_ is a 3 x N matrix. " );
	CHECK ( 	A_.cols() == B_.cols(), " absoluteOrientation() requires the columns of input matrix A_ and B_ are equal. " );


	//Compute the centroid of each point set
	Eigen::Vector3d eivCentroidA, eivCentroidB;

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
		Eigen::Vector4d a, b;
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

	Eigen::EigenSolver <Eigen::Matrix4d> eigensolver ( M );

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

	Eigen::Vector4d e;
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
	Eigen::Vector3d eivE;

	for ( int nC = 0; nC < A_.cols(); nC++ )
	{
		eivE = B_.col ( nC ) - ( ( *pdScale_ ) * ( *pR_ ) * A_.col ( nC ) + ( *pT_ ) );
		dE += eivE.norm();
	}

	return dE / A_.cols();
}

template< class T, int ROW, int COL >
T matNormL1 ( const Eigen::Matrix< T, ROW, COL >& eimMat1_, const Eigen::Matrix< T, ROW, COL >& eimMat2_ )
{
	Eigen::Matrix< T, ROW, COL > eimTmp = eimMat1_ - eimMat2_;
	Eigen::Matrix< T, ROW, COL > eimAbs = eimTmp.cwiseAbs();
	return (T) eimAbs.sum();
}

}//utility
}//btl
#endif
