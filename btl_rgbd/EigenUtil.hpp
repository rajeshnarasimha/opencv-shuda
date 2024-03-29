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
Eigen::Matrix< T , 4, 4 > setModelViewGLfromRTCV ( const Eigen::Matrix< T, 3, 3 >& mR_, const Eigen::Matrix< T, 3, 1 >& vT_ )
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
template< class T >
Eigen::Matrix< T , 4, 4 > setModelViewGLfromRCCV ( const Eigen::Matrix< T, 3, 3 >& mR_, const Eigen::Matrix< T, 3, 1 >& vC_ )
{
	Eigen::Matrix< T, 3,1> eivT = -mR.transpose()*vC_;
	return setModelViewGLfromRTCV(mR_,vC_);
}
template< class T1, class T2 >
void unprojectCamera2World ( const int& nX_, const int& nY_, const unsigned short& nD_, const Eigen::Matrix< T1, 3, 3 >& mK_, Eigen::Matrix< T2, 3, 1 >* pVec_ )
{
	//the pixel coordinate is defined w.r.t. opencv camera reference, which is defined as x-left, y-downward and z-forward. It's
	//a right hand system.
	//when rendering the point using opengl's camera reference which is defined as x-left, y-upward and z-backward. the
	//	glVertex3d ( Pt(0), -Pt(1), -Pt(2) );
	if ( nD_ > 400 ) {
		T2 dZ = nD_ / 1000.; //convert to meter
		T2 dX = ( nX_ - mK_ ( 0, 2 ) ) / mK_ ( 0, 0 ) * dZ;
		T2 dY = ( nY_ - mK_ ( 1, 2 ) ) / mK_ ( 1, 1 ) * dZ;
		( *pVec_ ) << dX + 0.0025, dY, dZ + 0.00499814; // the value is esimated using CCalibrateKinectExtrinsics::calibDepth()
		// 0.0025 by experience.
	}
	else {
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

template< class T > /*Eigen::Matrix<float,-1,-1,0,-1,-1> = Eigen::MatrixXf*/
T absoluteOrientation ( Eigen::Matrix<T,-1,-1,0,-1,-1> & eimA_, Eigen::Matrix<T,-1,-1,0,-1,-1>&  eimB_, bool bEstimateScale_, Eigen::Matrix< T, 3, 3>* pR_, Eigen::Matrix< T , 3, 1 >* pT_, T* pdScale_ ){
	// A is Ref B is Cur
	// eimB_ = R * eimA_ + T;
	// main references: http://www.mathworks.com/matlabcentral/fileexchange/22422-absolute-orientation
	CHECK ( 	eimA_.rows() == 3, " absoluteOrientation() requires the input matrix A_ is a 3 x N matrix. " );
	CHECK ( 	eimB_.rows() == 3, " absoluteOrientation() requires the input matrix B_ is a 3 x N matrix. " );
	CHECK ( 	eimA_.cols() == eimB_.cols(), " absoluteOrientation() requires the columns of input matrix A_ and B_ are equal. " );

	//Compute the centroid of each point set
	
	Eigen::Matrix<T,3,1> eivCentroidA(0,0,0), eivCentroidB(0,0,0); //Matrix<float,3,1,0,3,1> = Vector3f
	for ( int nC = 0; nC < eimA_.cols(); nC++ ){
		eivCentroidA += eimA_.col ( nC );
		eivCentroidB += eimB_.col ( nC );
	}
	eivCentroidA /= eimA_.cols();
	eivCentroidB /= eimA_.cols();
	//PRINT( eivCentroidA );
	//PRINT( eivCentroidB );

	//Remove the centroid
	/*Eigen::MatrixXd */
	Eigen::Matrix<T,-1,-1,0,-1,-1> An ( 3, eimA_.cols() ), Bn ( 3, eimA_.cols() );
	for ( int nC = 0; nC < eimA_.cols(); nC++ ){
		An.col ( nC ) = eimA_.col ( nC ) - eivCentroidA;
		Bn.col ( nC ) = eimB_.col ( nC ) - eivCentroidB;
	}

	//PRINT( An );
	//PRINT( Bn );

	//Compute the quaternions
	Eigen::Matrix<T,4,4> M; M.setZero();
	Eigen::Matrix<T,4,4> Ma, Mb;
	for ( int nC = 0; nC < eimA_.cols(); nC++ ){
		//pure imaginary Shortcuts
		/*Eigen::Vector4d*/
		Eigen::Matrix<T,4,1> a(0,0,0,0), b(0,0,0,0);
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

	Eigen::EigenSolver <Eigen::Matrix<T,4,4>> eigensolver ( M );
	Eigen::Matrix< std::complex< T >, 4, 1 > v = eigensolver.eigenvalues();

	//find the largest eigenvalue;
	float dLargest = -1000000;
	int n;

	for ( int i = 0; i < 4; i++ ) {
		if ( dLargest < v ( i ).real() ) {
			dLargest = v ( i ).real();
			n = i;
		}
	}

	//PRINT( dLargest );
	//PRINT( n );

	Eigen::Matrix<T,4,1> e;
	e << eigensolver.eigenvectors().col ( n ) ( 0 ).real(),
		eigensolver.eigenvectors().col ( n ) ( 1 ).real(),
		eigensolver.eigenvectors().col ( n ) ( 2 ).real(),
		eigensolver.eigenvectors().col ( n ) ( 3 ).real();

	//PRINT( e );

	Eigen::Matrix<T,4,4>M1, M2, R;
	//Compute the rotation matrix
	M1 <<  e ( 0 ), -e ( 1 ), -e ( 2 ), -e ( 3 ),
		e ( 1 ),  e ( 0 ),  e ( 3 ), -e ( 2 ),
		e ( 2 ), -e ( 3 ),  e ( 0 ),  e ( 1 ),
		e ( 3 ),  e ( 2 ), -e ( 1 ),  e ( 0 );
	M2 <<  e ( 0 ), -e ( 1 ), -e ( 2 ), -e ( 3 ),
		e ( 1 ),  e ( 0 ), -e ( 3 ),  e ( 2 ),
		e ( 2 ),  e ( 3 ),  e ( 0 ), -e ( 1 ),
		e ( 3 ), -e ( 2 ),  e ( 1 ),  e ( 0 );
	R = M1.transpose() * M2;
	( *pR_ ) = R.block ( 1, 1, 3, 3 );

	//Compute the scale factor if necessary
	if ( bEstimateScale_ ){
		T a = 0, b = 0;
		for ( int nC = 0; nC < eimA_.cols(); nC++ ) {
			a += Bn.col ( nC ).transpose() * ( *pR_ ) * An.col ( nC );
			b += Bn.col ( nC ).transpose() * Bn.col ( nC );
		}
		//PRINT( a );
		//PRINT( b );
		( *pdScale_ ) = b / a;
	}
	else{
		( *pdScale_ ) = 1;
	}
	//Compute the final translation
	( *pT_ ) = eivCentroidB - ( *pdScale_ ) * ( *pR_ ) * eivCentroidA;

	//Compute the residual error
	T dE = 0;
	Eigen::Matrix<T,3,1> eivE;

	for ( int nC = 0; nC < eimA_.cols(); nC++ ) {
		eivE = eimB_.col ( nC ) - ( ( *pdScale_ ) * ( *pR_ ) * eimA_.col ( nC ) + ( *pT_ ) );
		dE += eivE.norm();
	}

	return dE / eimA_.cols();
}

template< class T, int ROW, int COL >
T matNormL1 ( const Eigen::Matrix< T, ROW, COL >& eimMat1_, const Eigen::Matrix< T, ROW, COL >& eimMat2_ )
{
	Eigen::Matrix< T, ROW, COL > eimTmp = eimMat1_ - eimMat2_;
	Eigen::Matrix< T, ROW, COL > eimAbs = eimTmp.cwiseAbs();
	return (T) eimAbs.sum();
}

template< class T >
void setSkew( T x_, T y_, T z_, Eigen::Matrix< T, 3,3 >* peimMat_){
	*peimMat_ << 0, -z_, y_, z_, 0, -x_, -y_, x_, 0 ;
}

template< class T >
void setRotMatrixUsingExponentialMap( T x_, T y_, T z_, Eigen::Matrix< T, 3,3 >* peimR_ ){
	//http://opencv.itseez.com/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html?highlight=rodrigues#void Rodrigues(InputArray src, OutputArray dst, OutputArray jacobian)
	T theta = sqrt( x_*x_ + y_*y_ + z_*z_ );
	if(	theta < std::numeric_limits<T>::epsilon() ){
		*peimR_ = Eigen::Matrix< T, 3,3 >::Identity();
		return;
	}
	T sinTheta = sin(theta);
	T cosTheta = cos(theta);
	Eigen::Matrix< T, 3,3 > eimSkew; 
	setSkew< T >(x_/theta,y_/theta,z_/theta,&eimSkew);
	*peimR_ = Eigen::Matrix< T, 3,3 >::Identity() + eimSkew*sinTheta + eimSkew*eimSkew*(1-cosTheta);
}

}//utility
}//btl
#endif
