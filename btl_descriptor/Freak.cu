
#include <thrust/sort.h>

#include <opencv2/gpu/gpumat.hpp>
#include <opencv2/gpu/device/common.hpp>

/*
#include "opencv2/gpu/device/limits.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"
#include "opencv2/gpu/device/utility.hpp"
#include "opencv2/gpu/device/functional.hpp"
#include "opencv2/gpu/device/filters.hpp"*/

#include "Freak.cuh"

	__constant__ int _nImgRows;
	__constant__ int _nImgCols;
	// the total # of octaves. 4 
    __constant__ int _nFREAK_OCTAVE;
    // the scaling constant. sizeCst
    __constant__ float _fFREAK_SIZE_Cst;
	// the smallest diameter key point in pixels 9
    __constant__ int _nFREAK_SMALLEST_KP_SIZE;
	// the total # of pattern points 43
	__constant__ int _nFREAK_NB_POINTS;
	// the total # of scales in scale space 64
	__constant__ int _nFREAK_NB_SCALES;
	// the total # of pairs of patches for computing orientation  45
	__constant__ int _nFREAK_NB_ORIENPAIRS;
	// the total # of sampling for orientation 256
	__constant__ int _nFREAK_NB_ORIENTATION;
	// the total # of pairs of patches for computing descriptor 512
	__constant__ int _nFREAK_NB_PAIRS;
	// log2
	__constant__ double _dFREAK_LOG2;
		// the aray holding pairs of patches for computing orientation
	__constant__ int4 _an4OrientationPair[ 45 ];//1x45 i,j,weight_x, weight_y
	// the array holding pairs of patches default
	__constant__ uchar2 _auc2DescriptorPair[ 512 ];//1x512 i,j

	__device__ unsigned int _devuTotal;

namespace btl { namespace device
{
namespace freak
{
//#define FREAK_NB_ORIENTATION 256
//#define FREAK_NB_POINTS 43
#define PI 3.14159265358979323846

	void loadGlobalConstants( int nFREAK_OCTAVE_, float fSizeCst_, int nFREAK_SMALLEST_KP_SIZE_, int nFREAK_NB_POINTS_, int nFREAK_NB_SCALES_,
							  int nFREAK_NB_ORIENPAIRS_,  int nFREAK_NB_ORIENTATION_,int nFREAK_NB_PAIRS_, double dFREAK_LOG2_)
    {
        cudaSafeCall( cudaMemcpyToSymbol(_nFREAK_OCTAVE,			&nFREAK_OCTAVE_,			sizeof(nFREAK_OCTAVE_)) );
        cudaSafeCall( cudaMemcpyToSymbol(_fFREAK_SIZE_Cst,			&fSizeCst_,					sizeof(fSizeCst_)) );
        cudaSafeCall( cudaMemcpyToSymbol(_nFREAK_SMALLEST_KP_SIZE,	&nFREAK_SMALLEST_KP_SIZE_,	sizeof(nFREAK_SMALLEST_KP_SIZE_)) );
		cudaSafeCall( cudaMemcpyToSymbol(_nFREAK_NB_POINTS,			&nFREAK_NB_POINTS_,			sizeof(nFREAK_NB_POINTS_)) );
        cudaSafeCall( cudaMemcpyToSymbol(_nFREAK_NB_SCALES,			&nFREAK_NB_SCALES_,			sizeof(nFREAK_NB_SCALES_)) );
        cudaSafeCall( cudaMemcpyToSymbol(_nFREAK_NB_ORIENPAIRS,		&nFREAK_NB_ORIENPAIRS_,		sizeof(nFREAK_NB_ORIENPAIRS_)) );
        cudaSafeCall( cudaMemcpyToSymbol(_nFREAK_NB_ORIENTATION,	&nFREAK_NB_ORIENTATION_,	sizeof(nFREAK_NB_ORIENTATION_)) );
        cudaSafeCall( cudaMemcpyToSymbol(_nFREAK_NB_PAIRS,			&nFREAK_NB_PAIRS_,			sizeof(nFREAK_NB_PAIRS_)) );
		cudaSafeCall( cudaMemcpyToSymbol(_dFREAK_LOG2,				&dFREAK_LOG2_,				sizeof(dFREAK_LOG2_)) );
		return;
    }

	void loadGlobalConstantsImgResolution( int nImgRows_, int nImgCols_)
    {
		cudaSafeCall( cudaMemcpyToSymbol(_nImgRows,					&nImgRows_,					sizeof(nImgRows_)) );
		cudaSafeCall( cudaMemcpyToSymbol(_nImgCols,					&nImgCols_,					sizeof(nImgCols_)) );
    }



	void loadOrientationAndDescriptorPair( int4 an4OrientationPair[ 45 ], uchar2 auc2DescriptorPair[ 512 ] ){
		cudaSafeCall( cudaMemcpyToSymbol(_an4OrientationPair,			an4OrientationPair,			sizeof(int4)*45 ) );
		cudaSafeCall( cudaMemcpyToSymbol(_auc2DescriptorPair,				auc2DescriptorPair,			sizeof(uchar2)*512 ) );
		return;
	}



	struct SFreakPattern{

		double _dScaleStep;
		double _dPatternScale;
		double* _radius;
		double* _sigma;
		int* _n;

		cv::gpu::DevMem2D_<float3> _cvgmPatternLookup;
		cv::gpu::DevMem2D_<int>    _cvgmPatternSize;

		// 64x256 threeads and 1 blocks
		__device__ __forceinline__ void  operator () () {
			int scaleIdx = threadIdx.x; // for

			__shared__ int patternSizes[64];
			patternSizes[scaleIdx] = 0; // proper initialization
			__shared__ int sn[8];
			if(scaleIdx<8) sn[scaleIdx]=_n[scaleIdx];
			__shared__ double sRadius[8];
			if(scaleIdx<8)  sRadius[scaleIdx]=_radius[scaleIdx];
			__shared__ double sSigma[8];
			if(scaleIdx<8)  sSigma[scaleIdx]=_sigma[scaleIdx];

			__syncthreads();

			double scalingFactor, alpha, beta, theta = 0;
			scalingFactor = ::pow(_dScaleStep,scaleIdx); //scale of the pattern, scaleStep ^ scaleIdx
			int orientationIdx = threadIdx.y + blockIdx.y * blockDim.y; // for
			if( orientationIdx >= _nFREAK_NB_ORIENTATION) return;
			theta = double(orientationIdx)* 2*PI /double(_nFREAK_NB_ORIENTATION); // orientation of the pattern

			int pointIdx = 0;

			for( int i = 0; i < 8; ++i ) { // 8 rings
				for( int k = 0 ; k < sn[i]; ++k ) { // 6,6,6,6, 6,6,6,1 = 43 in total
					beta = PI/sn[i]* (i%2); // orientation offset so that groups of points on each circles are staggered
					alpha = double(k)* 2*PI/double(sn[i])+beta+theta;
					// add the point to the look-up table
					float3& point = _cvgmPatternLookup.ptr(0)[scaleIdx*_nFREAK_NB_ORIENTATION*_nFREAK_NB_POINTS + orientationIdx*_nFREAK_NB_POINTS + pointIdx];
					point.x = __double2float_rn( sRadius[i] *::cos(alpha) * scalingFactor * _dPatternScale );
					point.y = __double2float_rn( sRadius[i] *::sin(alpha) * scalingFactor * _dPatternScale);
					point.z = __double2float_rn( sSigma[i] * scalingFactor * _dPatternScale);//sigma
					++pointIdx;
				}
				// adapt the sizeList if necessary
				const int sizeMax = __double2int_rn(ceil((sRadius[i]+sSigma[i])*scalingFactor*_dPatternScale)) + 1;
				if( patternSizes[scaleIdx] < sizeMax )
					patternSizes[scaleIdx] = sizeMax;	
			}
			int& nPatternSize = _cvgmPatternSize.ptr(0)[scaleIdx];
			if( nPatternSize < patternSizes[scaleIdx] )
				nPatternSize = patternSizes[scaleIdx];
		}//operator()

	};

	__global__ void kernelBuildPattern( SFreakPattern sFP_ ){
		sFP_();
	}

	void cudaBuildFreakPattern(const double& dPatternScale_, const double& dScaleStep_, int nFREAK_NB_ORIENTATION_, const int n[8], const double radius[8],const double sigma[8], cv::gpu::GpuMat* pcvgmPatternLookup_, cv::gpu::GpuMat* pcvgmPatternSize_){

		SFreakPattern sFP;
		sFP._dScaleStep = dScaleStep_;

		cudaSafeCall( cudaMalloc ( (void**) &(sFP._radius), sizeof(double)*8 ) );
		cudaSafeCall( cudaMemcpy ( sFP._radius, radius,  sizeof(double)*8, cudaMemcpyHostToDevice ) ); 
		cudaSafeCall( cudaMalloc ( (void**) &(sFP._sigma), sizeof(double)*8 ) );
		cudaSafeCall( cudaMemcpy ( sFP._sigma,  sigma,  sizeof(double)*8, cudaMemcpyHostToDevice ) ); 
		cudaSafeCall( cudaMalloc ( (void**) &(sFP._n), sizeof(int)*8 ) );
		cudaSafeCall( cudaMemcpy ( sFP._n, n, sizeof(int)*8, cudaMemcpyHostToDevice ) ); 

		sFP._cvgmPatternLookup = *pcvgmPatternLookup_;
		pcvgmPatternSize_->setTo(0);
		sFP._cvgmPatternSize = *pcvgmPatternSize_;
		sFP._dPatternScale = dPatternScale_;

		dim3 block(64,8);
		dim3 grid;
		grid.x = 1;
		grid.y = cv::gpu::divUp(nFREAK_NB_ORIENTATION_,8);

		kernelBuildPattern<<<grid,block>>>(sFP);
		cudaSafeCall( cudaGetLastError() );
		cudaSafeCall( cudaDeviceSynchronize() );
		return;
	}//


	struct SComputeScaleIndex{
		enum KeypointLayout
		{
			X_ROW = 0,
			Y_ROW,
			LAPLACIAN_ROW,
			OCTAVE_ROW,
			SIZE_ROW,
			ANGLE_ROW,
			HESSIAN_ROW,
			ROWS_COUNT
		};
	
		cv::gpu::DevMem2D_<int> _cvgmPatternSize; //input
		cv::gpu::DevMem2D_<float> _cvgmKeyPoints; //output
		cv::gpu::DevMem2D_<short> _cvgmKpScale; //output


		//  total # of keypoints and 64 threads in each block
		__device__ __forceinline__ void  normalized() {
			int nIdx = threadIdx.x + blockIdx.x * blockDim.x;

			if( nIdx >= _cvgmKeyPoints.cols ) return; 
			//Is k non-zero? If so, decrement it and continue"
			float fSize = _cvgmKeyPoints.ptr(SIZE_ROW)[nIdx];
			short sScale = ::max( (short)(log(fSize/_nFREAK_SMALLEST_KP_SIZE)*_fFREAK_SIZE_Cst+0.5) ,0); //calc the scale index w.r.t. FREAK scale samples,
			if( sScale >= _nFREAK_NB_SCALES ) // it should lie within 0 and 63 
				sScale = _nFREAK_NB_SCALES-1;

			_cvgmKpScale.ptr()[nIdx] = sScale;

			float fKpX = _cvgmKeyPoints.ptr(X_ROW)[nIdx];
			float fKpY = _cvgmKeyPoints.ptr(Y_ROW)[nIdx];

			int nSize = _cvgmPatternSize.ptr()[sScale];
			if( fKpX <= nSize || //check if the description at this specific position and scale fits inside the image
				fKpY <= nSize ||
				fKpX >= _nImgCols - nSize ||
				fKpY >= _nImgRows - nSize
			) {
				_cvgmKeyPoints.ptr(HESSIAN_ROW)[nIdx] = -1000.f;
			}
			return;
		}
		__device__ __forceinline__ void  unNormalized() {
			/*const int scIdx = max( (int)(1.0986122886681*sizeCst+0.5) ,0);
			for( size_t k = keypoints.size(); k--; ) {
				kpScaleIdx[k] = scIdx; // equivalent to the formule when the scale is normalized with a constant size of keypoints[k].size=3*SMALLEST_KP_SIZE
				if( kpScaleIdx[k] >= FREAK_NB_SCALES ) {
					kpScaleIdx[k] = FREAK_NB_SCALES-1;
				}
				if( keypoints[k].pt.x <= patternSizes[kpScaleIdx[k]] ||
					keypoints[k].pt.y <= patternSizes[kpScaleIdx[k]] ||
					keypoints[k].pt.x >= image.cols-patternSizes[kpScaleIdx[k]] ||
					keypoints[k].pt.y >= image.rows-patternSizes[kpScaleIdx[k]]	) 
				{
					keypoints.erase(kpBegin+k);
					kpScaleIdx.erase(ScaleIdxBegin+k);
				}
			}*/
			return;
		}

	};//struct 
	__global__ void kernelNormalizedComputeScaleIndex( SComputeScaleIndex sCSI_ )
	{
		sCSI_.normalized();
	}
	__global__ void kernelUnNormalizedComputeScaleIndex( SComputeScaleIndex sCSI_ )
	{
		sCSI_.unNormalized();
	}
	//float, float, short
	void cudaComputeScaleIndex(const cv::gpu::GpuMat& cvgmPatternSize_, cv::gpu::GpuMat* pcvgmKeypoint_, cv::gpu::GpuMat* pcvgmKpScale_){

		SComputeScaleIndex sCSI;
		sCSI._cvgmPatternSize = cvgmPatternSize_;
		sCSI._cvgmKeyPoints   = *pcvgmKeypoint_;
		sCSI._cvgmKpScale	  = *pcvgmKpScale_;
			 
		dim3 block(64,1);
		dim3 grid;
		grid.x = cv::gpu::divUp(pcvgmKeypoint_->cols,64);
		grid.y = 1;

		kernelNormalizedComputeScaleIndex<<<grid,block>>>(sCSI);
		cudaSafeCall( cudaGetLastError() );
		cudaSafeCall( cudaDeviceSynchronize() );

		return;
	}
	
#define TH_Y 1
	struct SComputeFreakDescriptor{
		enum KeypointLayout	{
			X_ROW = 0,
			Y_ROW,
			LAPLACIAN_ROW,
			OCTAVE_ROW,
			SIZE_ROW,
			ANGLE_ROW,
			HESSIAN_ROW,
			ROWS_COUNT
		};



		cv::gpu::DevMem2D_<uchar> _cvgmImg;
		cv::gpu::DevMem2D_<int>   _cvgmImgInt;

		cv::gpu::DevMem2D_<float> _cvgmKeyPoints;
		cv::gpu::DevMem2D_<short> _cvgmKpScaleIdx;

		cv::gpu::DevMem2D_<float3> _cvgmPatternLookup; //1x FREAK_NB_SCALES*FREAK_NB_ORIENTATION*FREAK_NB_POINTS, x,y,sigma
		cv::gpu::DevMem2D_<int>	   _cvgmPatternSize; //1x64 

		cv::gpu::DevMem2D_<uchar> _cvgmFreakDescriptor;

		/*//for debug
		cv::gpu::DevMem2D_<uchar> _cvgmFreakPointPerKp;
		cv::gpu::DevMem2D_<uchar> _cvgmFreakPointPerKp2;
		cv::gpu::DevMem2D_<float> _cvgmSigma;
		cv::gpu::DevMem2D_<int>   _cvgmInt;
		cv::gpu::DevMem2D_<int>   _cvgmTheta;*/
		//calc FreakDescriptor
		__device__ unsigned char devGetFreakDescriptor(uchar shaucPointIntensity[43], int nDescIdx_)
		{
			int nIdx = 8 * nDescIdx_; //compare 8 pairs of points, and that is 16 points in total

			unsigned char val;
			val  = (shaucPointIntensity[_auc2DescriptorPair[nIdx].x] >= shaucPointIntensity[_auc2DescriptorPair[nIdx].y]) << 0; nIdx++;
			val |= (shaucPointIntensity[_auc2DescriptorPair[nIdx].x] >= shaucPointIntensity[_auc2DescriptorPair[nIdx].y]) << 1; nIdx++;
			val |= (shaucPointIntensity[_auc2DescriptorPair[nIdx].x] >= shaucPointIntensity[_auc2DescriptorPair[nIdx].y]) << 2; nIdx++;
			val |= (shaucPointIntensity[_auc2DescriptorPair[nIdx].x] >= shaucPointIntensity[_auc2DescriptorPair[nIdx].y]) << 3; nIdx++;
			val |= (shaucPointIntensity[_auc2DescriptorPair[nIdx].x] >= shaucPointIntensity[_auc2DescriptorPair[nIdx].y]) << 4; nIdx++;
			val |= (shaucPointIntensity[_auc2DescriptorPair[nIdx].x] >= shaucPointIntensity[_auc2DescriptorPair[nIdx].y]) << 5; nIdx++;
			val |= (shaucPointIntensity[_auc2DescriptorPair[nIdx].x] >= shaucPointIntensity[_auc2DescriptorPair[nIdx].y]) << 6; nIdx++;
			val |= (shaucPointIntensity[_auc2DescriptorPair[nIdx].x] >= shaucPointIntensity[_auc2DescriptorPair[nIdx].y]) << 7; nIdx++;

			return val;
		} 


		// simply take average on a square patch, not even gaussian approx
		__device__ __forceinline__ uchar meanIntensity( const float fKpX_, const float fKpY_, 
													    const short sScale_, const short sThetaIdx_, 
													    const short sPointIdx_ ){
			const float3 f3FreakPoint = _cvgmPatternLookup.ptr()[sScale_*_nFREAK_NB_ORIENTATION*_nFREAK_NB_POINTS + sThetaIdx_*_nFREAK_NB_POINTS + sPointIdx_];
			const float xf = f3FreakPoint.x+fKpX_;
			const float yf = f3FreakPoint.y+fKpY_;
			const int x = int(xf); 
			const int y = int(yf);

			// get the sigma:
			const float radius = f3FreakPoint.z; //sigma
			/*_cvgmSigma.ptr(threadIdx.y + blockIdx.y * blockDim.y)[sPointIdx_] = yf;
			_cvgmInt.ptr(threadIdx.y + blockIdx.y * blockDim.y)[sPointIdx_] = y;*/

			// calculate output:
			if( radius < 0.5 ) {
				// interpolation multipliers:
				const int r_x = int((xf-x)*1024);
				const int r_y = int((yf-y)*1024);
				const int r_x_1 = (1024-r_x);
				const int r_y_1 = (1024-r_y);
				uchar* ptr = _cvgmImg.data+x+y*_nImgCols;
				int ret_val;
				// linear interpolation:
				ret_val = (r_x_1*r_y_1*int(*ptr));
				ptr++;
				ret_val += (r_x*r_y_1*int(*ptr));
				ptr += _nImgCols;
				ret_val += (r_x*r_y*int(*ptr));
				ptr--;
				ret_val += (r_x_1*r_y*int(*ptr));
				//return the rounded mean
				ret_val += 2 * 1024 * 1024;

				return unsigned char(ret_val / (4 * 1024 * 1024));
			}
			else{
				// expected case:

				// calculate borders
				const int x_left = int(xf-radius+0.5);
				const int y_top = int(yf-radius+0.5);
				const int x_right = int(xf+radius+1.5);//integral image is 1px wider
				const int y_bottom = int(yf+radius+1.5);//integral image is 1px higher
				int ret_val;

				ret_val =  _cvgmImgInt(y_bottom,x_right);//bottom right corner
				ret_val -= _cvgmImgInt(y_bottom,x_left);
				ret_val += _cvgmImgInt(y_top,x_left);
				ret_val -= _cvgmImgInt(y_top,x_right);
				ret_val = ret_val/( (x_right-x_left)* (y_bottom-y_top) );

				return unsigned char(ret_val);
			}
		}//meanIntensity()

		//the each row handle a feature. there are 
		__device__ __forceinline__ void  mainFunc() {
			const int nThr = threadIdx.x; //64 threads to compute one single descriptor
			const int nIdx = threadIdx.y + blockIdx.y * blockDim.y; // idx of key points

			if(nIdx >= _cvgmKeyPoints.cols) {
				__syncthreads();//0.5
				__syncthreads();//1.5
				__syncthreads();//2
				__syncthreads();//3
				__syncthreads();//4
				return;
			}

			__shared__ float fKpX[TH_Y];
			__shared__ float fKpY[TH_Y];
			__shared__ short sScale[TH_Y];
			__shared__ float fResponse[TH_Y];
			
			__shared__ int shnDirection0[TH_Y]; 
			__shared__ int shnDirection1[TH_Y]; 

			if(nThr == 0){
				fKpX[threadIdx.y] = _cvgmKeyPoints.ptr(X_ROW)[nIdx]; //get key point locations
				fKpY[threadIdx.y] = _cvgmKeyPoints.ptr(Y_ROW)[nIdx];
				sScale[threadIdx.y] = _cvgmKpScaleIdx.ptr()[nIdx]; //get scale idx
				fResponse[threadIdx.y] = _cvgmKeyPoints.ptr(HESSIAN_ROW)[nIdx]; //get hessian score
				shnDirection0[threadIdx.y] = 0; 
				shnDirection1[threadIdx.y] = 0;
			}

			__syncthreads(); //0.5

			if( fResponse[threadIdx.y] < 0.f ){ // if hessian score is negative, ignore current
				__syncthreads();//1.5
				__syncthreads();//2
				__syncthreads();//3
				__syncthreads();//4
				return;
			}
			__shared__ uchar shaucPointIntensity[TH_Y][43]; //hold the intensity of the Freak points

			if( nThr < _nFREAK_NB_POINTS ){ //get intensity for all freak points
				shaucPointIntensity[threadIdx.y][ nThr ] = meanIntensity( fKpX[threadIdx.y], fKpY[threadIdx.y], sScale[threadIdx.y], 0, nThr );
				//_cvgmFreakPointPerKp.ptr(nIdx)[nThr] = shaucPointIntensity[ nThr ]; 
			}

			__syncthreads();//1.5

			if( nThr < _nFREAK_NB_ORIENPAIRS ) {
				//iterate through the orientation pairs
				const int delta = shaucPointIntensity[ _an4OrientationPair[nThr].x ] - shaucPointIntensity[ _an4OrientationPair[nThr].y ];
				int d0 = delta*(_an4OrientationPair[nThr].z)/2048; //weight_dx
				atomicAdd( &(shnDirection0[threadIdx.y]), d0 );
				int d1 = delta*(_an4OrientationPair[nThr].w)/2048; //weight_dy
				atomicAdd( &(shnDirection1[threadIdx.y]), d1 );
			}

			__syncthreads();//2

			__shared__ int shnThetaIdx[TH_Y];			
			if( nThr == 0 ){
				float fAngle = (180./PI)*::atan2(double(shnDirection1[threadIdx.y]),double(shnDirection0[threadIdx.y]));//estimate orientation
				_cvgmKeyPoints.ptr(ANGLE_ROW)[nIdx] = fAngle;
				shnThetaIdx[threadIdx.y] = int(_nFREAK_NB_ORIENTATION*fAngle/360.f+.5f);
				if( shnThetaIdx[threadIdx.y] < 0 )
					shnThetaIdx[threadIdx.y] += _nFREAK_NB_ORIENTATION;

				if( shnThetaIdx[threadIdx.y] >= _nFREAK_NB_ORIENTATION )
					shnThetaIdx[threadIdx.y] -= _nFREAK_NB_ORIENTATION;

				atomicAdd( &_devuTotal, 1 );
				//_cvgmTheta.ptr(nIdx)[0] = shnThetaIdx;
			}
			
			__syncthreads();//3

			// extract descriptor at the computed orientation
			if( nThr < _nFREAK_NB_POINTS ) {
				shaucPointIntensity[threadIdx.y][nThr] = meanIntensity( fKpX[threadIdx.y], fKpY[threadIdx.y], sScale[threadIdx.y], shnThetaIdx[threadIdx.y], nThr );
				//_cvgmFreakPointPerKp2.ptr(nIdx)[nThr] = shaucPointIntensity[ nThr ]; 
			}

			__syncthreads();//4

			// extracting descriptor preserving the order of SSE version

			if( nThr < 64 ){
				_cvgmFreakDescriptor.ptr(nIdx)[nThr] = devGetFreakDescriptor(shaucPointIntensity[threadIdx.y], nThr);
			}
			return;
		}
	};//struct computeFreakDescriptor

	__global__ void kernelComputeFreakDescriptor( SComputeFreakDescriptor sCFD_ ){
		sCFD_.mainFunc();
	}

	//
	unsigned int cudaComputeFreakDescriptor(const cv::gpu::GpuMat& cvgmImg_, 
									const cv::gpu::GpuMat& cvgmImgInt_, 
										  cv::gpu::GpuMat& cvgmKeyPoint_, 
									const cv::gpu::GpuMat& cvgmKpScaleIdx_,
									const cv::gpu::GpuMat& cvgmPatternLookup_, //1x FREAK_NB_SCALES*FREAK_NB_ORIENTATION*FREAK_NB_POINTS, x,y,sigma
									const cv::gpu::GpuMat& cvgmPatternSize_, //1x64 
									cv::gpu::GpuMat* pcvgmFreakDescriptor_/*,
									cv::gpu::GpuMat* pcvgmFreakPointPerKp_,
									cv::gpu::GpuMat* pcvgmFreakPointPerKp2_,
									cv::gpu::GpuMat* pcvgmSigma_,
									cv::gpu::GpuMat* pcvgmInt_,
									cv::gpu::GpuMat* pcvgmTheta_*/)
	{
		SComputeFreakDescriptor sCFD;
		sCFD._cvgmImg		  = cvgmImg_;
		sCFD._cvgmImgInt	  = cvgmImgInt_;
		sCFD._cvgmKeyPoints	  = cvgmKeyPoint_;
		sCFD._cvgmKpScaleIdx  = cvgmKpScaleIdx_;

		sCFD._cvgmPatternLookup = cvgmPatternLookup_; //1x FREAK_NB_SCALES*FREAK_NB_ORIENTATION*FREAK_NB_POINTS, x,y,sigma
		sCFD._cvgmPatternSize = cvgmPatternSize_; //1x64 
		
		sCFD._cvgmFreakDescriptor = *pcvgmFreakDescriptor_;
		/*sCFD._cvgmFreakPointPerKp = *pcvgmFreakPointPerKp_;
		sCFD._cvgmFreakPointPerKp2 = *pcvgmFreakPointPerKp2_;
		sCFD._cvgmSigma = *pcvgmSigma_;
		sCFD._cvgmInt = *pcvgmInt_;
		sCFD._cvgmTheta = *pcvgmTheta_;*/

		void* pTotal;
		cudaSafeCall( cudaGetSymbolAddress(&pTotal, _devuTotal) );
		cudaSafeCall( cudaMemset(pTotal, 0, sizeof(unsigned int)) );
			 
		dim3 block(64,TH_Y);
		dim3 grid;
		grid.x = 1;
		grid.y = cv::gpu::divUp(cvgmKeyPoint_.cols,1);

		kernelComputeFreakDescriptor<<<grid,block>>>(sCFD);
		cudaSafeCall( cudaGetLastError() );
		cudaSafeCall( cudaDeviceSynchronize() );

		unsigned int uTotal;
		cudaSafeCall( cudaMemcpy(&uTotal, pTotal, sizeof(unsigned int), cudaMemcpyDeviceToHost) );

		return uTotal;
	}

}//freak
}//device
}//btl