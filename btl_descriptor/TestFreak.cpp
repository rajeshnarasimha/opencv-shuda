#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

#include "TestFreak.h"

__device__ short2 operator + (const short2 s2O1_, const short2 s2O2_);
__device__ short2 operator - (const short2 s2O1_, const short2 s2O2_);
__device__ float2 operator * (const float fO1_, const short2 s2O2_);
__device__ short2 operator * (const short sO1_, const short2 s2O2_);
__device__ __host__ float2 operator + (const float2 f2O1_, const float2 f2O2_);
__device__ __host__ float2 operator - (const float2 f2O1_, const float2 f2O2_);
__device__  short2 convert2s2(const float2 f2O1_);


#define PI 3.14159265358979323846
int _nImgRows;
int _nImgCols;
// the total # of octaves. 4 
int _nFREAK_OCTAVE;
// the scaling constant. sizeCst
float _fFREAK_SIZE_Cst;
// the smallest diameter key point in pixels 9
int _nFREAK_SMALLEST_KP_SIZE;
// the total # of pattern points 43
int _nFREAK_NB_POINTS;
// the total # of scales in scale space 64
int _nFREAK_NB_SCALES;
// the total # of pairs of patches for computing orientation  45
int _nFREAK_NB_ORIENPAIRS;
// the total # of sampling for orientation 256
int _nFREAK_NB_ORIENTATION;
// the total # of pairs of patches for computing descriptor 512
int _nFREAK_NB_PAIRS;
// log2
double _dFREAK_LOG2;
// the aray holding pairs of patches for computing orientation
__constant__ int4 _an4OrientationPair[ 45 ];//1x45 i,j,weight_x, weight_y
// the array holding pairs of patches default
__constant__ uchar2 _auc2DescriptorPair[ 512 ];//1x512 i,j


namespace test{

void loadGlobalConstants( int nImgRows_, int nImgCols_, int nFREAK_OCTAVE_, float fSizeCst_, int nFREAK_SMALLEST_KP_SIZE_, int nFREAK_NB_POINTS_, int nFREAK_NB_SCALES_,
						 int nFREAK_NB_ORIENPAIRS_,  int nFREAK_NB_ORIENTATION_,int nFREAK_NB_PAIRS_, double dFREAK_LOG2_)
{
	_nImgRows = nImgRows_;
	_nImgCols = nImgCols_;
	_nFREAK_OCTAVE =  nFREAK_OCTAVE_;
	_fFREAK_SIZE_Cst = fSizeCst_;
	_nFREAK_SMALLEST_KP_SIZE = nFREAK_SMALLEST_KP_SIZE_;
	_nFREAK_NB_POINTS = nFREAK_NB_POINTS_;
	_nFREAK_NB_SCALES = nFREAK_NB_SCALES_;
	_nFREAK_NB_ORIENPAIRS = nFREAK_NB_ORIENPAIRS_;
	_nFREAK_NB_ORIENTATION = nFREAK_NB_ORIENTATION_;
	_nFREAK_NB_PAIRS = nFREAK_NB_PAIRS_;
	_dFREAK_LOG2 = dFREAK_LOG2_;
	return;
}



void loadOrientationAndDescriptorPair( int4 an4OrientationPair[ 45 ], uchar2 auc2DescriptorPair[ 512 ] ){
	for(int i=0; i < 45; i++) {
		_an4OrientationPair[i]= an4OrientationPair[i];
	}

	for(int i=0; i < 512; i++){
		_auc2DescriptorPair[i] = auc2DescriptorPair[i];
	}
	return;
}



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
	cv::Mat_<uchar> _cvgmImg;
	cv::Mat_<int>   _cvgmImgInt;

	cv::Mat_<float> _cvgmKeyPoints;
	cv::Mat_<short> _cvgmKpScaleIdx;

	cv::Mat_<float3> _cvgmPatternLookup; //1x FREAK_NB_SCALES*FREAK_NB_ORIENTATION*FREAK_NB_POINTS, x,y,sigma
	cv::Mat_<int>	   _cvgmPatternSize; //1x64 

	cv::Mat_<uchar> _cvgmFreakDescriptor;

	/*cv::Mat_<uchar> _cvgmFreakPointPerKp;
	cv::Mat_<uchar> _cvgmFreakPointPerKp2;
	cv::Mat_<float> _cvgmSigma;
	cv::Mat_<int>   _cvgmInt;
	cv::Mat_<int>   _cvgmTheta;
	int _nIdx;*/
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
			const float3 f3FreakPoint = _cvgmPatternLookup.ptr<float3>()[sScale_*_nFREAK_NB_ORIENTATION*_nFREAK_NB_POINTS + sThetaIdx_*_nFREAK_NB_POINTS + sPointIdx_];
			const float xf = f3FreakPoint.x+fKpX_;
			const float yf = f3FreakPoint.y+fKpY_;
			const int x = int(xf); 
			const int y = int(yf);

			// get the sigma:
			const float radius = f3FreakPoint.z; //sigma
			/*_cvgmSigma.ptr<float>(_nIdx)[sPointIdx_] = yf;
			_cvgmInt.ptr<int>(_nIdx)[sPointIdx_] = y;*/


			// calculate output:
			if( radius < 0.5 ) {
				// interpolation multipliers:
				const int r_x = int ((xf-x)*1024);
				const int r_y = int ((yf-y)*1024);
				const int r_x_1 = (1024-r_x);
				const int r_y_1 = (1024-r_y);
				uchar* ptr = _cvgmImg.data+x+y*_nImgCols;
				int ret_val;
				// linear interpolation:
				ret_val = (r_x_1*r_y_1* int (*ptr));
				ptr++;
				ret_val += (r_x*r_y_1* int (*ptr));
				ptr += _nImgCols;
				ret_val += (r_x*r_y* int (*ptr));
				ptr--;
				ret_val += (r_x_1*r_y* int(*ptr));
				//return the rounded mean
				ret_val += 2 * 1024 * 1024;

				return uchar(ret_val / (4 * 1024 * 1024));
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

				return uchar(ret_val);//static_cast<uchar>(ret_val);
			}
	}//meanIntensity()

	__device__ __forceinline__ void  mainFunc() {
		
		for (int nIdx =0; nIdx < _cvgmKeyPoints.cols; nIdx++) {
			//_nIdx = nIdx;

			float fKpX = _cvgmKeyPoints.ptr<float>(X_ROW)[nIdx]; //get key point locations
			float fKpY = _cvgmKeyPoints.ptr<float>(Y_ROW)[nIdx];
			short sScale = _cvgmKpScaleIdx.ptr<short>()[nIdx]; //get scale idx
			float fResponse = _cvgmKeyPoints.ptr<float>(HESSIAN_ROW)[nIdx]; //get hessian score

			if( fResponse < 0.f ){ // if hessian score is negative, ignore current
				continue;
			}

			__shared__ uchar shaucPointIntensity[43]; //hold the intensity of the Freak points

			for (int nThr =0; nThr <64; nThr++) {
				if( nThr < _nFREAK_NB_POINTS ){ //get intensity for all freak points
					shaucPointIntensity[ nThr ] = meanIntensity( fKpX, fKpY, sScale, 0, nThr );
					//_cvgmFreakPointPerKp.ptr<uchar>(nIdx)[nThr] =  shaucPointIntensity[ nThr ];
				}
			}

			//__syncthreads();//1
			__shared__ int shnDirection0; 
			__shared__ int shnDirection1; 

			for (int nThr =0; nThr <64; nThr++) {
				if(nThr == 0){ shnDirection0 = 0; shnDirection1 = 0; }

				if( nThr < _nFREAK_NB_ORIENPAIRS ) {
					//iterate through the orientation pairs
					const int delta = shaucPointIntensity[ _an4OrientationPair[nThr].x ] - shaucPointIntensity[ _an4OrientationPair[nThr].y ];
					int d0 = delta*(_an4OrientationPair[nThr].z)/2048; //weight_dx
					shnDirection0 += d0;
					int d1 = delta*(_an4OrientationPair[nThr].w)/2048; //weight_dy
					shnDirection1 += d1;
				}
			}
			//__syncthreads();//2
			__shared__ int shnThetaIdx;		

			for (int nThr =0; nThr <64; nThr++) {
				if( nThr == 0 ){
					float fAngle = (180./PI)*::atan2((double)shnDirection1,(double)shnDirection0);//estimate orientation
					_cvgmKeyPoints.ptr<float>(ANGLE_ROW)[nIdx] = fAngle;
					shnThetaIdx = int(_nFREAK_NB_ORIENTATION*fAngle/360.f+0.5f);
					if( shnThetaIdx < 0 )
						shnThetaIdx += _nFREAK_NB_ORIENTATION;

					if( shnThetaIdx >= _nFREAK_NB_ORIENTATION )
						shnThetaIdx -= _nFREAK_NB_ORIENTATION;

					//_cvgmTheta.ptr<int>(nIdx)[0] = shnThetaIdx;
				}
			}
			//__syncthreads();//3

			// extract descriptor at the computed orientation
			for (int nThr =0; nThr <64; nThr++) {
				if( nThr < _nFREAK_NB_POINTS ) {
					shaucPointIntensity[nThr] = meanIntensity( fKpX, fKpY, sScale, shnThetaIdx, nThr );
					//_cvgmFreakPointPerKp2.ptr<uchar>(nIdx)[nThr] =  shaucPointIntensity[ nThr ];
				}
			}
			//__syncthreads();//4



			// extracting descriptor preserving the order of SSE version
			for (int nThr =0; nThr <64; nThr++) {
				if( nThr < 64 ){
					_cvgmFreakDescriptor.ptr<uchar>(nIdx)[nThr] = devGetFreakDescriptor(shaucPointIntensity, nThr);
				}
			}//for nThr
		}//for nIdx

		return;
	}
};//struct computeFreakDescriptor

__global__ void kernelComputeFreakDescriptor( SComputeFreakDescriptor sCFD_ ){
	sCFD_.mainFunc();
}

//
void cudaComputeFreakDescriptor(const cv::gpu::GpuMat& cvgmImg_, 
								const cv::gpu::GpuMat& cvgmImgInt_, 
									  cv::gpu::GpuMat& cvgmKeyPoint_, //output angle
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
	cvgmImg_.download(sCFD._cvgmImg);
	cvgmImgInt_.download(sCFD._cvgmImgInt);
	cvgmKeyPoint_.download(sCFD._cvgmKeyPoints);
	cvgmKpScaleIdx_.download(sCFD._cvgmKpScaleIdx);

	cvgmPatternLookup_.download(sCFD._cvgmPatternLookup); //1x FREAK_NB_SCALES*FREAK_NB_ORIENTATION*FREAK_NB_POINTS, x,y,sigma
	cvgmPatternSize_.download(sCFD._cvgmPatternSize); //1x64 

	pcvgmFreakDescriptor_->download(sCFD._cvgmFreakDescriptor);
	/*pcvgmFreakPointPerKp_->download(sCFD._cvgmFreakPointPerKp);
	pcvgmFreakPointPerKp2_->download(sCFD._cvgmFreakPointPerKp2);
	pcvgmSigma_->download(sCFD._cvgmSigma);
	pcvgmInt_->download(sCFD._cvgmInt);
	pcvgmTheta_->download(sCFD._cvgmTheta);*/
	kernelComputeFreakDescriptor (sCFD);

	pcvgmFreakDescriptor_->upload(sCFD._cvgmFreakDescriptor);
	cvgmKeyPoint_.upload(sCFD._cvgmKeyPoints);
	/*pcvgmFreakPointPerKp_->upload(sCFD._cvgmFreakPointPerKp);
	pcvgmFreakPointPerKp2_->upload(sCFD._cvgmFreakPointPerKp2);
	pcvgmSigma_->upload(sCFD._cvgmSigma);
	pcvgmInt_->upload(sCFD._cvgmInt);
	pcvgmTheta_->upload(sCFD._cvgmTheta);*/
	return;
}

}//test