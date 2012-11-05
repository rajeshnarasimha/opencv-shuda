
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#endif

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/devmem2d.hpp>
#include <math_constants.h>
#include "cv/common.hpp" //copied from opencv
#include "pcl/limits.hpp"
#include "pcl/device.hpp"
#include "pcl/vector_math.hpp"

namespace btl{ namespace device
{
using namespace pcl::device;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct STSDF{
	enum{
        MAX_WEIGHT = 1 << 7
    };
};
/*
__constant__ double _aRW[9]; //camera externals Rotation defined in world
__constant__ double _aTW[3]; //camera externals Translation defined in world
__constant__ double _aCW[3]; //camera center*/
struct SVolumn{
	pcl::device::Intr sCameraIntrinsics_;
	float _fVoxelSize; 
	float _fTruncDistanceM; 

	pcl::device::Mat33 _Rw;
	//float3 _Tw; 
	float3 _Cw; 

	cv::gpu::DevMem2D_<float> _cvgmDepthScaled;
	cv::gpu::DevMem2D_<short2> _cvgmYZxXVolume;
	
	__device__ __forceinline__ float3 gridToCoordinateVolume(const int3& n3Grid_ ) 
	{
		float x =  n3Grid_.x * _fVoxelSize;
		float y =  n3Grid_.y * _fVoxelSize;// - convert from cv to GL
		float z =  n3Grid_.z * _fVoxelSize;// - convert from cv to GL
		return make_float3( x,y,z );	
	}

	__device__ __forceinline__ void operator () (){
		int nX = threadIdx.x + blockIdx.x * blockDim.x; // for each y*z z0,z1,...
		int nY = threadIdx.y + blockIdx.y * blockDim.y; 
		if (nX >= _cvgmYZxXVolume.cols && nY >= _cvgmYZxXVolume.rows) return;
		int nHalfCols = _cvgmYZxXVolume.rows/2;
		float fHalfVoxelSize = _fVoxelSize/2.f;

		//calc grid idx
		int3 n3Grid;
		n3Grid.x = nY;
		n3Grid.y = nX/_cvgmYZxXVolume.rows;
		n3Grid.z = nX%_cvgmYZxXVolume.rows;
		//calc voxel center coordinate, 0,1|2,3 // -1.5,-0.5|0.5,1.5 //fVoxelSize = 1.0
		float3 fVoxelCenter = gridToCoordinateVolume(n3Grid) ;

		//convert voxel to camera coordinate (local coordinate)
		//fVoxelCenterLocal = R * fVoxelCenter + T = R * ( fVoxelCenter - Cw )
		float3 fVoxelCenterLocal;
		fVoxelCenterLocal = _Rw * ( fVoxelCenter - _Cw );
		
		/*fVoxelCenterLocal.x = _aRW[0]*fVoxelCenter.x+_aRW[3]*fVoxelCenter.y+_aRW[6]*fVoxelCenter.z+_aTW[0];
		fVoxelCenterLocal.y = _aRW[1]*fVoxelCenter.x+_aRW[4]*fVoxelCenter.y+_aRW[7]*fVoxelCenter.z+_aTW[1];
		fVoxelCenterLocal.z = _aRW[2]*fVoxelCenter.x+_aRW[5]*fVoxelCenter.y+_aRW[8]*fVoxelCenter.z+_aTW[2];*/
		//project voxel local to image to pick up corresponding depth
		int c = __float2int_rn((sCameraIntrinsics_.fx * fVoxelCenterLocal.x + sCameraIntrinsics_.cx * fVoxelCenterLocal.z)/fVoxelCenterLocal.z);
		int r = __float2int_rn((sCameraIntrinsics_.fy * fVoxelCenterLocal.y + sCameraIntrinsics_.cy * fVoxelCenterLocal.z)/fVoxelCenterLocal.z);
		if (c < 0 || r < 0 || c >= _cvgmDepthScaled.cols || r >= _cvgmDepthScaled.rows) return;

		//get the depthScaled
		const float& fDepth = _cvgmDepthScaled.ptr(r)[c];	if(isnan<float>(fDepth) || fDepth < 0.1) return;

		float3 Tmp; 
		Tmp = fVoxelCenter - _Cw;
		/*Tmp.x = fVoxelCenter.x - _aCW[0];
		Tmp.y = fVoxelCenter.y - _aCW[1];
		Tmp.z = fVoxelCenter.z - _aCW[2];*/
		float fSignedDistance = fDepth - sqrt(Tmp.x*Tmp.x + Tmp.y*Tmp.y+ Tmp.z*Tmp.z); //- outside + inside
		float fTrancDistInv = 1.0f / _fTruncDistanceM;
		/*float fTSDF;
		if(fSignedDistance > 0 ){

				fTSDF = fmin ( 1.0f, fSignedDistance * fTrancDistInv ); 
		}
		else{
				fTSDF = fmax (-1.0f, fSignedDistance * fTrancDistInv );
		}// truncated and normalize the Signed Distance to [-1,1]
	
		//read an unpack tsdf value and store into the volumes
		short2& sValue = _cvgmYZxXVolume.ptr(nY)[nX];
		float fTSDFNew;
		int nWeightNew;
		if(sValue.x < 30000 ){
			float fTSDFPrev;
			int nWeightPrev;
			pcl::device::unpack_tsdf(sValue,fTSDFPrev,nWeightPrev);
			fTSDFNew = (fTSDFPrev*nWeightPrev + fTSDF*1.f)/(1.f+nWeightPrev);
			nWeightNew = min(STSDF::MAX_WEIGHT,nWeightPrev+1);
		}else{
			fTSDFNew = fTSDF;
			nWeightNew = 1;
		}
		pcl::device::pack_tsdf( fTSDFNew,nWeightNew,sValue);*/


		float fTSDF = fSignedDistance * fTrancDistInv;
		//read an unpack tsdf value and store into the volumes
		short2& sValue = _cvgmYZxXVolume.ptr(nY)[nX];
		float fTSDFNew,fTSDFPrev;
		int nWeightNew,nWeightPrev;
		if(fTSDF > 0.f ){
			fTSDF = fmin ( 1.f, fTSDF );
			
			if(abs(sValue.x) < 30000 ){
				pcl::device::unpack_tsdf(sValue,fTSDFPrev,nWeightPrev);
				fTSDFNew = (fTSDFPrev*nWeightPrev + fTSDF*1.f)/(1.f+nWeightPrev);
				nWeightNew = min(STSDF::MAX_WEIGHT,nWeightPrev+1);
			}else{
				fTSDFNew = fTSDF;
				nWeightNew = 1;
			}
			pcl::device::pack_tsdf( fTSDFNew,nWeightNew,sValue);	
		}
		else{//if (fTSDF < = 0.f)
			fTSDF = fmax ( -1.f, fTSDF );
			
			if(abs(sValue.x) < 30000 ){
				pcl::device::unpack_tsdf(sValue,fTSDFPrev,nWeightPrev);
				fTSDFNew = (fTSDFPrev*nWeightPrev + fTSDF*1.f)/(1.f+nWeightPrev);
				nWeightNew = min(STSDF::MAX_WEIGHT,nWeightPrev+1);
			}else{
				fTSDFNew = fTSDF;
				nWeightNew = 1;
			}
			pcl::device::pack_tsdf( fTSDFNew,nWeightNew,sValue);	
		}// truncated and normalize the Signed Distance to [-1,1]
		
		return;
	}//kernelIntegrateFrame2VolumeCVmCVm()
};
 
__global__ void kernelIntegrateFrame2VolumeCVmCVm( SVolumn sSV_ ){
	sSV_();
}

void integrateFrame2VolumeCVCV(cv::gpu::GpuMat& cvgmDepthScaled_, const unsigned short usPyrLevel_, 
const float fVoxelSize_, const float fTruncDistanceM_, 
const pcl::device::Mat33& Rw_, const float3& Cw_, 
//const double* pR_, const double* pT_,  const double* pC_, 
const float fFx_, const float fFy_, const float u_, const float v_, cv::gpu::GpuMat* pcvgmYZxXVolume_){
	//pR_ is colume major 
	/*size_t sN1 = sizeof(double) * 9;
	cudaSafeCall( cudaMemcpyToSymbol(_aRW, pR_, sN1) );
	size_t sN2 = sizeof(double) * 3;
	cudaSafeCall( cudaMemcpyToSymbol(_aTW, pT_, sN2) );
	cudaSafeCall( cudaMemcpyToSymbol(_aCW, pC_, sN2) );*/

	SVolumn sSV;

	sSV._Rw = Rw_;
	sSV._Cw = Cw_;

	sSV.sCameraIntrinsics_ = pcl::device::Intr(fFx_,fFy_,u_,v_)(usPyrLevel_);
	sSV._cvgmDepthScaled = cvgmDepthScaled_;
	sSV._fVoxelSize = fVoxelSize_;
	sSV._fTruncDistanceM = fTruncDistanceM_;
	sSV._cvgmYZxXVolume = *pcvgmYZxXVolume_;
	//define grid and block
	dim3 block(64, 16);
    dim3 grid(cv::gpu::divUp(pcvgmYZxXVolume_->cols, block.x), cv::gpu::divUp(pcvgmYZxXVolume_->rows, block.y));
	kernelIntegrateFrame2VolumeCVmCVm<<<grid,block>>>( sSV );
	cudaSafeCall ( cudaGetLastError () );
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__constant__ float _aParam[2];//0:_fThreshold;1:_fSize

__global__ void kernelThresholdVolume2by2CVGL(const cv::gpu::DevMem2D_<short2> cvgmYZxXVolume_,cv::gpu::DevMem2D_<float3> cvgmYZxXVolCenter_){
	int nX = threadIdx.x + blockIdx.x * blockDim.x; // for each y*z z0,z1,...
    int nY = threadIdx.y + blockIdx.y * blockDim.y; 
	if (nX >= cvgmYZxXVolume_.cols && nY >= cvgmYZxXVolume_.rows) return; //both nX and nX and bounded by cols as the structure is a cubic

    const short2& sValue = cvgmYZxXVolume_.ptr(nY)[nX];
	float3& fCenter = cvgmYZxXVolCenter_.ptr(nY)[nX];
	
	int nGridX = nY;
	int nGridY = nX/cvgmYZxXVolume_.rows;
	int nGridZ = nX%cvgmYZxXVolume_.rows;
	float fTSDF = pcl::device::unpack_tsdf(sValue);
	if(fabsf(fTSDF)<_aParam[0]){
		fCenter.x = nGridX *_aParam[1] ;
		fCenter.y = nGridY *_aParam[1] ;// - convert from cv to GL
		fCenter.z = nGridZ *_aParam[1] ;// - convert from cv to GL
	}//within threshold
	else{
		fCenter.x = fCenter.y = fCenter.z = pcl::device::numeric_limits<float>::quiet_NaN();
	}
	return;
}//kernelThresholdVolume()

void thresholdVolumeCVGL(const cv::gpu::GpuMat& cvgmYZxXVolume_, const float fThreshold_, const float fVoxelSize_, const cv::gpu::GpuMat* pcvgmYZxXVolCenter_){
	size_t sN = sizeof(float)*2;
	float* const pParam = (float*) malloc( sN );
	pParam[0] = fThreshold_;
	pParam[1] = fVoxelSize_;
	cudaSafeCall( cudaMemcpyToSymbol(_aParam, pParam, sN) );
	dim3 block(64, 16);
    dim3 grid(cv::gpu::divUp(cvgmYZxXVolume_.cols, block.x), cv::gpu::divUp(cvgmYZxXVolume_.rows, block.y));
	//kernelThresholdVolumeCVGL<<<grid,block>>>(cvgmYZxXVolume_,*pcvgmYZxXVolCenter_);
	kernelThresholdVolume2by2CVGL<<<grid,block>>>(cvgmYZxXVolume_,*pcvgmYZxXVolCenter_);
	cudaSafeCall ( cudaGetLastError () );
}//thresholdVolume()
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct SCross{
	ushort _usV;
	cv::gpu::DevMem2D_<short2> _cvgmYZxXVolume;
	cv::gpu::DevMem2D_<uchar3> _cvgmCross;
	ushort _usType; // cross-section intersept with X, Y, or Z axis

	__device__ __forceinline__ void operator () () {
		int nX = threadIdx.x + blockIdx.x * blockDim.x; // for each y*z z0,z1,...
		int nY = threadIdx.y + blockIdx.y * blockDim.y; 
		if (nX >= _cvgmYZxXVolume.cols && nY >= _cvgmYZxXVolume.rows) return;
		
		//calc grid idx
		int3 n3Grid;
		n3Grid.x = nY;
		n3Grid.y = nX/_cvgmYZxXVolume.rows;
		n3Grid.z = nX%_cvgmYZxXVolume.rows;

		int Axis,XX,YY;
		switch(_usType){
			case 1: //intercepting X
				Axis = n3Grid.x;
				XX = n3Grid.y;
				YY = n3Grid.z;
				break;
			case 2: //intercepting Y
				Axis = n3Grid.y;
				XX = n3Grid.x;
				YY = n3Grid.z;
				break;
			case 3: //intercepting Z
				Axis = n3Grid.z;
				XX = n3Grid.x;
				YY = n3Grid.y;
				break;
		}//switch

		if( Axis == _usV ){
			// get truncated signed distance value and weight
			short2& sValue = _cvgmYZxXVolume.ptr(nY)[nX];
			float fTSDF;
			int nWeight;
			pcl::device::unpack_tsdf(sValue,fTSDF,nWeight);
			uchar3& pixel = _cvgmCross.ptr(YY)[XX];  
			if( fTSDF > 0.f  )
			{
				if (fTSDF > 1.f){
					pixel.x = 0;
					pixel.y = (uchar)255;
					pixel.z = 0;
				}
				else{
					pixel.x = pixel.y = pixel.z = uchar(abs(fTSDF)*255 );
				}
			}
			else{
				if (fTSDF < -1.f){
					pixel.x = (uchar)255;
					pixel.y = 0;
					pixel.z = 0;
				}
				else{
					pixel.x = pixel.y = pixel.z = uchar(abs(fTSDF)*255 );
				}
			}
		}
	}//kernelIntegrateFrame2VolumeCVmCVm()
};

__global__ void kernelExportVolume2CrossSection( SCross sSC_ ){
	sSC_();
}
void exportVolume2CrossSectionX(const cv::gpu::GpuMat& cvgmYZxXVolContentCV_, ushort usV_, ushort usType_, cv::gpu::GpuMat* pcvgmCross_){
	SCross sSC;
	sSC._usV = usV_;
	sSC._usType = usType_;
	sSC._cvgmCross = *pcvgmCross_;
	sSC._cvgmYZxXVolume = cvgmYZxXVolContentCV_;

	dim3 block(64, 16);
    dim3 grid(cv::gpu::divUp(cvgmYZxXVolContentCV_.cols, block.x), cv::gpu::divUp(cvgmYZxXVolContentCV_.rows, block.y));
	kernelExportVolume2CrossSection<<<grid,block>>>( sSC );
	cudaSafeCall ( cudaGetLastError () );
}//exportVolume2CrossSectionX()
















}//device
}//btl