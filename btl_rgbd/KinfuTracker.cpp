
#define INFO
//gl
#include <gl/glew.h>
#include <gl/freeglut.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
//boost
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
//stl
#include <vector>
#include <fstream>
#include <list>
#include <limits>
//opencv
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <Eigen/Core>
//self
#include "OtherUtil.hpp"
#include "Converters.hpp"
#include "EigenUtil.hpp"
#include "Camera.h"
#include "Kinect.h"
#include "GLUtil.h"
#include "PlaneObj.h"
#include "Histogram.h"
#include "KeyFrame.h"
#include "KinfuTracker.h"
#include "cuda/CudaLib.h"
#include "cuda/Volume.h"
#include "cuda/pcl/internal.h"
#include "cuda/RayCaster.h"

namespace btl{ namespace geometry
{

CKinfuTracker::CKinfuTracker()
{
	_fVolumeSizeM = 3.f; //3m
	_fVoxelSizeM = _fVolumeSizeM/VOLUME_RESOL;
	_fTruncateDistanceM = _fVoxelSizeM*12;
	_cvgmYZxXVolContentCV.create(VOLUME_RESOL,VOLUME_LEVEL,CV_16SC2);//y*z,x
	_cvgmYZxXVolContentCV.setTo(std::numeric_limits<short>::max());//set EACH channel to max().
	//_cvgmYZxXVolContentCV.setTo(0);
}
CKinfuTracker::~CKinfuTracker(void)
{
	releaseVBOPBO();
	//if(_pGL) _pGL->releaseVBO(_uVBO,_pResourceVBO);
}

void CKinfuTracker::releaseVBOPBO()
{
	//release VBO
	cudaSafeCall( cudaGraphicsUnregisterResource( _pResourceVBO ) );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
	glDeleteBuffers( 1, &_uVBO );

	//release PBO
	// unregister this buffer object with CUDA
	//http://rickarkin.blogspot.co.uk/2012/03/use-pbo-to-share-buffer-between-cuda.html
	cudaSafeCall( cudaGraphicsUnregisterResource( _pResourcePBO ) );
	glDeleteBuffers(1, &_uPBO);
}
void CKinfuTracker::reset(){
	_cvgmYZxXVolContentCV.setTo(std::numeric_limits<short>::max());
}
void CKinfuTracker::unpack_tsdf (short2 value, float& tsdf, int& weight)
{
    weight = value.y;
    tsdf =  (value.x) / 30000;//32767;   //*/ * INV_DIV;
}
void CKinfuTracker::gpuIntegrateFrameIntoVolumeCVCV(const btl::kinect::CKeyFrame& cFrame_){
	//Note: the point cloud int cFrame_ must be transformed into world before calling it
	Eigen::Vector3d eivCw = - cFrame_._eimRw.transpose() *cFrame_._eivTw ; //get camera center in world coordinate
	BTL_ASSERT( btl::utility::BTL_CV == cFrame_._eConvention, "the frame depth data must be captured in cv-convention");
	btl::device::integrateFrame2VolumeCVCV(*cFrame_._acvgmShrPtrPyrDepths[0],0,
		_fVoxelSizeM,_fTruncateDistanceM, 
		cFrame_._eimRw.data(),cFrame_._eivTw.data(), eivCw.data(),//camera parameters
		cFrame_._pRGBCamera->_fFx,cFrame_._pRGBCamera->_fFy,cFrame_._pRGBCamera->_u,cFrame_._pRGBCamera->_v,//
		&_cvgmYZxXVolContentCV);
	/*{
		//test2	
		cv::Mat cvmTest;
		_cvgmYZxXVolContentCV.download(cvmTest);
		short2* pData = (short2*) cvmTest.data;
		for (int r=0; r<cvmTest.rows; r++)
			for (int c=0; c<cvmTest.cols; c++){
				float fTSDF; int nWeight;
				unpack_tsdf(*pData++,fTSDF,nWeight);
				if(fabs(fTSDF)<0.8&&nWeight>0)
					PRINT(fTSDF);
			}
	}*/
	//_cvgmYZxXVolContentCV.download(cvmTest);
}
void CKinfuTracker::gpuRaycast(btl::kinect::CKeyFrame* pVirtualFrame_ ) const {
	Eigen::Matrix3f eimcmRwCur = pVirtualFrame_->_eimRw.cast<float>();
	//device cast do the transpose implicitly because eimcmRwCur is col major by default.
	pcl::device::Mat33& devRwCurTrans = pcl::device::device_cast<pcl::device::Mat33> (eimcmRwCur);
	//Cw = -Rw'*Tw
	Eigen::Vector3f eivCwCur = Eigen::Vector3d( - pVirtualFrame_->_eimRw.transpose() * pVirtualFrame_->_eivTw ).cast<float>();
	float3& devCwCur = pcl::device::device_cast<float3> (eivCwCur);
	//btl::device::raycast(pcl::device::Intr(pVirtualFrame_->_pRGBCamera->_fFx,pVirtualFrame_->_pRGBCamera->_fFy,pVirtualFrame_->_pRGBCamera->_u,pVirtualFrame_->_pRGBCamera->_v)(0),
	//	devRwCurTrans,devCwCur,_fTruncateDistanceM,_fVolumeSizeM, _cvgmYZxXVolContentCV,&*pVirtualFrame_->_acvgmShrPtrPyrDepths[0]);
	btl::device::raycast(pcl::device::Intr(pVirtualFrame_->_pRGBCamera->_fFx,pVirtualFrame_->_pRGBCamera->_fFy,pVirtualFrame_->_pRGBCamera->_u,pVirtualFrame_->_pRGBCamera->_v)(0),
		devRwCurTrans,devCwCur,_fTruncateDistanceM,_fVolumeSizeM, _cvgmYZxXVolContentCV,&*pVirtualFrame_->_acvgmShrPtrPyrPts[0],&*pVirtualFrame_->_acvgmShrPtrPyrNls[0],&*pVirtualFrame_->_acvgmShrPtrPyrDepths[0]);

	//down-sampling
	pVirtualFrame_->_acvgmShrPtrPyrPts[0]->download(*pVirtualFrame_->_acvmShrPtrPyrPts[0]);
	pVirtualFrame_->_acvgmShrPtrPyrNls[0]->download(*pVirtualFrame_->_acvmShrPtrPyrNls[0]);
	for (short s=1; s<pVirtualFrame_->pyrHeight(); s++ ){
		btl::device::resizeMap(false,*pVirtualFrame_->_acvgmShrPtrPyrPts[s-1],&*pVirtualFrame_->_acvgmShrPtrPyrPts[s]);
		btl::device::resizeMap(true, *pVirtualFrame_->_acvgmShrPtrPyrNls[s-1],&*pVirtualFrame_->_acvgmShrPtrPyrNls[s]);
		pVirtualFrame_->_acvgmShrPtrPyrPts[s]->download(*pVirtualFrame_->_acvmShrPtrPyrPts[s]);
		pVirtualFrame_->_acvgmShrPtrPyrNls[s]->download(*pVirtualFrame_->_acvmShrPtrPyrNls[s]);
	}//for each pyramid level
	////construct the pyramid
	//float _fThresholdDepthInMeter = 0.1f;
	//float _fSigmaSpace = 3;
	//float _fSigmaDisparity = 1.f/.6f - 1.f/(.6f+_fThresholdDepthInMeter);
	//pVirtualFrame_->constructPyramid(_fSigmaSpace, _fSigmaDisparity);
}

void CKinfuTracker::gpuCreateVBO(btl::gl_util::CGLUtil::tp_ptr pGL_){
	_pGL = pGL_;
	if(_pGL){
		_pGL->createVBO(_cvgmYZxXVolContentCV.rows,_cvgmYZxXVolContentCV.cols,3,sizeof(float),&_uVBO,&_pResourceVBO);
		_pGL->createPBO(_cvgmYZxXVolContentCV.rows,_cvgmYZxXVolContentCV.cols,3,sizeof(uchar),&_uPBO,&_pResourcePBO,&_uTexture);
	}
}
void CKinfuTracker::gpuRenderVoxelInWorldCVGL(){
	// map OpenGL buffer object for writing from CUDA
	void *pDev;
	cudaGraphicsMapResources(1, &_pResourceVBO, 0);
	size_t nSize; 
	cudaGraphicsResourceGetMappedPointer((void **)&pDev, &nSize, _pResourceVBO );
	cv::gpu::GpuMat cvgmYZxZVolCentersGL(_cvgmYZxXVolContentCV.rows,_cvgmYZxXVolContentCV.cols,CV_32FC3,pDev);
	cvgmYZxZVolCentersGL.setTo(std::numeric_limits<float>::quiet_NaN());


	cv::gpu::GpuMat cvgmYZxZVolColor(_cvgmYZxXVolContentCV.rows,_cvgmYZxXVolContentCV.cols,CV_8UC3,pDev);
	cvgmYZxZVolColor.setTo(std::numeric_limits<uchar>::quiet_NaN());
	
	// execute the kernel
	//download the voxel centers lies between the -threshold and +threshold
	btl::device::thresholdVolumeCVGL(_cvgmYZxXVolContentCV,0.5f,_fVoxelSizeM,&cvgmYZxZVolCentersGL);

	cudaGraphicsUnmapResources(1, &_pResourceVBO, 0);

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, _uVBO);
	glVertexPointer(3, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 0.0, 0.0);
	glDrawArrays(GL_POINTS, 0, VOXEL_TOTAL );
	glDisableClientState(GL_VERTEX_ARRAY);
}//gpuRenderVoxelInWorldCVGL()

void CKinfuTracker::gpuExportVolume(const std::string& strPath_, ushort usNo_, ushort usV_, ushort usAxis_) const{
	cv::gpu::GpuMat cvgmCross(VOLUME_RESOL,VOLUME_RESOL,CV_8UC3);
	btl::device::exportVolume2CrossSectionX(_cvgmYZxXVolContentCV,usV_,usAxis_,&cvgmCross);
	cv::Mat cvmCross(VOLUME_RESOL,VOLUME_RESOL,CV_8UC3);
	cvgmCross.download(cvmCross);
	std::string strVariableName = strPath_ + "cross"+  boost::lexical_cast<std::string> ( usNo_ ) + boost::lexical_cast<std::string> ( usAxis_ ) + "X" + boost::lexical_cast<std::string> ( usV_ ) + ".bmp";
	cv::imwrite(strVariableName.c_str(),cvmCross);
	return;
}

}//geometry
}//btl
