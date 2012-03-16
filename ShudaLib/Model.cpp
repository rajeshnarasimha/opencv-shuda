
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
#include "Model.h"
#include "cuda/CudaLib.h"


namespace btl{ namespace geometry
{

CModel::CModel()
{
	_fVolumeSizeM = 4.f; //3m
	_fVoxelSizeM = _fVolumeSizeM/VOLUME_RESOL;
	_fTruncateDistanceM = _fVoxelSizeM*3;
	_cvgmYZxXVolContentCV.create(VOLUME_RESOL,VOLUME_LEVEL,CV_16SC2);//y*z,x
	_cvgmYZxXVolContentCV.setTo(std::numeric_limits<short>::max());
	//_cvgmYZxXVolContentCV.setTo(0);
	_cvgmYZxXVolContentCV.download(_cvmYZxXVolContent);
}
CModel::~CModel(void)
{
	if(_pGL) _pGL->releaseVBO(_uVBO,_pResourceVBO);
}
void CModel::unpack_tsdf (short2 value, float& tsdf, int& weight)
{
    weight = value.y;
    tsdf =  (value.x) / 32767;   //*/ * INV_DIV;
}
void CModel::gpuIntegrateFrameIntoVolumeCVCV(const btl::kinect::CKeyFrame& cFrame_, unsigned short usPyrLevel_ ){
	Eigen::Vector3d eivCw = - cFrame_._eimRw.transpose() *cFrame_._eivTw ; //get camera center in world coordinate
	BTL_ASSERT( btl::utility::BTL_CV == cFrame_._eConvention, "the frame depth data must be captured in cv-convention");
	_cvgmYZxXVolContentCV.setTo(std::numeric_limits<short>::max());
	btl::device::integrateFrame2VolumeCVCV(*cFrame_._acvgmPyrDepths[usPyrLevel_],usPyrLevel_,
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
void CModel::gpuCreateVBO(){
	if(_pGL) _pGL->createVBO(_cvgmYZxXVolContentCV.rows,_cvgmYZxXVolContentCV.cols,3,sizeof(float),&_uVBO,&_pResourceVBO);
}
void CModel::gpuRenderVoxelInWorldCVGL(){
	// map OpenGL buffer object for writing from CUDA
	void *pDev;
	cudaGraphicsMapResources(1, &_pResourceVBO, 0);
	size_t nSize; 
	cudaGraphicsResourceGetMappedPointer((void **)&pDev, &nSize, _pResourceVBO );
	cv::gpu::GpuMat cvgmYZxZVolCentersGL(_cvgmYZxXVolContentCV.rows,_cvgmYZxXVolContentCV.cols,CV_32FC3,pDev);
	cvgmYZxZVolCentersGL.setTo(std::numeric_limits<float>::quiet_NaN());
	
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
}
}//geometry
}//btl
