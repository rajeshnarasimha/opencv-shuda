
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
#include "Histogram.h"
#include "KeyFrame.h"
#include "Model.h"
#include "cuda/CudaLib.h"


namespace btl{ namespace geometry
{

CModel::CModel()
{
	_fVolumeSize = 3.f; //3m
	_fVoxelSize = _fVolumeSize/VOLUME_RESOL;
	_cvgmYZxXVolContentCV.create(VOLUME_RESOL,VOLUME_LEVEL,CV_16SC2);//y*z,x
	_cvgmYZxXVolContentCV.setTo(0);
	_cvgmYZxXVolContentCV.download(_cvmYZxXVolContent);
}
CModel::~CModel(void)
{
	if(_pGL) _pGL->releaseVBO(_uVBO,_pResourceVBO);
}

void CModel::gpuIntegrate( btl::kinect::CKeyFrame& cFrame_, unsigned short usPyrLevel_ ){
	//BTL_ASSERT( btl::utility::BTL_CV == cFrame_._eConvention, "the frame depth data must be captured in cv-convention");
	//btl::cuda_util::cudaIntegrate(*cFrame_._acvgmShrPtrPyrPts[usPyrLevel_],cFrame_._eimR.data(),cFrame_._eivT.data(),&_cvgmYZxXVolContentCV);
	//_cvgmYZxXVolContentCV.download(_cvmYZxXVolContent);
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
	btl::cuda_util::thresholdVolumeCVGL(_cvgmYZxXVolContentCV,0.01f,_fVoxelSize,&cvgmYZxZVolCentersGL);

	cudaGraphicsUnmapResources(1, &_pResourceVBO, 0);

	// render from the vbo
	btl::gl_util::CGLUtil::glBindBuffer(GL_ARRAY_BUFFER, _uVBO);
	glVertexPointer(3, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 0.0, 0.0);
	glDrawArrays(GL_POINTS, 0, VOXEL_TOTAL );
	glDisableClientState(GL_VERTEX_ARRAY);
}
}//geometry
}//btl
