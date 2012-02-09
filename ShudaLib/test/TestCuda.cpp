#define INFO

//#include "boost/date_time/posix_time/posix_time.hpp"
#include <iostream>

//using namespace boost::posix_time;
//using namespace boost::gregorian;
using namespace std;
#include <opencv2/gpu/gpu.hpp>
#include "../cuda/CudaLib.h"
#include "../OtherUtil.hpp"

int testCuda()
{
	PRINTSTR("test Cuda: cudaDepth2Disparity() && cudaDisparity2Depth()");
	const int nRow = 4;
	const int nCol = 6;
	const int N = nRow*nCol;
	cv::Mat cvmDepth( nRow,nCol,CV_32F );
	cv::Mat cvmDisparity,cvmResult;

	float* pDepth = (float*) cvmDepth.data;
    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++) {
        *pDepth++ = i;
    }

	cv::gpu::GpuMat cvgmDepth,cvgmDisparity,cvgmResult;
	cvgmDepth.upload(cvmDepth);
	PRINT(cvmDepth);
    btl::cuda_util::cudaDepth2Disparity( cvgmDepth, &cvgmDisparity ); 
	btl::cuda_util::cudaDisparity2Depth( cvgmDisparity, &cvgmResult ); 
	cvgmDisparity.download(cvmDisparity);
	PRINT(cvmDisparity);
	cvgmResult.download(cvmResult);
	PRINT(cvmResult);
    return 0;
}
