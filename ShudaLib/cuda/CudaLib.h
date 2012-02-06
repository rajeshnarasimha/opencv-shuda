#ifndef BTL_CUDA_HEADER
#define BTL_CUDA_HEADER

namespace btl
{
namespace cuda_util
{

int cudaDepth2Disparity( const float* pDepth_, const int& nRow_, const int& nCol_, float *pDisparity_  ); 
int cudaDisparity2Depth( const float* pDisparity_, const int& nRow_, const int& nCol_, float *pDepth_  );

}//cuda_util
}//btl
#endif