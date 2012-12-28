
#include <thrust/sort.h>

#include "opencv2/gpu/device/common.hpp"
#include "opencv2/gpu/device/utility.hpp"
#include "opencv2/gpu/device/functional.hpp"

namespace btl { namespace device {  namespace orb  {
////////////////////////////////////////////////////////////////////////////////////////////////////////
// cull
// sort the fast corners according to their strength
// nSize_ is the total no. of points to be sorted
// nCullPoints_ is the # of points will left
// nSize_ is total # of corners before culling
int thrustSortFastCornersAndCull(int* pnLoc_, float* pfResponse_, const int nCornersBeforeCulling_, const int nCullPoints_)
{
    thrust::device_ptr<int> loc_ptr(pnLoc_);
    thrust::device_ptr<float> response_ptr(pfResponse_);

    thrust::sort_by_key(response_ptr, response_ptr + nCornersBeforeCulling_, loc_ptr, thrust::greater<float>());

    return nCullPoints_;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// kernelHarrisResponses

__global__ void kernelHarrisResponses(const cv::gpu::PtrStepb cvgmImg_, const short2* loc_, float* pfResponse_, const int nPoints_, const int nBlockSize_, const float harris_k)
{
    __shared__ int smem[8 * 32];

    volatile int* srow = smem + threadIdx.y * blockDim.x;

    const int nPtIdx = blockIdx.x * blockDim.y + threadIdx.y;

    if (nPtIdx < nPoints_)
    {
        const short2 loc = loc_[nPtIdx];

        const int r = nBlockSize_ / 2;
        const int x0 = loc.x - r;
        const int y0 = loc.y - r;

        int a = 0, b = 0, c = 0;

        for (int ind = threadIdx.x; ind < nBlockSize_ * nBlockSize_; ind += blockDim.x)
        {
            const int i = ind / nBlockSize_;
            const int j = ind % nBlockSize_;

            int Ix = (cvgmImg_(y0 + i, x0 + j + 1)     - cvgmImg_(y0 + i, x0 + j - 1)) * 2 +
					 (cvgmImg_(y0 + i - 1, x0 + j + 1) - cvgmImg_(y0 + i - 1, x0 + j - 1)) +
					 (cvgmImg_(y0 + i + 1, x0 + j + 1) - cvgmImg_(y0 + i + 1, x0 + j - 1));

            int Iy = (cvgmImg_(y0 + i + 1, x0 + j)     - cvgmImg_(y0 + i - 1, x0 + j)) * 2 +
					 (cvgmImg_(y0 + i + 1, x0 + j - 1) - cvgmImg_(y0 + i - 1, x0 + j - 1)) +
					 (cvgmImg_(y0 + i + 1, x0 + j + 1) - cvgmImg_(y0 + i - 1, x0 + j + 1));

            a += Ix * Ix;
            b += Iy * Iy;
            c += Ix * Iy;
        }
        cv::gpu::device::reduce<32>(srow, a, threadIdx.x, cv::gpu::device::plus<volatile int>());
        cv::gpu::device::reduce<32>(srow, b, threadIdx.x, cv::gpu::device::plus<volatile int>());
        cv::gpu::device::reduce<32>(srow, c, threadIdx.x, cv::gpu::device::plus<volatile int>());

        if (threadIdx.x == 0)
        {
            float scale = (1 << 2) * nBlockSize_ * 255.0f;
            scale = 1.0f / scale;
            const float scale_sq_sq = scale * scale * scale * scale;

            pfResponse_[nPtIdx] = ((float)a * b - (float)c * c - harris_k * ((float)a + b) * ((float)a + b)) * scale_sq_sq;
        }
    }//if (nPtIdx < nPoints_)
}

void cudaCalcHarrisResponses(cv::gpu::PtrStepSzb cvgmImg_, const short2* s2Loc_, float* pfResponse_, const int nPoints_, int nBlockSize_, float harris_k, cudaStream_t stream)
{
    dim3 block(32, 8);

    dim3 grid;
    grid.x = cv::gpu::divUp(nPoints_, block.y);

    kernelHarrisResponses<<<grid, block, 0, stream>>>(cvgmImg_, s2Loc_, pfResponse_, nPoints_, nBlockSize_, harris_k);

    cudaSafeCall( cudaGetLastError() );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// IC_Angle

__constant__ int c_u_max[32];

void loadUMax(const int* u_max, int count)
{
    cudaSafeCall( cudaMemcpyToSymbol(c_u_max, u_max, count * sizeof(int)) );
}

__global__ void IC_Angle(const cv::gpu::PtrStepb image, const short2* loc_, float* pAngle_, const int nPoints_, const int half_k)
{
    __shared__ int smem[8 * 32];//Every thread in the block shares the shared memory

    volatile int* srow = smem + threadIdx.y * blockDim.x; //The volatile keyword specifies that the value associated with the name that follows can be modified by actions other than those in the user application. 

    const int nPtIdx = blockIdx.x * blockDim.y + threadIdx.y;

    if (nPtIdx < nPoints_)
    {
        int m_01 = 0, m_10 = 0;

        const short2 loc = loc_[nPtIdx];

        // Treat the center line differently, v=0
        for (int u = threadIdx.x - half_k; u <= half_k; u += blockDim.x)
            m_10 += u * image(loc.y, loc.x + u);

        cv::gpu::device::reduce<32>(srow, m_10, threadIdx.x, cv::gpu::device::plus<volatile int>());

        for (int v = 1; v <= half_k; ++v)
        {
            // Proceed over the two lines
            int v_sum = 0;
            int m_sum = 0;
            const int d = c_u_max[v];//1/4 circular patch

            for (int u = threadIdx.x - d; u <= d; u += blockDim.x)
            {
                int val_plus = image(loc.y + v, loc.x + u);
                int val_minus = image(loc.y - v, loc.x + u);

                v_sum += (val_plus - val_minus);
                m_sum += u * (val_plus + val_minus);
            }

            cv::gpu::device::reduce<32>(srow, v_sum, threadIdx.x, cv::gpu::device::plus<volatile int>());
            cv::gpu::device::reduce<32>(srow, m_sum, threadIdx.x, cv::gpu::device::plus<volatile int>());

            m_10 += m_sum;
            m_01 += v * v_sum;
        }

        if (threadIdx.x == 0)
        {
            float kp_dir = ::atan2f((float)m_01, (float)m_10);
            kp_dir += (kp_dir < 0) * (2.0f * CV_PI);
            kp_dir *= 180.0f / CV_PI;

            pAngle_[nPtIdx] = kp_dir;
        }
    }
	return;
}

void IC_Angle_gpu(cv::gpu::PtrStepSzb image, const short2* ps2Loc_, float* pAngle_, int nPoints_, int half_k, cudaStream_t stream)
{
    dim3 block(32, 8);

    dim3 grid;
    grid.x = cv::gpu::divUp(nPoints_, block.y);

    IC_Angle<<<grid, block, 0, stream>>>(image, ps2Loc_, pAngle_, nPoints_, half_k);

    cudaSafeCall( cudaGetLastError() );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// kernelComputeOrbDescriptor

template <int WTA_K> struct OrbDescriptor;

#define GET_VALUE(idx) \
    cvgmImg_(s2Loc_.y + __float2int_rn(pnPatternX_[idx] * sina + pnPatternY_[idx] * cosa), \
             s2Loc_.x + __float2int_rn(pnPatternX_[idx] * cosa - pnPatternY_[idx] * sina))

template <> struct OrbDescriptor<2>
{
    __device__ static unsigned char calc(const cv::gpu::PtrStepb& cvgmImg_, short2 s2Loc_, const int* pnPatternX_, const int* pnPatternY_, float sina, float cosa, int nDescIdx_)
    {
        pnPatternX_ += 16 * nDescIdx_; //compare 8 pairs of points, and that is 16 points in total
        pnPatternY_ += 16 * nDescIdx_;

        int t0, t1;
		unsigned char val;

        t0 = GET_VALUE(0); t1 = GET_VALUE(1);
        val = t0 < t1;

        t0 = GET_VALUE(2); t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;

        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;

        t0 = GET_VALUE(6); t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;

        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;

        t0 = GET_VALUE(10); t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;

        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;

        t0 = GET_VALUE(14); t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        return val;
    }
};

template <> struct OrbDescriptor<3>
{
    __device__ static unsigned char calc(const cv::gpu::PtrStepb& cvgmImg_, short2 s2Loc_, const int* pnPatternX_, const int* pnPatternY_, float sina, float cosa, int nDescIdx_)
    {
        pnPatternX_ += 12 * nDescIdx_;
        pnPatternY_ += 12 * nDescIdx_;

        int t0, t1, t2;
		unsigned char val;

        t0 = GET_VALUE(0); t1 = GET_VALUE(1); t2 = GET_VALUE(2);
        val = t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0);

        t0 = GET_VALUE(3); t1 = GET_VALUE(4); t2 = GET_VALUE(5);
        val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 2;

        t0 = GET_VALUE(6); t1 = GET_VALUE(7); t2 = GET_VALUE(8);
        val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 4;

        t0 = GET_VALUE(9); t1 = GET_VALUE(10); t2 = GET_VALUE(11);
        val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 6;

        return val;
    }
};

template <> struct OrbDescriptor<4>
{
    __device__ static unsigned char calc(const cv::gpu::PtrStepb& cvgmImg_, short2 s2Loc_, const int* pnPatternX_, const int* pnPatternY_, float sina, float cosa, int nDescIdx_)
    {
        pnPatternX_ += 16 * nDescIdx_;
        pnPatternY_ += 16 * nDescIdx_;

        int t0, t1, t2, t3, k;
        int a, b;
		unsigned char val;

        t0 = GET_VALUE(0); t1 = GET_VALUE(1);
        t2 = GET_VALUE(2); t3 = GET_VALUE(3);
        a = 0, b = 2;
        if( t1 > t0 ) t0 = t1, a = 1;
        if( t3 > t2 ) t2 = t3, b = 3;
        k = t0 > t2 ? a : b;
        val = k;

        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        t2 = GET_VALUE(6); t3 = GET_VALUE(7);
        a = 0, b = 2;
        if( t1 > t0 ) t0 = t1, a = 1;
        if( t3 > t2 ) t2 = t3, b = 3;
        k = t0 > t2 ? a : b;
        val |= k << 2;

        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        t2 = GET_VALUE(10); t3 = GET_VALUE(11);
        a = 0, b = 2;
        if( t1 > t0 ) t0 = t1, a = 1;
        if( t3 > t2 ) t2 = t3, b = 3;
        k = t0 > t2 ? a : b;
        val |= k << 4;

        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        t2 = GET_VALUE(14); t3 = GET_VALUE(15);
        a = 0, b = 2;
        if( t1 > t0 ) t0 = t1, a = 1;
        if( t3 > t2 ) t2 = t3, b = 3;
        k = t0 > t2 ? a : b;
        val |= k << 6;

        return val;
    }
};

#undef GET_VALUE

template <int WTA_K>
__global__ void kernelComputeOrbDescriptor(const cv::gpu::PtrStepb cvgmImg_, const short2* pLoc_, const float* pAngle_, const int nPoints_,
    const int* pnPatternX_, const int* pnPatternY_, cv::gpu::PtrStepb cvgmDescriptor_, int nDescriptorSize_)
{
    const int nDescIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nPtIdx   = blockIdx.y * blockDim.y + threadIdx.y;

    if (nPtIdx < nPoints_ && nDescIdx < nDescriptorSize_) {
        float fAngle = pAngle_[nPtIdx];
        fAngle *= (float)(CV_PI / 180.f);

        float sina, cosa;
        ::sincosf(fAngle, &sina, &cosa);//Calculate the sine and cosine of the first input argument x (measured in radians).

        cvgmDescriptor_.ptr(nPtIdx)[nDescIdx] = OrbDescriptor<WTA_K>::calc(cvgmImg_, pLoc_[nPtIdx], pnPatternX_, pnPatternY_, sina, cosa, nDescIdx);
    }
}

void cudaComputeOrbDescriptor(cv::gpu::PtrStepb cvgmImg_, const short2* pLoc_, const float* pAngle_, const int nPoints_,
    const int* pnPatternX_, const int* pnPatternY_, cv::gpu::PtrStepb cvgmDescriptor_, int nDescriptorSize_, int WTA_K, cudaStream_t stream)
{
    dim3 block(32, 8);

    dim3 grid;
    grid.x = cv::gpu::divUp(nDescriptorSize_, block.x);
    grid.y = cv::gpu::divUp(nPoints_, block.y);

    switch (WTA_K)
    {
    case 2:
        kernelComputeOrbDescriptor<2><<<grid, block, 0, stream>>>(cvgmImg_, pLoc_, pAngle_, nPoints_, pnPatternX_, pnPatternY_, cvgmDescriptor_, nDescriptorSize_);
        break;

    case 3:
        kernelComputeOrbDescriptor<3><<<grid, block, 0, stream>>>(cvgmImg_, pLoc_, pAngle_, nPoints_, pnPatternX_, pnPatternY_, cvgmDescriptor_, nDescriptorSize_);
        break;

    case 4:
        kernelComputeOrbDescriptor<4><<<grid, block, 0, stream>>>(cvgmImg_, pLoc_, pAngle_, nPoints_, pnPatternX_, pnPatternY_, cvgmDescriptor_, nDescriptorSize_);
        break;
    }

    cudaSafeCall( cudaGetLastError() );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// mergeLocation

__global__ void kernelMergeLocation(const short2* s2Loc_, float* pfX_, float* pfY_, const int nPoints_, float fScale_)
{
    const int nPtIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (nPtIdx < nPoints_)
    {
        short2 s2Loc = s2Loc_[nPtIdx];

        pfX_[nPtIdx] = s2Loc.x * fScale_;
        pfY_[nPtIdx] = s2Loc.y * fScale_;
    }
}
//convert the location of key points at different level of the pyramid into first level
void cudaMergeLocation(const short2* s2Loc_, float* pfX_, float* pfY_, const int nPoints_, const float fScale_, cudaStream_t stream)
{
    dim3 block(256);

    dim3 grid;
    grid.x = cv::gpu::divUp(nPoints_, block.x);

    kernelMergeLocation<<<grid, block, 0, stream>>>(s2Loc_, pfX_, pfY_, nPoints_, fScale_);

    cudaSafeCall( cudaGetLastError() );

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}


} //namespace orb
} //namespace device
} //namespace btl


