//#include "precomp.hpp"

#include "opencv2/gpu/gpu.hpp"
#include <opencv2/gpu/device/common.hpp>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <npp.h>
#include "Surf.h"

using namespace cv;
using namespace cv::gpu;
using namespace std;



namespace btl { namespace device
{
    namespace surf
    {
        void loadGlobalConstants(int maxCandidates, int maxFeatures, int img_rows, int img_cols, int nOctaveLayers, float hessianThreshold);
        void loadOctaveConstants(int octave, int layer_rows, int layer_cols);

        void bindImgTex(PtrStepSzb img);
        size_t bindSumTex(PtrStepSz<unsigned int> sum);
        size_t bindMaskSumTex(PtrStepSz<unsigned int> maskSum);

        void icvCalcLayerDetAndTrace_gpu(const PtrStepf& det, const PtrStepf& trace, int img_rows, int img_cols,
            int octave, int nOctaveLayers, const size_t sumOffset);

        void icvFindMaximaInLayer_gpu(const PtrStepf& det, const PtrStepf& trace, int4* maxPosBuffer, unsigned int* maxCounter,
            int img_rows, int img_cols, int octave, bool use_mask, int nLayers, const size_t maskOffset);

        void icvInterpolateKeypoint_gpu(const PtrStepf& det, const int4* maxPosBuffer, unsigned int maxCounter,
            float* featureX, float* featureY, int* featureLaplacian, int* featureOctave, float* featureSize, float* featureHessian,
            unsigned int* featureCounter);

        void icvCalcOrientation_gpu(const float* featureX, const float* featureY, const float* featureSize, float* featureDir, int nFeatures);

        void compute_descriptors_gpu(const PtrStepSzf& descriptors,
            const float* featureX, const float* featureY, const float* featureSize, const float* featureDir, int nFeatures);
    }
}//device
}//btl


namespace
{
    int calcSize(int octave, int layer)
    {
        /* Wavelet size at first layer of first octave. */
        const int HAAR_SIZE0 = 9;

        /* Wavelet size increment between layers. This should be an even number,
         such that the wavelet sizes in an octave are either all even or all odd.
         This ensures that when looking for the neighbours of a sample, the layers

         above and below are aligned correctly. */
        const int HAAR_SIZE_INC = 6;

        return (HAAR_SIZE0 + HAAR_SIZE_INC * layer) << octave;
    }

    class CSurfInvoker
    {
    public:
        CSurfInvoker(CSurf& surf, const GpuMat& img, const GpuMat& mask) :
            surf_(surf),
            img_cols(img.cols), img_rows(img.rows),
            use_mask(!mask.empty())
        {
            CV_Assert(!img.empty() && img.type() == CV_8UC1);
            CV_Assert(mask.empty() || (mask.size() == img.size() && mask.type() == CV_8UC1));
            CV_Assert(surf_.nOctaves > 0 && surf_.nOctaveLayers > 0);

            if (!TargetArchs::builtWith(GLOBAL_ATOMICS) || !DeviceInfo().supports(GLOBAL_ATOMICS))
                CV_Error(CV_StsNotImplemented, "The device doesn't support global atomics");

            const int min_size = calcSize(surf_.nOctaves - 1, 0);
            CV_Assert(img_rows - min_size >= 0);
            CV_Assert(img_cols - min_size >= 0);

            const int layer_rows = img_rows >> (surf_.nOctaves - 1);
            const int layer_cols = img_cols >> (surf_.nOctaves - 1);
            const int min_margin = ((calcSize((surf_.nOctaves - 1), 2) >> 1) >> (surf_.nOctaves - 1)) + 1;
            CV_Assert(layer_rows - 2 * min_margin > 0);
            CV_Assert(layer_cols - 2 * min_margin > 0);

            maxFeatures = min(static_cast<int>(img.size().area() * surf.keypointsRatio), 65535);
            maxCandidates = min(static_cast<int>(1.5 * maxFeatures), 65535);

            CV_Assert(maxFeatures > 0);

            counters.create(1, surf_.nOctaves + 1, CV_32SC1);
            counters.setTo(Scalar::all(0));

            btl::device::surf::loadGlobalConstants(maxCandidates, maxFeatures, img_rows, img_cols, surf_.nOctaveLayers, static_cast<float>(surf_.hessianThreshold));

            btl::device::surf::bindImgTex(img);
            integralBuffered(img, surf_.sum, surf_.intBuffer);

            sumOffset = btl::device::surf::bindSumTex(surf_.sum);

            return;

            if (use_mask)
            {
                min(mask, 1.0, surf_.mask1);
                integralBuffered(surf_.mask1, surf_.maskSum, surf_.intBuffer);
                maskOffset = btl::device::surf::bindMaskSumTex(surf_.maskSum);
            }
        }

        void detectKeypoints(GpuMat& keypoints)
        {
            ensureSizeIsEnough(img_rows * (surf_.nOctaveLayers + 2), img_cols, CV_32FC1, surf_.det);
            ensureSizeIsEnough(img_rows * (surf_.nOctaveLayers + 2), img_cols, CV_32FC1, surf_.trace);

            ensureSizeIsEnough(1, maxCandidates, CV_32SC4, surf_.maxPosBuffer);
            ensureSizeIsEnough(CSurf::ROWS_COUNT, maxFeatures, CV_32FC1, keypoints);
            keypoints.setTo(Scalar::all(0));

            for (int octave = 0; octave < surf_.nOctaves; ++octave)
            {
                const int layer_rows = img_rows >> octave;
                const int layer_cols = img_cols >> octave;

                btl::device::surf::loadOctaveConstants(octave, layer_rows, layer_cols);

                btl::device::surf::icvCalcLayerDetAndTrace_gpu(surf_.det, surf_.trace, img_rows, img_cols, octave, surf_.nOctaveLayers, sumOffset);

                btl::device::surf::icvFindMaximaInLayer_gpu(surf_.det, surf_.trace, surf_.maxPosBuffer.ptr<int4>(), counters.ptr<unsigned int>() + 1 + octave,
                    img_rows, img_cols, octave, use_mask, surf_.nOctaveLayers, maskOffset);

                unsigned int maxCounter;
                cudaSafeCall( cudaMemcpy(&maxCounter, counters.ptr<unsigned int>() + 1 + octave, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
                maxCounter = std::min(maxCounter, static_cast<unsigned int>(maxCandidates));

                if (maxCounter > 0)
                {
                    btl::device::surf::icvInterpolateKeypoint_gpu(surf_.det, surf_.maxPosBuffer.ptr<int4>(), maxCounter,
                        keypoints.ptr<float>(CSurf::X_ROW), keypoints.ptr<float>(CSurf::Y_ROW),
                        keypoints.ptr<int>(CSurf::LAPLACIAN_ROW), keypoints.ptr<int>(CSurf::OCTAVE_ROW),
                        keypoints.ptr<float>(CSurf::SIZE_ROW), keypoints.ptr<float>(CSurf::HESSIAN_ROW),
                        counters.ptr<unsigned int>());
                }
            }
            unsigned int featureCounter;
            cudaSafeCall( cudaMemcpy(&featureCounter, counters.ptr<unsigned int>(), sizeof(unsigned int), cudaMemcpyDeviceToHost) );
            featureCounter = std::min(featureCounter, static_cast<unsigned int>(maxFeatures));

            keypoints.cols = featureCounter;

            if (surf_.upright)
                keypoints.row(CSurf::ANGLE_ROW).setTo(Scalar::all(360.0 - 90.0));
            else
                findOrientation(keypoints);
        }

        void findOrientation(GpuMat& keypoints)
        {
            const int nFeatures = keypoints.cols;
            if (nFeatures > 0)
            {
                btl::device::surf::icvCalcOrientation_gpu(keypoints.ptr<float>(CSurf::X_ROW), keypoints.ptr<float>(CSurf::Y_ROW),
                    keypoints.ptr<float>(CSurf::SIZE_ROW), keypoints.ptr<float>(CSurf::ANGLE_ROW), nFeatures);
            }
        }

        void computeDescriptors(const GpuMat& keypoints, GpuMat& descriptors, int descriptorSize)
        {
            const int nFeatures = keypoints.cols;
            if (nFeatures > 0)
            {
                ensureSizeIsEnough(nFeatures, descriptorSize, CV_32F, descriptors);
                btl::device::surf::compute_descriptors_gpu(descriptors, keypoints.ptr<float>(CSurf::X_ROW), keypoints.ptr<float>(CSurf::Y_ROW),
                    keypoints.ptr<float>(CSurf::SIZE_ROW), keypoints.ptr<float>(CSurf::ANGLE_ROW), nFeatures);
            }
        }

    private:
        CSurf& surf_;

        int img_cols, img_rows;

        bool use_mask;

        int maxCandidates;
        int maxFeatures;

        size_t maskOffset;
        size_t sumOffset;

        GpuMat counters;
    };
}

CSurf::CSurf()
{
    hessianThreshold = 100;
    extended = true;
    nOctaves = 4;
    nOctaveLayers = 2;
    keypointsRatio = 0.01f;
    upright = false;
}

CSurf::CSurf(double _threshold, int _nOctaves, int _nOctaveLayers, bool _extended, float _keypointsRatio, bool _upright)
{
    hessianThreshold = _threshold;
    extended = _extended;
    nOctaves = _nOctaves;
    nOctaveLayers = _nOctaveLayers;
    keypointsRatio = _keypointsRatio;
    upright = _upright;
}

int CSurf::descriptorSize() const
{
    return extended ? 128 : 64;
}

void CSurf::uploadKeypoints(const vector<KeyPoint>& keypoints, GpuMat& keypointsGPU)
{
    if (keypoints.empty())
        keypointsGPU.release();
    else
    {
        Mat keypointsCPU(CSurf::ROWS_COUNT, static_cast<int>(keypoints.size()), CV_32FC1);

        float* kp_x = keypointsCPU.ptr<float>(CSurf::X_ROW);
        float* kp_y = keypointsCPU.ptr<float>(CSurf::Y_ROW);
        int* kp_laplacian = keypointsCPU.ptr<int>(CSurf::LAPLACIAN_ROW);
        int* kp_octave = keypointsCPU.ptr<int>(CSurf::OCTAVE_ROW);
        float* kp_size = keypointsCPU.ptr<float>(CSurf::SIZE_ROW);
        float* kp_dir = keypointsCPU.ptr<float>(CSurf::ANGLE_ROW);
        float* kp_hessian = keypointsCPU.ptr<float>(CSurf::HESSIAN_ROW);

        for (size_t i = 0, size = keypoints.size(); i < size; ++i)
        {
            const KeyPoint& kp = keypoints[i];
            kp_x[i] = kp.pt.x;
            kp_y[i] = kp.pt.y;
            kp_octave[i] = kp.octave;
            kp_size[i] = kp.size;
            kp_dir[i] = kp.angle;
            kp_hessian[i] = kp.response;
            kp_laplacian[i] = 1;
        }

        keypointsGPU.upload(keypointsCPU);
    }
}

void CSurf::downloadKeypoints(const GpuMat& keypointsGPU, vector<KeyPoint>& keypoints)
{
    const int nFeatures = keypointsGPU.cols;

    if (nFeatures == 0)
        keypoints.clear();
    else
    {
        CV_Assert(keypointsGPU.type() == CV_32FC1 && keypointsGPU.rows == ROWS_COUNT);

        Mat keypointsCPU(keypointsGPU);

        keypoints.resize(nFeatures);

        float* kp_x = keypointsCPU.ptr<float>(CSurf::X_ROW);
        float* kp_y = keypointsCPU.ptr<float>(CSurf::Y_ROW);
        int* kp_laplacian = keypointsCPU.ptr<int>(CSurf::LAPLACIAN_ROW);
        int* kp_octave = keypointsCPU.ptr<int>(CSurf::OCTAVE_ROW);
        float* kp_size = keypointsCPU.ptr<float>(CSurf::SIZE_ROW);
        float* kp_dir = keypointsCPU.ptr<float>(CSurf::ANGLE_ROW);
        float* kp_hessian = keypointsCPU.ptr<float>(CSurf::HESSIAN_ROW);

        for (int i = 0; i < nFeatures; ++i)
        {
            KeyPoint& kp = keypoints[i];
            kp.pt.x = kp_x[i];
            kp.pt.y = kp_y[i];
            kp.class_id = kp_laplacian[i];
            kp.octave = kp_octave[i];
            kp.size = kp_size[i];
            kp.angle = kp_dir[i];
            kp.response = kp_hessian[i];
        }
    }
}

void CSurf::downloadDescriptors(const GpuMat& descriptorsGPU, vector<float>& descriptors)
{
    if (descriptorsGPU.empty())
        descriptors.clear();
    else
    {
        CV_Assert(descriptorsGPU.type() == CV_32F);

        descriptors.resize(descriptorsGPU.rows * descriptorsGPU.cols);
        Mat descriptorsCPU(descriptorsGPU.size(), CV_32F, &descriptors[0]);
        descriptorsGPU.download(descriptorsCPU);
    }
}

void CSurf::operator()(const GpuMat& img, const GpuMat& mask, GpuMat& keypoints)
{
    if (!img.empty())
    {
        CSurfInvoker surf(*this, img, mask);

        surf.detectKeypoints(keypoints);
    }
}

void CSurf::operator()(const GpuMat& img, const GpuMat& mask, GpuMat& keypoints, GpuMat& descriptors,
                                   bool useProvidedKeypoints)
{
    if (!img.empty())
    {
        CSurfInvoker surf(*this, img, mask);

        if (!useProvidedKeypoints)
            surf.detectKeypoints(keypoints);
        else if (!upright)
        {
            surf.findOrientation(keypoints);
        }

        surf.computeDescriptors(keypoints, descriptors, descriptorSize());
    }
}

void CSurf::operator()(const GpuMat& img, const GpuMat& mask, vector<KeyPoint>& keypoints)
{
    GpuMat keypointsGPU;

    (*this)(img, mask, keypointsGPU);

    downloadKeypoints(keypointsGPU, keypoints);
}

void CSurf::operator()(const GpuMat& img, const GpuMat& mask, vector<KeyPoint>& keypoints,
    GpuMat& descriptors, bool useProvidedKeypoints)
{
    GpuMat keypointsGPU;

    if (useProvidedKeypoints)
        uploadKeypoints(keypoints, keypointsGPU);

    (*this)(img, mask, keypointsGPU, descriptors, useProvidedKeypoints);

    downloadKeypoints(keypointsGPU, keypoints);
}

void CSurf::operator()(const GpuMat& img, const GpuMat& mask, vector<KeyPoint>& keypoints,
    vector<float>& descriptors, bool useProvidedKeypoints)
{
    GpuMat descriptorsGPU;

    (*this)(img, mask, keypoints, descriptorsGPU, useProvidedKeypoints);

    downloadDescriptors(descriptorsGPU, descriptors);
}

void CSurf::releaseMemory()
{
    sum.release();
    mask1.release();
    maskSum.release();
    intBuffer.release();
    det.release();
    trace.release();
    maxPosBuffer.release();
}

