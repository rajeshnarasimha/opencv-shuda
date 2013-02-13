namespace btl { namespace device
{
namespace surf
{
	////////////////////////////////// SURF //////////////////////////////////////////
using namespace std;
using namespace cv;
using namespace cv::gpu;

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
}//surf
}//device
}//btl