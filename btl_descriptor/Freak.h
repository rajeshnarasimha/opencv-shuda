#ifndef __FREAK_H__
#define __FREAK_H__

#ifdef __cplusplus
#include <limits>
namespace btl{
namespace image{

using namespace std;
using namespace cv;

/*!
  FREAK implementation
*/
class CV_EXPORTS CFreak 
{
public:
    /** Constructor
         * @param orientationNormalized enable orientation normalization
         * @param scaleNormalized enable scale normalization
         * @param patternScale scaling of the description pattern 
         * @param nbOctave number of octaves covered by the detected keypoints
         * @param selectedPairs (optional) user defined selected pairs
    */
    explicit CFreak( bool orientationNormalized = true,
           bool scaleNormalized = true, 
           float patternScale = 22.0f, //the radius othe pattern in pixels
           int nOctaves = 4,
           const vector<int>& selectedPairs = vector<int>());
    CFreak( const CFreak& rhs );
    CFreak& operator=( const CFreak& );

    virtual ~CFreak();

    /** returns the descriptor length in bytes */
    virtual int descriptorSize() const;

    /** returns the descriptor type */
    virtual int descriptorType() const;

    /** select the 512 "best description pairs"
         * @param images grayscale images set
         * @param keypoints set of detected keypoints
         * @param corrThresh correlation threshold
         * @param verbose print construction information
         * @return list of best pair indexes
    */
    vector<int> selectPairs( const vector<Mat>& images, vector<vector<KeyPoint> >& keypoints,
                      const double corrThresh = 0.7, bool verbose = true );

    enum {
        NB_SCALES = 64, NB_PAIRS = 512, NB_ORIENPAIRS = 45
    };
	void compute( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) ;
	unsigned int gpuCompute( const cv::gpu::GpuMat& cvgmImg, const cv::gpu::GpuMat& cvgmImgInt_, cv::gpu::GpuMat& cvgmKeyPoint_, cv::gpu::GpuMat* pcvgmDescriptor_ );
	//unsigned int gpuCompute( const Mat& image, cv::gpu::GpuMat& cvgmKeyPoint_, cv::gpu::GpuMat* pcvgmDescriptor_ );
	void downloadKeypoints(const cv::gpu::GpuMat& keypointsGPU, vector<KeyPoint>& keypoints);

protected:
    void buildPattern();
    uchar meanIntensity( const Mat& image, const Mat& integral, const float kp_x, const float kp_y,
                         const unsigned int scale, const unsigned int rot, const unsigned int point ) ;
	bool compare() const; //for debugging gpuBuildPattern() 
	void gpuBuildPattern();
	bool orientationNormalized; //true if the orientation is normalized, false otherwise
    bool scaleNormalized; //true if the scale is normalized, false otherwise
    double patternScale; //scaling of the pattern
    int _nOctaves; //number of octaves
    bool extAll; // true if all pairs need to be extracted for pairs selection

    double patternScale0;
    int nOctaves0;
    vector<int> selectedPairs0;

    struct PatternPoint  {
        float x; // x coordinate relative to center
        float y; // x coordinate relative to center
        float sigma; // Gaussian smoothing sigma
    };

    struct DescriptionPair  {
        uchar i; // index of the first point
        uchar j; // index of the second point
    };

    struct OrientationPair
    {
        uchar i; // index of the first point
        uchar j; // index of the second point
        int weight_dx; // dx/(norm_sq))*4096
        int weight_dy; // dy/(norm_sq))*4096
    };

    vector<PatternPoint> patternLookup; // look-up table for the pattern points (position+sigma of all pattern points at all scales and orientation)
										// 64 scale x 256 orientation x 43 points
    int patternSizes[NB_SCALES]; // size of the pattern at a specific scale (used to check if a point is within image boundaries)
    DescriptionPair descriptionPairs[NB_PAIRS]; //512 pairs of patches
    OrientationPair orientationPairs[NB_ORIENPAIRS]; //45 pairs of patches the same as used in paper

	cv::gpu::GpuMat _cvgmPatternLookup; // float3 x,y,sigma, 64 scale x 256 orientation x 43 points x ( x y sigma ); 
	cv::gpu::GpuMat _cvgmPatternSize;  // int, 1 x 64 scale
	cv::gpu::GpuMat _cvgmOrientationPair; // int4 i,j,weight_x,weight_y NB_ORIENPAIRS  45 pairs of patches the same as used in paper
	cv::gpu::GpuMat _cvgmDescriptorPair; // uchar2 i,j, NB_PAIRS 512 pairs of patches for descriptor

	cv::gpu::GpuMat _cvgmKpScaleIdx; // short, size of the pattern at a specific scale (used to check if a point is within image boundaries)
	std::vector<int> kpScaleIdx;
};

}// namespace image
}// namespace btl

#endif
#endif