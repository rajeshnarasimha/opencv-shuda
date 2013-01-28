#ifndef __FREAK_H__
#define __FREAK_H__



#ifdef __cplusplus
#include <limits>
namespace btl{
namespace image{

using namespace std;
using namespace cv;

	/*
 * Abstract base class for computing descriptors for image keypoints.
 *
 * In this interface we assume a keypoint descriptor can be represented as a
 * dense, fixed-dimensional vector of some basic type. Most descriptors used
 * in practice follow this pattern, as it makes it very easy to compute
 * distances between descriptors. Therefore we represent a collection of
 * descriptors as a Mat, where each row is one keypoint descriptor.
 */
class CV_EXPORTS_W DescriptorExtractor //: public virtual Algorithm
{
public:
    virtual ~DescriptorExtractor();

    /*
     * Compute the descriptors for a set of keypoints in an image.
     * image        The image.
     * keypoints    The input keypoints. Keypoints for which a descriptor cannot be computed are removed.
     * descriptors  Copmputed descriptors. Row i is the descriptor for keypoint i.
     */
    void compute( const cv::Mat& image, vector<KeyPoint>& keypoints, cv::Mat& descriptors ) const;

    /*
     * Compute the descriptors for a keypoints collection detected in image collection.
     * images       Image collection.
     * keypoints    Input keypoints collection. keypoints[i] is keypoints detected in images[i].
     *              Keypoints for which a descriptor cannot be computed are removed.
     * descriptors  Descriptor collection. descriptors[i] are descriptors computed for set keypoints[i].
     */
    void compute( const vector<Mat>& images, vector<vector<KeyPoint> >& keypoints, vector<Mat>& descriptors ) const;

    CV_WRAP virtual int descriptorSize() const = 0;
    CV_WRAP virtual int descriptorType() const = 0;

    CV_WRAP virtual bool empty() const;

    //CV_WRAP static Ptr<DescriptorExtractor> create( const string& descriptorExtractorType );

    //void compute( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const = 0;
protected:
    /*
     * Remove keypoints within borderPixels of an image edge.
     */
    static void removeBorderKeypoints( vector<KeyPoint>& keypoints,
                                      Size imageSize, int borderSize );
};

/*!
  FREAK implementation
*/
class CV_EXPORTS FREAK : public DescriptorExtractor
{
public:
    /** Constructor
         * @param orientationNormalized enable orientation normalization
         * @param scaleNormalized enable scale normalization
         * @param patternScale scaling of the description pattern
         * @param nbOctave number of octaves covered by the detected keypoints
         * @param selectedPairs (optional) user defined selected pairs
    */
    explicit FREAK( bool orientationNormalized = true,
           bool scaleNormalized = true,
           float patternScale = 22.0f,
           int nOctaves = 4,
           const vector<int>& selectedPairs = vector<int>());
    FREAK( const FREAK& rhs );
    FREAK& operator=( const FREAK& );

    virtual ~FREAK();

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

   // AlgorithmInfo* info() const;

    enum
    {
        NB_SCALES = 64, NB_PAIRS = 512, NB_ORIENPAIRS = 45
    };
	void compute( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) ;

protected:
    void buildPattern();
    uchar meanIntensity( const Mat& image, const Mat& integral, const float kp_x, const float kp_y,
                         const unsigned int scale, const unsigned int rot, const unsigned int point ) const;

    bool orientationNormalized; //true if the orientation is normalized, false otherwise
    bool scaleNormalized; //true if the scale is normalized, false otherwise
    double patternScale; //scaling of the pattern
    int nOctaves; //number of octaves
    bool extAll; // true if all pairs need to be extracted for pairs selection

    double patternScale0;
    int nOctaves0;
    vector<int> selectedPairs0;

    struct PatternPoint
    {
        float x; // x coordinate relative to center
        float y; // x coordinate relative to center
        float sigma; // Gaussian smoothing sigma
    };

    struct DescriptionPair
    {
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

    vector<PatternPoint> patternLookup; // look-up table for the pattern points (position+sigma of all points at all scales and orientation)
										// 64 scale x 256 orientation x 43 points
    int patternSizes[NB_SCALES]; // size of the pattern at a specific scale (used to check if a point is within image boundaries)
    DescriptionPair descriptionPairs[NB_PAIRS]; //512 pairs of patches
    OrientationPair orientationPairs[NB_ORIENPAIRS]; //45 pairs of patches the same as used in paper
};

}// namespace image
}// namespace btl

#endif
#endif