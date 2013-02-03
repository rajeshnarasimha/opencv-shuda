//#include "precomp.hpp"
#include <fstream>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <bitset>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>
#define CV_SSE2 0
#include "Freak.h"
#include "Freak.cuh"
#include "Surf.h"
#include "TestFreak.h"

namespace btl {
namespace image {

static const double FREAK_SQRT2 = 1.4142135623731;
static const double FREAK_INV_SQRT2 = 1.0 / FREAK_SQRT2;
static const double FREAK_LOG2 = 0.693147180559945;
static const int FREAK_NB_ORIENTATION = 256;
static const int FREAK_NB_POINTS = 43;
static const int FREAK_SMALLEST_KP_SIZE = 7;// smallest size of keypoints
static const int FREAK_NB_SCALES = FREAK::NB_SCALES;
static const int FREAK_NB_PAIRS = FREAK::NB_PAIRS;
static const int FREAK_NB_ORIENPAIRS = FREAK::NB_ORIENPAIRS;

static const short FREAK_DEF_PAIRS[FREAK::NB_PAIRS] =
{ // default pairs
     404,431,818,511,181,52,311,874,774,543,719,230,417,205,11,
     560,149,265,39,306,165,857,250,8,61,15,55,717,44,412,
     592,134,761,695,660,782,625,487,549,516,271,665,762,392,178,
     796,773,31,672,845,548,794,677,654,241,831,225,238,849,83,
     691,484,826,707,122,517,583,731,328,339,571,475,394,472,580,
     381,137,93,380,327,619,729,808,218,213,459,141,806,341,95,
     382,568,124,750,193,749,706,843,79,199,317,329,768,198,100,
     466,613,78,562,783,689,136,838,94,142,164,679,219,419,366,
     418,423,77,89,523,259,683,312,555,20,470,684,123,458,453,833,
     72,113,253,108,313,25,153,648,411,607,618,128,305,232,301,84,
     56,264,371,46,407,360,38,99,176,710,114,578,66,372,653,
     129,359,424,159,821,10,323,393,5,340,891,9,790,47,0,175,346,
     236,26,172,147,574,561,32,294,429,724,755,398,787,288,299,
     769,565,767,722,757,224,465,723,498,467,235,127,802,446,233,
     544,482,800,318,16,532,801,441,554,173,60,530,713,469,30,
     212,630,899,170,266,799,88,49,512,399,23,500,107,524,90,
     194,143,135,192,206,345,148,71,119,101,563,870,158,254,214,
     276,464,332,725,188,385,24,476,40,231,620,171,258,67,109,
     844,244,187,388,701,690,50,7,850,479,48,522,22,154,12,659,
     736,655,577,737,830,811,174,21,237,335,353,234,53,270,62,
     182,45,177,245,812,673,355,556,612,166,204,54,248,365,226,
     242,452,700,685,573,14,842,481,468,781,564,416,179,405,35,
     819,608,624,367,98,643,448,2,460,676,440,240,130,146,184,
     185,430,65,807,377,82,121,708,239,310,138,596,730,575,477,
     851,797,247,27,85,586,307,779,326,494,856,324,827,96,748,
     13,397,125,688,702,92,293,716,277,140,112,4,80,855,839,1,
     413,347,584,493,289,696,19,751,379,76,73,115,6,590,183,734,
     197,483,217,344,330,400,186,243,587,220,780,200,793,246,824,
     41,735,579,81,703,322,760,720,139,480,490,91,814,813,163,
     152,488,763,263,425,410,576,120,319,668,150,160,302,491,515,
     260,145,428,97,251,395,272,252,18,106,358,854,485,144,550,
     131,133,378,68,102,104,58,361,275,209,697,582,338,742,589,
     325,408,229,28,304,191,189,110,126,486,211,547,533,70,215,
     670,249,36,581,389,605,331,518,442,822
};

struct PairStat
{ // used to sort pairs during pairs selection
    double mean;
    int idx;
};

struct sortMean
{
    bool operator()( const PairStat& a, const PairStat& b ) const {
        return a.mean < b.mean;
    }
};

void FREAK::buildPattern()
{
    if( patternScale == patternScale0 && _nOctaves == nOctaves0 && !patternLookup.empty() )
        return;

    nOctaves0 = _nOctaves;
    patternScale0 = patternScale;

    patternLookup.resize(FREAK_NB_SCALES*FREAK_NB_ORIENTATION*FREAK_NB_POINTS);
	//sample the nOctaves into 64 steps
    double scaleStep = pow(2.0, (double)(_nOctaves)/FREAK_NB_SCALES ); // 2 ^ ( (nOctaves-1) /nbScales)
    double scalingFactor, alpha, beta, theta = 0;

    // pattern definition, radius normalized to 1.0 (outer point position+sigma=1.0)
    const int n[8] = {6,6,6,6,6,6,6,1}; // number of points on each concentric circle (from outer to inner) 43 in total
    const double bigR(2.0/3.0); // bigger radius; assuming the whole circular patch is 1.0
    const double smallR(2.0/24.0); // smaller radius
    const double unitSpace( (bigR-smallR)/21.0 ); // define spaces between concentric circles (from center to outer: 1,2,3,4,5,6) the total is 21
    // radii of the concentric cirles (from outer to inner)
    const double radius[8] = {bigR, bigR-6*unitSpace, bigR-11*unitSpace, bigR-15*unitSpace, bigR-18*unitSpace, bigR-20*unitSpace, smallR, 0.0};
    // sigma of pattern points (each group of 6 points on a concentric circle has the same sigma)
    const double sigma[8] = {radius[0]/2.0, radius[1]/2.0, radius[2]/2.0,
                             radius[3]/2.0, radius[4]/2.0, radius[5]/2.0,
                             radius[6]/2.0, radius[6]/2.0};
    // fill the lookup table
    for( int scaleIdx=0; scaleIdx < FREAK_NB_SCALES; ++scaleIdx ) { //64 scales
        patternSizes[scaleIdx] = 0; // proper initialization
        scalingFactor = pow(scaleStep,scaleIdx); //scale of the pattern, scaleStep ^ scaleIdx

        for( int orientationIdx = 0; orientationIdx < FREAK_NB_ORIENTATION; ++orientationIdx ) { // 256 orientations
            theta = double(orientationIdx)* 2*CV_PI/double(FREAK_NB_ORIENTATION); // orientation of the pattern
            int pointIdx = 0;

            PatternPoint* patternLookupPtr = &patternLookup[0];
            for( size_t i = 0; i < 8; ++i ) { // 8 rings
                for( int k = 0 ; k < n[i]; ++k ) { // 6,6,6,6, 6,6,6,1 = 43 in total
                    beta = CV_PI/n[i] * (i%2); // orientation offset so that groups of points on each circles are staggered
                    alpha = double(k)* 2*CV_PI/double(n[i])+beta+theta;

                    // add the point to the look-up table
                    PatternPoint& point = patternLookupPtr[ scaleIdx*FREAK_NB_ORIENTATION*FREAK_NB_POINTS+orientationIdx*FREAK_NB_POINTS+pointIdx ];
                    point.x = static_cast<float>(radius[i] * cos(alpha) * scalingFactor * patternScale);
                    point.y = static_cast<float>(radius[i] * sin(alpha) * scalingFactor * patternScale);
                    point.sigma = static_cast<float>(sigma[i] * scalingFactor * patternScale);

                    ++pointIdx;
                }
				// adapt the sizeList if necessary
				const int sizeMax = static_cast<int>(ceil((radius[i]+sigma[i])*scalingFactor*patternScale)) + 1;
				if( patternSizes[scaleIdx] < sizeMax )
					patternSizes[scaleIdx] = sizeMax;
            }
        }
    }

    // build the list of orientation pairs
    orientationPairs[0].i=0; orientationPairs[0].j=3; orientationPairs[1].i=1; orientationPairs[1].j=4; orientationPairs[2].i=2; orientationPairs[2].j=5;
    orientationPairs[3].i=0; orientationPairs[3].j=2; orientationPairs[4].i=1; orientationPairs[4].j=3; orientationPairs[5].i=2; orientationPairs[5].j=4;
    orientationPairs[6].i=3; orientationPairs[6].j=5; orientationPairs[7].i=4; orientationPairs[7].j=0; orientationPairs[8].i=5; orientationPairs[8].j=1;

    orientationPairs[9].i=6; orientationPairs[9].j=9; orientationPairs[10].i=7; orientationPairs[10].j=10; orientationPairs[11].i=8; orientationPairs[11].j=11;
    orientationPairs[12].i=6; orientationPairs[12].j=8; orientationPairs[13].i=7; orientationPairs[13].j=9; orientationPairs[14].i=8; orientationPairs[14].j=10;
    orientationPairs[15].i=9; orientationPairs[15].j=11; orientationPairs[16].i=10; orientationPairs[16].j=6; orientationPairs[17].i=11; orientationPairs[17].j=7;

    orientationPairs[18].i=12; orientationPairs[18].j=15; orientationPairs[19].i=13; orientationPairs[19].j=16; orientationPairs[20].i=14; orientationPairs[20].j=17;
    orientationPairs[21].i=12; orientationPairs[21].j=14; orientationPairs[22].i=13; orientationPairs[22].j=15; orientationPairs[23].i=14; orientationPairs[23].j=16;
    orientationPairs[24].i=15; orientationPairs[24].j=17; orientationPairs[25].i=16; orientationPairs[25].j=12; orientationPairs[26].i=17; orientationPairs[26].j=13;

    orientationPairs[27].i=18; orientationPairs[27].j=21; orientationPairs[28].i=19; orientationPairs[28].j=22; orientationPairs[29].i=20; orientationPairs[29].j=23;
    orientationPairs[30].i=18; orientationPairs[30].j=20; orientationPairs[31].i=19; orientationPairs[31].j=21; orientationPairs[32].i=20; orientationPairs[32].j=22;
    orientationPairs[33].i=21; orientationPairs[33].j=23; orientationPairs[34].i=22; orientationPairs[34].j=18; orientationPairs[35].i=23; orientationPairs[35].j=19;

    orientationPairs[36].i=24; orientationPairs[36].j=27; orientationPairs[37].i=25; orientationPairs[37].j=28; orientationPairs[38].i=26; orientationPairs[38].j=29;
    orientationPairs[39].i=30; orientationPairs[39].j=33; orientationPairs[40].i=31; orientationPairs[40].j=34; orientationPairs[41].i=32; orientationPairs[41].j=35;
    orientationPairs[42].i=36; orientationPairs[42].j=39; orientationPairs[43].i=37; orientationPairs[43].j=40; orientationPairs[44].i=38; orientationPairs[44].j=41;

    for( unsigned m = FREAK_NB_ORIENPAIRS; m--; ) {
        const float dx = patternLookup[orientationPairs[m].i].x-patternLookup[orientationPairs[m].j].x;
        const float dy = patternLookup[orientationPairs[m].i].y-patternLookup[orientationPairs[m].j].y;
        const float norm_sq = (dx*dx+dy*dy);
        orientationPairs[m].weight_dx = int((dx/(norm_sq))*4096.0+0.5);//quantization into 4096 levels
        orientationPairs[m].weight_dy = int((dy/(norm_sq))*4096.0+0.5);
    }

    // build the list of description pairs
    std::vector<DescriptionPair> allPairs;
    for( unsigned int i = 1; i < (unsigned int)FREAK_NB_POINTS; ++i ) {
        // (generate all the pairs)
        for( unsigned int j = 0; (unsigned int)j < i; ++j ) {
            DescriptionPair pair = {(uchar)i,(uchar)j};
            allPairs.push_back(pair);
        }
    }
    // Input vector provided
    if( !selectedPairs0.empty() ) {
        if( (int)selectedPairs0.size() == FREAK_NB_PAIRS ) {
            for( int i = 0; i < FREAK_NB_PAIRS; ++i )
                 descriptionPairs[i] = allPairs[selectedPairs0.at(i)];
        }
        else {
            CV_Error(CV_StsVecLengthErr, "Input vector does not match the required size");
        }
    }
    else { // default selected pairs
        for( int i = 0; i < FREAK_NB_PAIRS; ++i ) // 512 in total
             descriptionPairs[i] = allPairs[FREAK_DEF_PAIRS[i]];//assign precomputed default pairs
    }
}

void FREAK::compute( const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors ) {

    if( image.empty() )
        return;
    if( keypoints.empty() )
        return;

    buildPattern();

    Mat imgIntegral;
    integral(image, imgIntegral);
	gpu::GpuMat cvgmImg(image);
	gpu::GpuMat cvgmImgInt(imgIntegral);
	gpu::integral(cvgmImg,cvgmImgInt);
    std::vector<int> kpScaleIdx(keypoints.size()); // used to save pattern scale index corresponding to each keypoints
    const std::vector<int>::iterator ScaleIdxBegin = kpScaleIdx.begin(); // used in std::vector erase function
    const std::vector<cv::KeyPoint>::iterator kpBegin = keypoints.begin(); // used in std::vector erase function
    const float sizeCst = static_cast<float>(FREAK_NB_SCALES/(FREAK_LOG2* _nOctaves));
    uchar pointsValue[FREAK_NB_POINTS];
    int thetaIdx = 0;
    int shnDirection0;
    int shnDirection1;

    // compute the scale index corresponding to the keypoint size and remove keypoints close to the border
    if( scaleNormalized ) {
        for( size_t k = keypoints.size(); k--; ) {
            //Is k non-zero? If so, decrement it and continue"
            kpScaleIdx[k] = max( (int)(log(keypoints[k].size/FREAK_SMALLEST_KP_SIZE)*sizeCst+0.5) ,0); //calc the scale index w.r.t. FREAK scale samples,
             if( kpScaleIdx[k] >= FREAK_NB_SCALES ) // it should lie within 0 and 63 
                kpScaleIdx[k] = FREAK_NB_SCALES-1;

            if( keypoints[k].pt.x <= patternSizes[kpScaleIdx[k]] || //check if the description at this specific position and scale fits inside the image
                 keypoints[k].pt.y <= patternSizes[kpScaleIdx[k]] ||
                 keypoints[k].pt.x >= image.cols-patternSizes[kpScaleIdx[k]] ||
                 keypoints[k].pt.y >= image.rows-patternSizes[kpScaleIdx[k]]
               ) {
                keypoints.erase(kpBegin+k);
                kpScaleIdx.erase(ScaleIdxBegin+k);
            }
        }
    }
    else {
        const int scIdx = max( (int)(1.0986122886681*sizeCst+0.5) ,0);
        for( size_t k = keypoints.size(); k--; ) {
            kpScaleIdx[k] = scIdx; // equivalent to the formule when the scale is normalized with a constant size of keypoints[k].size=3*SMALLEST_KP_SIZE
            if( kpScaleIdx[k] >= FREAK_NB_SCALES ) {
                kpScaleIdx[k] = FREAK_NB_SCALES-1;
            }
            if( keypoints[k].pt.x <= patternSizes[kpScaleIdx[k]] ||
                keypoints[k].pt.y <= patternSizes[kpScaleIdx[k]] ||
                keypoints[k].pt.x >= image.cols-patternSizes[kpScaleIdx[k]] ||
                keypoints[k].pt.y >= image.rows-patternSizes[kpScaleIdx[k]]
               ) {
                keypoints.erase(kpBegin+k);
                kpScaleIdx.erase(ScaleIdxBegin+k);
            }
        }
    }

    // allocate descriptor memory, estimate orientations, extract descriptors
    if( !extAll ) {
        // extract the best comparisons only
        descriptors = cv::Mat::zeros((int)keypoints.size(), FREAK_NB_PAIRS/8, CV_8U);
#if CV_SSE2
        __m128i* ptr= (__m128i*) (descriptors.data+(keypoints.size()-1)*descriptors.step[0]);
#else
        std::bitset<FREAK_NB_PAIRS>* ptr = (std::bitset<FREAK_NB_PAIRS>*) (descriptors.data+(keypoints.size()-1)*descriptors.step[0]);
#endif
        for( size_t k = keypoints.size(); k--; ) {
            // estimate orientation (gradient)
            if( !orientationNormalized ) {
                thetaIdx = 0; // assign 0бу to all keypoints
                keypoints[k].angle = 0.0;
            }
            else {
                // get the points intensity value in the un-rotated pattern
                for( int i = FREAK_NB_POINTS; i--; ) {
                    pointsValue[i] = meanIntensity(image, imgIntegral, keypoints[k].pt.x,keypoints[k].pt.y, kpScaleIdx[k], 0, i);
                }
                shnDirection0 = 0;
                shnDirection1 = 0;
                for( int m = 45; m--; ) {
                    //iterate through the orientation pairs
                    const int delta = (pointsValue[ orientationPairs[m].i ]-pointsValue[ orientationPairs[m].j ]);
                    shnDirection0 += delta*(orientationPairs[m].weight_dx)/2048;
                    shnDirection1 += delta*(orientationPairs[m].weight_dy)/2048;
                }

                keypoints[k].angle = static_cast<float>(atan2((float)shnDirection1,(float)shnDirection0)*(180.0/CV_PI));//estimate orientation
                thetaIdx = int(FREAK_NB_ORIENTATION*keypoints[k].angle*(1/360.0)+0.5);
                if( thetaIdx < 0 )
                    thetaIdx += FREAK_NB_ORIENTATION;

                if( thetaIdx >= FREAK_NB_ORIENTATION )
                    thetaIdx -= FREAK_NB_ORIENTATION;
            }
            // extract descriptor at the computed orientation
            for( int i = FREAK_NB_POINTS; i--; ) {
                pointsValue[i] = meanIntensity(image, imgIntegral, keypoints[k].pt.x,keypoints[k].pt.y, kpScaleIdx[k], thetaIdx, i);
            }
#if CV_SSE2
            // note that comparisons order is modified in each block (but first 128 comparisons remain globally the same-->does not affect the 128,384 bits segmanted matching strategy)
            int cnt = 0;
            for( int n = FREAK_NB_PAIRS/128; n-- ; )
            {
                __m128i result128 = _mm_setzero_si128();
                for( int m = 128/16; m--; cnt += 16 )
                {
                    __m128i operand1 = _mm_set_epi8(
                        pointsValue[descriptionPairs[cnt+0].i],
                        pointsValue[descriptionPairs[cnt+1].i],
                        pointsValue[descriptionPairs[cnt+2].i],
                        pointsValue[descriptionPairs[cnt+3].i],
                        pointsValue[descriptionPairs[cnt+4].i],
                        pointsValue[descriptionPairs[cnt+5].i],
                        pointsValue[descriptionPairs[cnt+6].i],
                        pointsValue[descriptionPairs[cnt+7].i],
                        pointsValue[descriptionPairs[cnt+8].i],
                        pointsValue[descriptionPairs[cnt+9].i],
                        pointsValue[descriptionPairs[cnt+10].i],
                        pointsValue[descriptionPairs[cnt+11].i],
                        pointsValue[descriptionPairs[cnt+12].i],
                        pointsValue[descriptionPairs[cnt+13].i],
                        pointsValue[descriptionPairs[cnt+14].i],
                        pointsValue[descriptionPairs[cnt+15].i]);

                    __m128i operand2 = _mm_set_epi8(
                        pointsValue[descriptionPairs[cnt+0].j],
                        pointsValue[descriptionPairs[cnt+1].j],
                        pointsValue[descriptionPairs[cnt+2].j],
                        pointsValue[descriptionPairs[cnt+3].j],
                        pointsValue[descriptionPairs[cnt+4].j],
                        pointsValue[descriptionPairs[cnt+5].j],
                        pointsValue[descriptionPairs[cnt+6].j],
                        pointsValue[descriptionPairs[cnt+7].j],
                        pointsValue[descriptionPairs[cnt+8].j],
                        pointsValue[descriptionPairs[cnt+9].j],
                        pointsValue[descriptionPairs[cnt+10].j],
                        pointsValue[descriptionPairs[cnt+11].j],
                        pointsValue[descriptionPairs[cnt+12].j],
                        pointsValue[descriptionPairs[cnt+13].j],
                        pointsValue[descriptionPairs[cnt+14].j],
                        pointsValue[descriptionPairs[cnt+15].j]);

                    __m128i workReg = _mm_min_epu8(operand1, operand2); // emulated "not less than" for 8-bit UNSIGNED integers
                    workReg = _mm_cmpeq_epi8(workReg, operand2);        // emulated "not less than" for 8-bit UNSIGNED integers

                    workReg = _mm_and_si128(_mm_set1_epi16(short(0x8080 >> m)), workReg); // merge the last 16 bits with the 128bits std::vector until full
                    result128 = _mm_or_si128(result128, workReg);
                }
                (*ptr) = result128;
                ++ptr;
            }
            ptr -= 8;
#else
            // extracting descriptor preserving the order of SSE version
            int cnt = 0;
            for( int n = 7; n < FREAK_NB_PAIRS; n += 128)
            {
                for( int m = 8; m--; )
                {
                    int nm = n-m;
                    for(int kk = nm+15*8; kk >= nm; kk-=8, ++cnt)
                    {
                        ptr->set(kk, pointsValue[descriptionPairs[cnt].i] >= pointsValue[descriptionPairs[cnt].j]);
                    }
                }
            }
            --ptr;
#endif
        }
    }
    else { // extract all possible comparisons for selection
        descriptors = cv::Mat::zeros((int)keypoints.size(), 128, CV_8U);
        std::bitset<1024>* ptr = (std::bitset<1024>*) (descriptors.data+(keypoints.size()-1)*descriptors.step[0]);

        for( size_t k = keypoints.size(); k--; ) {
            //estimate orientation (gradient)
            if( !orientationNormalized ) {
                thetaIdx = 0;//assign 0бу to all keypoints
                keypoints[k].angle = 0.0;
            }
            else {
                //get the points intensity value in the un-rotated pattern
                for( int i = FREAK_NB_POINTS;i--; )
                    pointsValue[i] = meanIntensity(image, imgIntegral, keypoints[k].pt.x,keypoints[k].pt.y, kpScaleIdx[k], 0, i);

                shnDirection0 = 0;
                shnDirection1 = 0;
                for( int m = 45; m--; ) {
                    //iterate through the orientation pairs
                    const int delta = (pointsValue[ orientationPairs[m].i ]-pointsValue[ orientationPairs[m].j ]);
                    shnDirection0 += delta*(orientationPairs[m].weight_dx)/2048;
                    shnDirection1 += delta*(orientationPairs[m].weight_dy)/2048;
                }

                keypoints[k].angle = static_cast<float>(atan2((float)shnDirection1,(float)shnDirection0)*(180.0/CV_PI)); //estimate orientation
                thetaIdx = int(FREAK_NB_ORIENTATION*keypoints[k].angle*(1/360.0)+0.5);

                if( thetaIdx < 0 )
                    thetaIdx += FREAK_NB_ORIENTATION;

                if( thetaIdx >= FREAK_NB_ORIENTATION )
                    thetaIdx -= FREAK_NB_ORIENTATION;
            }
            // get the points intensity value in the rotated pattern
            for( int i = FREAK_NB_POINTS; i--; ) {
                pointsValue[i] = meanIntensity( image, imgIntegral, keypoints[k].pt.x,
                                                keypoints[k].pt.y, kpScaleIdx[k], thetaIdx, i);
            }

            int cnt(0);
            for( int i = 1; i < FREAK_NB_POINTS; ++i ) {
                //(generate all the pairs)
                for( int j = 0; j < i; ++j ) {
                    ptr->set(cnt, pointsValue[i] >= pointsValue[j] );
                    ++cnt;
                }
            }
            --ptr;
        }
    }
}

// simply take average on a square patch, not even gaussian approx
uchar FREAK::meanIntensity( const cv::Mat& image, const cv::Mat& integral,
                            const float kp_x,
                            const float kp_y,
                            const unsigned int scale,
                            const unsigned int rot,
                            const unsigned int point)  {
    // get point position in image
    const PatternPoint& FreakPoint = patternLookup[scale*FREAK_NB_ORIENTATION*FREAK_NB_POINTS + rot*FREAK_NB_POINTS + point];
    const float xf = FreakPoint.x+kp_x;
    const float yf = FreakPoint.y+kp_y;
    const int x = int(xf);
    const int y = int(yf);
    const int& imagecols = image.cols;

    // get the sigma:
    const float radius = FreakPoint.sigma;

    // calculate output:
    if( radius < 0.5 ) {
        // interpolation multipliers:
        const int r_x = static_cast<int>((xf-x)*1024);
        const int r_y = static_cast<int>((yf-y)*1024);
        const int r_x_1 = (1024-r_x);
        const int r_y_1 = (1024-r_y);
        uchar* ptr = image.data+x+y*imagecols;
        int ret_val;
        // linear interpolation:
        ret_val = (r_x_1*r_y_1*int(*ptr));
        ptr++;
        ret_val += (r_x*r_y_1*int(*ptr));
        ptr += imagecols;
        ret_val += (r_x*r_y*int(*ptr));
        ptr--;
        ret_val += (r_x_1*r_y*int(*ptr));
        //return the rounded mean
        ret_val += 2 * 1024 * 1024;
        return static_cast<uchar>(ret_val / (4 * 1024 * 1024));
    }

    // expected case:

    // calculate borders
    const int x_left = int(xf-radius+0.5);
    const int y_top = int(yf-radius+0.5);
    const int x_right = int(xf+radius+1.5);//integral image is 1px wider
    const int y_bottom = int(yf+radius+1.5);//integral image is 1px higher
    int ret_val;

    ret_val = integral.at<int>(y_bottom,x_right);//bottom right corner
    ret_val -= integral.at<int>(y_bottom,x_left);
    ret_val += integral.at<int>(y_top,x_left);
    ret_val -= integral.at<int>(y_top,x_right);
    ret_val = ret_val/( (x_right-x_left)* (y_bottom-y_top) );
    //~ std::cout<<integral.step[1]<<std::endl;
    return static_cast<uchar>(ret_val);
}

// pair selection algorithm from a set of training images and corresponding keypoints
vector<int> FREAK::selectPairs(const std::vector<Mat>& images
                                        , std::vector<std::vector<KeyPoint> >& keypoints
                                        , const double corrTresh
                                        , bool verbose )
{
    extAll = true;
    // compute descriptors with all pairs
    Mat descriptors;

    if( verbose )
        std::cout << "Number of images: " << images.size() << std::endl;

    for( size_t i = 0;i < images.size(); ++i ) {
        Mat descriptorsTmp;
        compute(images[i],keypoints[i],descriptorsTmp);
        descriptors.push_back(descriptorsTmp);
    }

    if( verbose )
        std::cout << "number of keypoints: " << descriptors.rows << std::endl;

    //descriptor in floating point format (each bit is a float)
    Mat descriptorsFloat = Mat::zeros(descriptors.rows, 903, CV_32F);

    std::bitset<1024>* ptr = (std::bitset<1024>*) (descriptors.data+(descriptors.rows-1)*descriptors.step[0]);
    for( int m = descriptors.rows; m--; ) {
        for( int n = 903; n--; ) {
            if( ptr->test(n) == true )
                descriptorsFloat.at<float>(m,n)=1.0f;
        }
        --ptr;
    }

    std::vector<PairStat> pairStat;
    for( int n = 903; n--; ) {
        // the higher the variance, the better --> mean = 0.5
        PairStat tmp = { fabs( mean(descriptorsFloat.col(n))[0]-0.5 ) ,n};
        pairStat.push_back(tmp);
    }

    std::sort( pairStat.begin(),pairStat.end(), sortMean() );

    std::vector<PairStat> bestPairs;
    for( int m = 0; m < 903; ++m ) {
        if( verbose )
            std::cout << m << ":" << bestPairs.size() << " " << std::flush;
        double corrMax(0);

        for( size_t n = 0; n < bestPairs.size(); ++n ) {
            int idxA = bestPairs[n].idx;
            int idxB = pairStat[m].idx;
            double corr(0);
            // compute correlation between 2 pairs
            corr = fabs(compareHist(descriptorsFloat.col(idxA), descriptorsFloat.col(idxB), CV_COMP_CORREL));

            if( corr > corrMax ) {
                corrMax = corr;
                if( corrMax >= corrTresh )
                    break;
            }
        }

        if( corrMax < corrTresh/*0.7*/ )
            bestPairs.push_back(pairStat[m]);

        if( bestPairs.size() >= 512 ) {
            if( verbose )
                std::cout << m << std::endl;
            break;
        }
    }

    std::vector<int> idxBestPairs;
    if( (int)bestPairs.size() >= FREAK_NB_PAIRS ) {
        for( int i = 0; i < FREAK_NB_PAIRS; ++i )
            idxBestPairs.push_back(bestPairs[i].idx);
    }
    else {
        if( verbose )
            std::cout << "correlation threshold too small (restrictive)" << std::endl;
        CV_Error(CV_StsError, "correlation threshold too small (restrictive)");
    }
    extAll = false;
    return idxBestPairs;
}


/*
void FREAKImpl::drawPattern()
{ // create an image showing the brisk pattern
    Mat pattern = Mat::zeros(1000, 1000, CV_8UC3) + Scalar(255,255,255);
    int sFac = 500 / patternScale;
    for( int n = 0; n < kNB_POINTS; ++n ) {
        PatternPoint& pt = patternLookup[n];
        circle(pattern, Point( pt.x*sFac,pt.y*sFac)+Point(500,500), pt.sigma*sFac, Scalar(0,0,255),2);
        // rectangle(pattern, Point( (pt.x-pt.sigma)*sFac,(pt.y-pt.sigma)*sFac)+Point(500,500), Point( (pt.x+pt.sigma)*sFac,(pt.y+pt.sigma)*sFac)+Point(500,500), Scalar(0,0,255),2);

        circle(pattern, Point( pt.x*sFac,pt.y*sFac)+Point(500,500), 1, Scalar(0,0,0),3);
        std::ostringstream oss;
        oss << n;
        putText( pattern, oss.str(), Point( pt.x*sFac,pt.y*sFac)+Point(500,500), FONT_HERSHEY_SIMPLEX,0.5, Scalar(0,0,0), 1);
    }
    imshow( "FreakDescriptorExtractor pattern", pattern );
    waitKey(0);
}
*/

// -------------------------------------------------
/* FREAK interface implementation */
FREAK::FREAK( bool _orientationNormalized, bool _scaleNormalized
            , float _patternScale, int _nOctaves, const std::vector<int>& _selectedPairs )
    : orientationNormalized(_orientationNormalized), scaleNormalized(_scaleNormalized),
    patternScale(_patternScale), _nOctaves(_nOctaves), extAll(false), nOctaves0(0), selectedPairs0(_selectedPairs)
{
	const float sizeCst = static_cast<float>(FREAK_NB_SCALES/(FREAK_LOG2* _nOctaves));
	btl::device::freak::loadGlobalConstants(_nOctaves, sizeCst , FREAK_SMALLEST_KP_SIZE, FREAK_NB_POINTS, FREAK_NB_SCALES,
											FREAK_NB_ORIENPAIRS, FREAK_NB_ORIENTATION,   FREAK_NB_PAIRS,  FREAK_LOG2);

	//buildPattern();//for test
	gpuBuildPattern();
	//compare(); //for test
	_cvgmKpScaleIdx.create(1, 8000, CV_16SC1);
}

FREAK::~FREAK()
{
}

int FREAK::descriptorSize() const {
    return FREAK_NB_PAIRS / 8; // descriptor length in bytes
}

int FREAK::descriptorType() const {
    return CV_8U;
}

bool FREAK::compare() const {
	bool bTrue = true;
	cv::Mat cvmPatternLookup; _cvgmPatternLookup.download(cvmPatternLookup);
	for (int i=0; i < cvmPatternLookup.cols; i++ ){
		float f1 = fabs( patternLookup[i].x - cvmPatternLookup.ptr<float3>(0)[i].x );
		float f2 = fabs( patternLookup[i].y - cvmPatternLookup.ptr<float3>(0)[i].y );
		float f3 = fabs( patternLookup[i].sigma - cvmPatternLookup.ptr<float3>(0)[i].z );
		if (f1+f2+f3> 0.00001){
			return false;
		}
	}

	cv::Mat cvmPatternSize; _cvgmPatternSize.download(cvmPatternSize);
	for (int i=0; i < cvmPatternSize.cols; i++ ){
		int n1 = abs( cvmPatternSize.ptr<int>(0)[i] - patternSizes[i] );
		if (n1!=0){
			return false;
		}
	}

	cv::Mat cvmOrientationPair; _cvgmOrientationPair.download(cvmOrientationPair);
	for (int i=0; i < cvmOrientationPair.cols; i++) {
		const int4 n4Orien = cvmOrientationPair.ptr<int4>(0)[i];
		int n1 = abs( n4Orien.x - orientationPairs[i].i);
		int n2 = abs( n4Orien.y - orientationPairs[i].j);
		int n3 = abs( n4Orien.z - orientationPairs[i].weight_dx);
		int n4 = abs( n4Orien.w - orientationPairs[i].weight_dy);
		if (n1+n2+n3+n4 != 0)
			return false;
	}

	cv::Mat cvmDescriptorPair; _cvgmDescriptorPair.download(cvmDescriptorPair);
	for (int i=0; i < cvmDescriptorPair.cols; i++) {
		int n1 = abs( cvmDescriptorPair.ptr<uchar2>(0)[i].x - descriptionPairs[i].i );
		int n2 = abs( cvmDescriptorPair.ptr<uchar2>(0)[i].y - descriptionPairs[i].j );
		if (n1+n2!=0)
			return false;
	}
	return true;
}
void FREAK::downloadKeypoints(const cv::gpu::GpuMat& keypointsGPU, vector<KeyPoint>& keypoints)
{
	using namespace btl::image;
	const int nFeatures = keypointsGPU.cols;

	keypoints.clear();
	kpScaleIdx.clear();
	if (nFeatures != 0)
	{
		CV_Assert(keypointsGPU.type() == CV_32FC1 && keypointsGPU.rows == CSurf::ROWS_COUNT);

		Mat keypointsCPU(keypointsGPU);
		keypoints.reserve(nFeatures);
		kpScaleIdx.reserve(nFeatures); // used to save pattern scale index corresponding to each keypoints

		float* kp_x = keypointsCPU.ptr<float>(CSurf::X_ROW);
		float* kp_y = keypointsCPU.ptr<float>(CSurf::Y_ROW);
		int* kp_laplacian = keypointsCPU.ptr<int>(CSurf::LAPLACIAN_ROW);
		int* kp_octave = keypointsCPU.ptr<int>(CSurf::OCTAVE_ROW);
		float* kp_size = keypointsCPU.ptr<float>(CSurf::SIZE_ROW);
		float* kp_dir = keypointsCPU.ptr<float>(CSurf::ANGLE_ROW);
		float* kp_hessian = keypointsCPU.ptr<float>(CSurf::HESSIAN_ROW);

		cv::Mat cvmKpScaleIdx; if(!_cvgmKpScaleIdx.empty()) _cvgmKpScaleIdx.download(cvmKpScaleIdx);

		for (int i = 0; i < nFeatures; ++i)
		{
			//if(kp_hessian[i]<0.f) continue;
			KeyPoint kp;
			kp.pt.x = kp_x[i];
			kp.pt.y = kp_y[i];
			kp.class_id = kp_laplacian[i];
			kp.octave = kp_octave[i];
			kp.size = kp_size[i];
			kp.angle = kp_dir[i];
			kp.response = kp_hessian[i];
			keypoints.push_back(kp);
			//if(!cvmKpScaleIdx.empty()) 
			kpScaleIdx.push_back(cvmKpScaleIdx.ptr<short>()[i]);
		}
	}
}
void FREAK::gpuBuildPattern()
{
	// the output of the gpuBuildPattern() including 
	// 1. _cvgmPatternLookup// 1 x FREAK_NB_SCALES*FREAK_NB_ORIENTATION*FREAK_NB_POINTS
	// 2. _cvgmPatternSize // 1 x 64 int
	// 3. _cvgmOrientationPair
	// 4. _cvgmDescriptorPair

	//sample the nOctaves into 64 steps
	double scaleStep = pow(2.0, (double)(_nOctaves)/FREAK_NB_SCALES ); // 2 ^ ( (nOctaves-1) /nbScales)

	// pattern definition, radius normalized to 1.0 (outer point position+sigma=1.0)
	const int n[8] = {6,6,6,6,6,6,6,1}; // number of points on each concentric circle (from outer to inner) 43 in total
	const double bigR(2.0/3.0); // bigger radius
	const double smallR(2.0/24.0); // smaller radius
	const double unitSpace( (bigR-smallR)/21.0 ); // define spaces between concentric circles (from center to outer: 1,2,3,4,5,6)
	// radii of the concentric cirles (from outer to inner)
	const double radius[8] = {bigR, bigR-6*unitSpace, bigR-11*unitSpace, bigR-15*unitSpace, bigR-18*unitSpace, bigR-20*unitSpace, smallR, 0.0};
	// sigma of pattern points (each group of 6 points on a concentric circle has the same sigma)
	const double sigma[8] = {radius[0]/2.0, radius[1]/2.0, radius[2]/2.0,
		radius[3]/2.0, radius[4]/2.0, radius[5]/2.0,
		radius[6]/2.0, radius[6]/2.0
	};

	//calc standard pattern points
	_cvgmPatternLookup.create(1, FREAK_NB_SCALES*FREAK_NB_ORIENTATION*FREAK_NB_POINTS, CV_32FC3); //x,y,sigma each col is a point in pattern
	_cvgmPatternSize.create(1, FREAK_NB_SCALES, CV_32SC1);
	btl::device::freak::cudaBuildFreakPattern(patternScale,scaleStep,FREAK_NB_ORIENTATION,n,radius,sigma, &_cvgmPatternLookup,&_cvgmPatternSize);


	_cvgmOrientationPair.create(1,45,CV_32SC4);
	cv::Mat cvmOrientationPair; cvmOrientationPair.create(1,45,CV_32SC4);

	// build the list of orientation pairs
	cvmOrientationPair.ptr<int4>(0)[0].x=0; cvmOrientationPair.ptr<int4>(0)[0].y=3; cvmOrientationPair.ptr<int4>(0)[1].x=1; cvmOrientationPair.ptr<int4>(0)[1].y=4; cvmOrientationPair.ptr<int4>(0)[2].x=2; cvmOrientationPair.ptr<int4>(0)[2].y=5;
	cvmOrientationPair.ptr<int4>(0)[3].x=0; cvmOrientationPair.ptr<int4>(0)[3].y=2; cvmOrientationPair.ptr<int4>(0)[4].x=1; cvmOrientationPair.ptr<int4>(0)[4].y=3; cvmOrientationPair.ptr<int4>(0)[5].x=2; cvmOrientationPair.ptr<int4>(0)[5].y=4;
	cvmOrientationPair.ptr<int4>(0)[6].x=3; cvmOrientationPair.ptr<int4>(0)[6].y=5; cvmOrientationPair.ptr<int4>(0)[7].x=4; cvmOrientationPair.ptr<int4>(0)[7].y=0; cvmOrientationPair.ptr<int4>(0)[8].x=5; cvmOrientationPair.ptr<int4>(0)[8].y=1;

	cvmOrientationPair.ptr<int4>(0)[9].x=6; cvmOrientationPair.ptr<int4>(0)[9].y=9; cvmOrientationPair.ptr<int4>(0)[10].x=7; cvmOrientationPair.ptr<int4>(0)[10].y=10; cvmOrientationPair.ptr<int4>(0)[11].x=8; cvmOrientationPair.ptr<int4>(0)[11].y=11;
	cvmOrientationPair.ptr<int4>(0)[12].x=6; cvmOrientationPair.ptr<int4>(0)[12].y=8; cvmOrientationPair.ptr<int4>(0)[13].x=7; cvmOrientationPair.ptr<int4>(0)[13].y=9; cvmOrientationPair.ptr<int4>(0)[14].x=8; cvmOrientationPair.ptr<int4>(0)[14].y=10;
	cvmOrientationPair.ptr<int4>(0)[15].x=9; cvmOrientationPair.ptr<int4>(0)[15].y=11; cvmOrientationPair.ptr<int4>(0)[16].x=10; cvmOrientationPair.ptr<int4>(0)[16].y=6; cvmOrientationPair.ptr<int4>(0)[17].x=11; cvmOrientationPair.ptr<int4>(0)[17].y=7;

	cvmOrientationPair.ptr<int4>(0)[18].x=12; cvmOrientationPair.ptr<int4>(0)[18].y=15; cvmOrientationPair.ptr<int4>(0)[19].x=13; cvmOrientationPair.ptr<int4>(0)[19].y=16; cvmOrientationPair.ptr<int4>(0)[20].x=14; cvmOrientationPair.ptr<int4>(0)[20].y=17;
	cvmOrientationPair.ptr<int4>(0)[21].x=12; cvmOrientationPair.ptr<int4>(0)[21].y=14; cvmOrientationPair.ptr<int4>(0)[22].x=13; cvmOrientationPair.ptr<int4>(0)[22].y=15; cvmOrientationPair.ptr<int4>(0)[23].x=14; cvmOrientationPair.ptr<int4>(0)[23].y=16;
	cvmOrientationPair.ptr<int4>(0)[24].x=15; cvmOrientationPair.ptr<int4>(0)[24].y=17; cvmOrientationPair.ptr<int4>(0)[25].x=16; cvmOrientationPair.ptr<int4>(0)[25].y=12; cvmOrientationPair.ptr<int4>(0)[26].x=17; cvmOrientationPair.ptr<int4>(0)[26].y=13;

	cvmOrientationPair.ptr<int4>(0)[27].x=18; cvmOrientationPair.ptr<int4>(0)[27].y=21; cvmOrientationPair.ptr<int4>(0)[28].x=19; cvmOrientationPair.ptr<int4>(0)[28].y=22; cvmOrientationPair.ptr<int4>(0)[29].x=20; cvmOrientationPair.ptr<int4>(0)[29].y=23;
	cvmOrientationPair.ptr<int4>(0)[30].x=18; cvmOrientationPair.ptr<int4>(0)[30].y=20; cvmOrientationPair.ptr<int4>(0)[31].x=19; cvmOrientationPair.ptr<int4>(0)[31].y=21; cvmOrientationPair.ptr<int4>(0)[32].x=20; cvmOrientationPair.ptr<int4>(0)[32].y=22;
	cvmOrientationPair.ptr<int4>(0)[33].x=21; cvmOrientationPair.ptr<int4>(0)[33].y=23; cvmOrientationPair.ptr<int4>(0)[34].x=22; cvmOrientationPair.ptr<int4>(0)[34].y=18; cvmOrientationPair.ptr<int4>(0)[35].x=23; cvmOrientationPair.ptr<int4>(0)[35].y=19;

	cvmOrientationPair.ptr<int4>(0)[36].x=24; cvmOrientationPair.ptr<int4>(0)[36].y=27; cvmOrientationPair.ptr<int4>(0)[37].x=25; cvmOrientationPair.ptr<int4>(0)[37].y=28; cvmOrientationPair.ptr<int4>(0)[38].x=26; cvmOrientationPair.ptr<int4>(0)[38].y=29;
	cvmOrientationPair.ptr<int4>(0)[39].x=30; cvmOrientationPair.ptr<int4>(0)[39].y=33; cvmOrientationPair.ptr<int4>(0)[40].x=31; cvmOrientationPair.ptr<int4>(0)[40].y=34; cvmOrientationPair.ptr<int4>(0)[41].x=32; cvmOrientationPair.ptr<int4>(0)[41].y=35;
	cvmOrientationPair.ptr<int4>(0)[42].x=36; cvmOrientationPair.ptr<int4>(0)[42].y=39; cvmOrientationPair.ptr<int4>(0)[43].x=37; cvmOrientationPair.ptr<int4>(0)[43].y=40; cvmOrientationPair.ptr<int4>(0)[44].x=38; cvmOrientationPair.ptr<int4>(0)[44].y=41;

	cv::Mat cvmPatternLookup; _cvgmPatternLookup.download(cvmPatternLookup);//x,y,sigma,1xcols pattern points
	for( unsigned m = FREAK_NB_ORIENPAIRS; m--; ) {
		const float dx = cvmPatternLookup.ptr<float3>(0)[cvmOrientationPair.ptr<int4>(0)[m].x].x - cvmPatternLookup.ptr<float3>(0)[cvmOrientationPair.ptr<int4>(0)[m].y].x;
		const float dy = cvmPatternLookup.ptr<float3>(0)[cvmOrientationPair.ptr<int4>(0)[m].x].y - cvmPatternLookup.ptr<float3>(0)[cvmOrientationPair.ptr<int4>(0)[m].y].y;
		const float norm_sq = (dx*dx+dy*dy);
		cvmOrientationPair.ptr<int4>(0)[m].z = int((dx/(norm_sq))*4096.0+0.5);//quantization into 4096 levels
		cvmOrientationPair.ptr<int4>(0)[m].w = int((dy/(norm_sq))*4096.0+0.5);
	}
	//_cvgmOrientationPair.upload(cvmOrientationPair);

	// build the list of description pairs
	//_cvgmDescriptorPair.create(1,NB_PAIRS,CV_8UC2);
	cv::Mat cvmDescriptorPair;cvmDescriptorPair.create(1,NB_PAIRS,CV_8UC2);
	std::vector<DescriptionPair> allPairs;
	for( unsigned int i = 1; i < (unsigned int)FREAK_NB_POINTS; ++i ) {
		// (generate all the pairs)
		for( unsigned int j = 0; (unsigned int)j < i; ++j ) {
			DescriptionPair pair = {(uchar)i,(uchar)j};
			allPairs.push_back(pair);
		}
	}
	// Input vector provided
	if( !selectedPairs0.empty() ) {
		if( (int)selectedPairs0.size() == FREAK_NB_PAIRS ) {
			for( int i = 0; i < FREAK_NB_PAIRS; ++i ){
				cvmDescriptorPair.ptr<uchar2>(0)[i].x = allPairs[selectedPairs0.at(i)].i;
				cvmDescriptorPair.ptr<uchar2>(0)[i].y = allPairs[selectedPairs0.at(i)].j;
			}
		}
		else {
			CV_Error(CV_StsVecLengthErr, "Input vector does not match the required size");
		}
	}
	else { // default selected pairs
		for( int i = 0; i < FREAK_NB_PAIRS; ++i ){ // 512 in total
			cvmDescriptorPair.ptr<uchar2>(0)[i].x = allPairs[FREAK_DEF_PAIRS[i]].i;//assign precomputed default pairs
			cvmDescriptorPair.ptr<uchar2>(0)[i].y = allPairs[FREAK_DEF_PAIRS[i]].j;
		}
	}
	//_cvgmDescriptorPair.upload(cvmDescriptorPair);

	btl::device::freak::loadOrientationAndDescriptorPair( cvmOrientationPair.ptr<int4>() ,cvmDescriptorPair.ptr<uchar2>());
}

//************************************
// Method:    gpuCompute
// FullName:  btl::image::FREAK::gpuCompute
// Access:    public 
// Returns:   void
// Qualifier:
// Parameter: const Mat & image 
// Parameter: cv::gpu::GpuMat & cvgmKeyPoint_ is the predetecte keypoints
// Parameter: cv::gpu::GpuMat * pcvgmDescriptors_ 
//************************************

unsigned int FREAK::gpuCompute( const cv::gpu::GpuMat& cvgmImg_, const cv::gpu::GpuMat& cvgmImgInt_, cv::gpu::GpuMat& cvgmKeyPoint_,  cv::gpu::GpuMat* pcvgmDescriptor_ ) {

	if( cvgmImg_.empty() )		return 0;
	if( cvgmKeyPoint_.empty() )	return 0;

	btl::device::freak::loadGlobalConstantsImgResolution( cvgmImg_.rows, cvgmImg_.cols );

	//test::loadGlobalConstants( cvgmImg_.rows, cvgmImg_.cols, _nOctaves, sizeCst , FREAK_SMALLEST_KP_SIZE, FREAK_NB_POINTS, FREAK_NB_SCALES,
	//						   FREAK_NB_ORIENPAIRS, FREAK_NB_ORIENTATION,   FREAK_NB_PAIRS,  FREAK_LOG2);
	//test::loadOrientationAndDescriptorPair( cvmOrientationPair.ptr<int4>() ,cvmDescriptorPair.ptr<uchar2>());
	cv::gpu::ensureSizeIsEnough(cvgmKeyPoint_.cols,FREAK_NB_PAIRS/8,CV_8U,*pcvgmDescriptor_);
	cv::gpu::ensureSizeIsEnough(1,cvgmKeyPoint_.cols,CV_16SC1,_cvgmKpScaleIdx);

	_cvgmKpScaleIdx.create(1, cvgmKeyPoint_.cols, CV_16SC1);
	btl::device::freak::cudaComputeScaleIndex(_cvgmPatternSize, &cvgmKeyPoint_, &_cvgmKpScaleIdx);//int, float, short
	//cv::gpu::GpuMat cvgmKeyPointTest = cvgmKeyPoint_.clone();

	//pcvgmDescriptor_->create( cvgmKeyPoint_.cols,FREAK_NB_PAIRS/8, CV_8U ); 
	//pcvgmDescriptor_->setTo(0);
	unsigned int uT = btl::device::freak::cudaComputeFreakDescriptor( cvgmImg_, 
													cvgmImgInt_, 
													cvgmKeyPoint_, 
													_cvgmKpScaleIdx,
													_cvgmPatternLookup,
													_cvgmPatternSize,
													&(*pcvgmDescriptor_)
													);

	/*cv::gpu::GpuMat cvgmDescriptorTest;	cvgmDescriptorTest.create( cvgmKeyPoint_.cols,FREAK_NB_PAIRS/8, CV_8U ); 
	test::cudaComputeFreakDescriptor( cvgmImg_, 
										cvgmImgInt_, 
										cvgmKeyPointTest, 
										_cvgmKpScaleIdx,
										_cvgmPatternLookup,
										_cvgmPatternSize,
										&cvgmDescriptorTest
										);

	
	cv::Mat cvmKp; cvgmKeyPoint_.download(cvmKp);
	cv::Mat cvmKpTest; cvgmKeyPointTest.download(cvmKpTest);

	cv::Mat cvmDescriptor; (*pcvgmDescriptor_).download(cvmDescriptor);
	cv::Mat cvmDescriptorTest; cvgmDescriptorTest.download(cvmDescriptorTest);

	float fDDescriptor1 = 0;

	for (int c=0;c< cvmKp.cols; c++){
		if (cvmKp.ptr<float>(CSurf::HESSIAN_ROW)[c] < 0.f ) continue;
		//descriptor
		for (int i=0; i< cvmDescriptor.cols; i++){
			uchar c1 = cvmDescriptor.ptr<uchar>(c)[i];
			uchar c2 = cvmDescriptorTest.ptr<uchar>(c)[i];
			fDDescriptor1 += abs( c1 - c2 );
		}
	}
	cout << "the difference with test of descriptor =" << fDDescriptor1;*/

	return uT;
}//gpuCompute()

}//namespace image
}//namespace btl
