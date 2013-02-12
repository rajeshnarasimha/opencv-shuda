/* ========================================================================
 * ======================================================================== */

#ifndef _FIND_PROSAC_HOMOGRAPY_H_
#define _FIND_PROSAC_HOMOGRAPY_H_

#include <vector>
#include <cxcore.h>
#include <cv.h>

#define MY_PROSAC 16
#define MY_RANSAC 32

using namespace std;

typedef struct _MatchedPoint {
    CvPoint2D32f pointScene;
    CvPoint2D32f pointReference;
    float distance;
    bool isInlier;

    _MatchedPoint(CvPoint2D32f pointScene, CvPoint2D32f pointReference, float distance = 0.0) {
        this->pointScene = pointScene;
        this->pointReference = pointReference;

        this->distance = distance;
        isInlier = false;
    }
    _MatchedPoint() {
        this->distance = 10000000;
        isInlier = false;
    }

    void operator=(_MatchedPoint oprd) {
        this->pointScene = oprd.pointScene;
        this->pointReference = oprd.pointReference;
        this->distance = oprd.distance;
        this->isInlier = oprd.isInlier;
    }

    bool operator==(_MatchedPoint oprd) {
        return this->distance == oprd.distance;
    }
    bool operator<( _MatchedPoint oprd)	{
        return this->distance < oprd.distance;
    }
} MatchedPoint;

class FindHomography
{
public:
    float reprojectionThreshold;
    float homography[9];
    CvRNG rng;

    std::vector<MatchedPoint>* matchedPoints;
public:
    FindHomography() {
        reprojectionThreshold = 5.0;
        matchedPoints = NULL;
        rng = cvRNG( cvGetTickCount());
        //rng = cvRNG(-1);
    }
    ~FindHomography() {
    }

    inline void SetReprojectionThreshold(float reprojectionThreshold=5.0f) {
        this->reprojectionThreshold = reprojectionThreshold;
    };

    inline void AttatchMatchedPoints(std::vector<MatchedPoint>* matchedPoints) {
        this->matchedPoints = matchedPoints;
    };
    inline float* GetHomography() {
        return this->homography;
    };

    static float ComputeReprojError(CvPoint2D32f point1, CvPoint2D32f point2, float* homography);

    virtual bool Calculate() = NULL;
    //virtual bool fastReject();

public:
    bool getSubset( vector<CvPoint2D32f> & pt1, vector<CvPoint2D32f> & pt2, int pool_size = 2808, int maxAttempts = 200,int sample_size = 4);
    bool checkSubset( const CvMat* m, int count );
};


typedef struct _CompareDistanceLess {
    bool operator()(const struct _MatchedPoint& p, const struct _MatchedPoint& q) const {
        return p.distance < q.distance;
    }
} CompareDistanceLess;

typedef struct _cmpMatch {
	bool operator()(const struct _MatchedPoint& a, const struct _MatchedPoint& b) const {
		CvPoint a_s = cvPointFrom32f(a.pointScene);
		CvPoint a_r = cvPointFrom32f(a.pointReference);
		CvPoint b_s = cvPointFrom32f(b.pointScene);
		CvPoint b_r = cvPointFrom32f(b.pointReference);

		if (a_s.x <b_s.x){
			return true;
		}else if (a_s.x == b_s.x){
			if (a_s.y < b_s.y){
				return true;
			}else if (a_s.y == b_s.y){
				if (a_r.x <b_r.x){
					return true;
				}else if (a_r.x == b_r.x){
					if (a_r.y < b_r.y){
						return true;
					}else if (a_r.y == b_r.y){
						return a.distance<b.distance;
					}
				}
			}
		}
		return false;
	}

} CmpMatch;

class FindPROSACHomography:public FindHomography
{
public:
    float confidence;
    int maxIteration;
    double timeout;
		int method;

public:
    FindPROSACHomography() {
        this->confidence = 0.99f; // constant
        SetTimeout(5.0);

        this->maxIteration = 2000;
        this->reprojectionThreshold = 5.0f;
        for(int i=0; i<9; i++)
            homography[i] = 0.0f;

        this->matchedPoints = NULL;
				method = MY_PROSAC;
    }
    ~FindPROSACHomography() {

    }

    inline void SetMaxIteration(int iteration=1000) {
        this->maxIteration = iteration;
    };
    inline void SetTimeout(double ms) {
        this->timeout = ms * 1000.0 * cvGetTickFrequency();
    };

public:
    bool Calculate();
};

bool find_homography(vector<CvPoint2D32f> & src, vector<CvPoint2D32f> & des, vector<double> & h, int method = CV_RANSAC, const vector<float> & feature_distances = vector<float>());

bool inline find_homography(vector<CvPoint2D32f> & src, vector<CvPoint2D32f> & des, double * h, int method = CV_RANSAC, const vector<float> & feature_distances = vector<float>())
{
    vector<double> homography;
    bool ret = find_homography(src,des,homography,method,feature_distances);
    memcpy(h,&homography[0],9*sizeof(double));
    return ret;
};


#endif