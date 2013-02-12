/* Written by ZG Tan to implement the prosac algorithm, based on OpenCV
  */

#include "prosac.h"
//#include "geometry.h"
#include <algorithm>
#include <vector>
#include <time.h>

using namespace std;

float FindHomography::ComputeReprojError(CvPoint2D32f point1, CvPoint2D32f point2, float* homography)
{
    float projectionPointZ = homography[6] * point1.x + homography[7] * point1.y + homography[8];
    float projectionPointX = (homography[0] * point1.x + homography[1] * point1.y + homography[2])/projectionPointZ;
    float projectionPointY = (homography[3] * point1.x + homography[4] * point1.y + homography[5])/projectionPointZ;

    return (float)(fabs( projectionPointX - point2.x) + fabs(projectionPointY - point2.y));
};

bool FindHomography::getSubset(vector<CvPoint2D32f> & pt1, vector<CvPoint2D32f> & pt2, int pool_size, int maxAttempts, int sample_size)
{
    int idx[4];
    int i = 0, j, idx_i, iters = 0;
    pt1.resize(sample_size);
    pt2.resize(sample_size);
    bool checkPartialSubsets = true;

    CvMat ms1 = cvMat(1, sample_size, CV_32FC2, &(pt1[0]));
    CvMat ms2 = cvMat(1, sample_size, CV_32FC2, &(pt2[0]));

    for(; iters < maxAttempts; iters++) {
        for( i = 0; i < sample_size && iters < maxAttempts; ) {
            idx[i] = idx_i = cvRandInt(&rng) % pool_size;
            for( j = 0; j < i; j++ )
                if( idx_i == idx[j] )
                    break;
            if( j < i )
                continue;

            int tempIndex = idx[j];
            pt1[i] = (*matchedPoints)[tempIndex].pointScene;
            pt2[i] = (*matchedPoints)[tempIndex].pointReference;

            if(checkPartialSubsets && (!checkSubset( &ms1, i+1 ) || !checkSubset( &ms2, i+1 ))) {
                iters++;
                continue;
            }
            i++;
        }
        if( !checkPartialSubsets && i == 4 &&
                (!checkSubset( &ms1, i ) || !checkSubset( &ms2, i )))
            continue;
        break;
    }

    return i == sample_size && iters < maxAttempts;
}


bool FindHomography::checkSubset( const CvMat* m, int count )
{
    int j, k, i, i0, i1;
    CvPoint2D32f* ptr = (CvPoint2D32f*)m->data.ptr;
    bool checkPartialSubsets = true;

    assert( CV_MAT_TYPE(m->type) == CV_32FC2 );

    if( checkPartialSubsets )
        i0 = i1 = count - 1;
    else
        i0 = 0, i1 = count - 1;

    for( i = i0; i <= i1; i++ ) {
        // check that the i-th selected point does not belong
        // to a line connecting some previously selected points
        for( j = 0; j < i; j++ ) {
            double dx1 = ptr[j].x - ptr[i].x;
            double dy1 = ptr[j].y - ptr[i].y;
            for( k = 0; k < j; k++ ) {
                double dx2 = ptr[k].x - ptr[i].x;
                double dy2 = ptr[k].y - ptr[i].y;
                if( fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
                    break;
            }
            if( k < j )
                break;
        }
        if( j < i )
            break;
    }

    return i >= i1;
}

int PROSACUpdateNumIters(double p, double ep, int model_points, int max_iters)
{
    double num, denom;
    p = MAX(p, 0.);
    p = MIN(p, 1.);
    ep = MAX(ep, 0.);
    ep = MIN(ep, 1.);

    // avoid inf's & nan's
    num = MAX(1. - p, DBL_MIN);
    denom = 1. - pow(1. - ep,model_points);
    num = log(num);
    denom = log(denom);

    int result = denom >= 0 || -num >= max_iters*(-denom) ? max_iters : cvRound(num/denom);
    return result;
}

bool fastReject(/*CvPoint src_corners[4],*/float h[9] )
{
    CvPoint src_corners[4] = {{0,0}, {284,0}, {284, 282}, {0, 282}};
    CvPoint dst_corners[4];
    for(int l = 0; l < 4; l++ )	{
        double x = src_corners[l].x, y = src_corners[l].y;
        double Z = 1./(h[6]*x + h[7]*y + h[8]);
        double X = (h[0]*x + h[1]*y + h[2])*Z;
        double Y = (h[3]*x + h[4]*y + h[5])*Z;
        dst_corners[l] = cvPoint(cvRound(X), cvRound(Y));
    }
    const double min_quad_size = 400;

    int length[4];
    for(int l = 0; l < 4; l++ )	{
        length[l] = abs(dst_corners[(l+1)%4].x - dst_corners[(l)%4].x) + abs(dst_corners[(l+1)%4].y - dst_corners[(l)%4].y);
        if (length[l]>500) {
            return true;
        }
    }
    //is the homography valid?
    /*if (!isTriangleClockwise(dst_corners)  / * ||!isQuadrilateralConvex(&dst_corners[0]) ||* / / *quadrilateralArea(&dst_corners[0]) < min_quad_size* /) { //is the homography valid?
        return true;*/
    //}

    return false;
}
inline float ptDistance(CvPoint2D32f &a,CvPoint2D32f &b)
{
	return abs(a.x-b.x)+abs(a.y-b.y);
}
bool FindPROSACHomography::Calculate()
{
    double t0 = (double)cvGetTickCount();

    float bestHomography[9];
    CvMat _bestH = cvMat(3, 3, CV_32F, bestHomography);
    CvMat _h = cvMat(3, 3, CV_32F, homography);

    std::vector<CvPoint2D32f> samplingObject(4);
    std::vector<CvPoint2D32f> samplingReference(4);
    CvMat samplingObjectPoints = cvMat(1, 4, CV_32FC2, &(samplingObject[0]));
    CvMat samplingReferencePoints = cvMat(1, 4, CV_32FC2, &(samplingReference[0]));

		std::vector<MatchedPoint> & matchs = *matchedPoints;
		
		//remove duplicate nodes;
/*
		sort(matchedPoints->begin(),matchedPoints->end(),CmpMatch());
		vector<int> tmp_idx; vector<int> tmp_matchedID;
		MatchedPoint mp;
		vector<MatchedPoint> refined_matches(0);
		size_t j=0;
		for (size_t i=0;i<matchs.size();i++)
		{
			float dist = ptDistance(mp.pointScene,matchs[i].pointScene) + ptDistance(mp.pointReference,matchs[i].pointReference);
			if (dist>reprojectionThreshold) 
			{
				mp = matchs[i];
				refined_matches.push_back(mp);
				j++;
			}else if(matchs[i].distance<mp.distance){
				mp = matchs[i];
				refined_matches.pop_back();
				refined_matches.push_back(mp);
			}
		}
		AttatchMatchedPoints(&refined_matches);
		matchs = refined_matches;
		double t1 = (double)cvGetTickCount();
*/
		//fprintf(stderr,"remove duplicates takes %6.2f ms. \n",(t1-t0)/cv::getTickFrequency()*1000);

		int bestCount = 0;
    int count = matchedPoints->size();
    if (count <4 ) return false;

/*
    if (count== 4) {
        for(int j=0; j<4; j++) {
            samplingObject[j] = matchs[j].pointScene;
            samplingReference[j] = matchs[j].pointReference;
        }
        // calculate homograpy
				if(!checkSubset(& samplingObjectPoints,4) || !checkSubset(& samplingReferencePoints,4)) return false;
        return cvFindHomography(&samplingReferencePoints, &samplingObjectPoints, &_h)>0;
    }
*/
		
    // sorting to give an increasing order of distances, thus correspondences with small distance would be sampled first.
    if (method == MY_PROSAC)
    {
			sort(matchedPoints->begin(), matchedPoints->end(), CompareDistanceLess());
    }
		
    int nTd;
		if (method == MY_PROSAC){
			nTd = count<50 ? count :50; // Td test as in <Td,d> lo-RanSAC
		}else{
			nTd = count;
		}
		
    maxIteration = 500;
    int m =4; //minimal sample size;
    int samplingCount = m;
    double Tn1 = 1;
    int T_n_prime = 1, T_n1_prime;
    double Tn = maxIteration;        // maximum number of ransacs.
    for(int i=0; i<m; i++) {
        Tn *= (samplingCount-i)/(double)(count-i);
    }

    int nValidModels = 0;
    int maxIter = PROSACUpdateNumIters(confidence, (double)(nTd - 4)/(double)nTd, 4, maxIteration);
    int i;
    for(i=1; i<maxIter; i++) {
				
			int inlinerCount = 0;
			if (method == MY_PROSAC)
			{
				if(samplingCount<count && i>T_n_prime ) {
				Tn1				=	Tn * (double)(samplingCount + 1) / (double)(samplingCount + 1 - m);
				T_n_prime =	T_n_prime + ceil(Tn1 - Tn);
				Tn				=	Tn1;
				samplingCount++;
			}

					if (i<T_n_prime) {
						getSubset(samplingObject,samplingReference,samplingCount-1,300,3);
						int tempIndex = samplingCount-1;
						samplingObject.push_back( matchs[tempIndex].pointScene);
						samplingReference.push_back( matchs[tempIndex].pointReference);
						if(!checkSubset(& samplingObjectPoints,4) || !checkSubset(& samplingReferencePoints,4)) continue;

					} else {
						if(!getSubset(samplingObject,samplingReference,samplingCount,300,4)) continue;
					}
				}else if (method == MY_RANSAC)
				{
					if(!getSubset(samplingObject,samplingReference,nTd,300,4)) continue;
				}

        // calculate homograpy
        cvFindHomography(&samplingReferencePoints, &samplingObjectPoints, &_h);
				
				//if (nTd>30)
				if(fastReject(/*corners,*/ homography)) continue; //if the

				// calculate consensus set
        for(int j=0; j<nTd; j++) {
            float error = ComputeReprojError(matchs[j].pointReference, matchs[j].pointScene, homography);
            if(error < this->reprojectionThreshold) {
                inlinerCount++;
            }
        }
        if(inlinerCount > bestCount) {
            bestCount = inlinerCount;
            for(int k=0; k<9; k++)
							bestHomography[k] = homography[k];

            maxIter = PROSACUpdateNumIters(this->confidence, (double)(nTd - inlinerCount)/(double)nTd, 4, maxIteration);
        }
    }

    // terminate
    if(bestCount >= 4) {
        //printf("number of trials by RANSAC: %d \n",i);
			//remove duplicate nodes;
			sort(matchedPoints->begin(),matchedPoints->begin()+nTd,CmpMatch());
			vector<int> tmp_idx; vector<int> tmp_matchedID;
			MatchedPoint mp;
			vector<MatchedPoint> refined_matches(0);
			size_t j=0;
			for (size_t i=0;i<nTd;i++)
			{
				float dist = ptDistance(mp.pointScene,matchs[i].pointScene) + ptDistance(mp.pointReference,matchs[i].pointReference);
				if (dist>reprojectionThreshold) 
				{
					mp = matchs[i];
					refined_matches.push_back(mp);
					j++;
				}else if(matchs[i].distance<mp.distance){
					mp = matchs[i];
					refined_matches.pop_back();
					refined_matches.push_back(mp);
				}
			}
			matchs = refined_matches;
			count = refined_matches.size();
			nTd = count<50 ? count :50; // Td test as in <Td,d> lo-RanSAC

        int nInliers = 0;
        for(int j=0; j<nTd; j++) {
            float error = ComputeReprojError(matchs[j].pointReference, matchs[j].pointScene, bestHomography);
            if(error < this->reprojectionThreshold) {
                matchs[j].isInlier = true;
                nInliers++;
            } else {
                matchs[j].isInlier = false;
            }
        }

        std::vector<CvPoint2D32f> consensusObject(nInliers);
        std::vector<CvPoint2D32f> consensusReference(nInliers);

				if (consensusObject.size()<4) return false;

        int index = 0;
        for(int j=0; j<nTd; j++){
            if(matchs[j].isInlier){
                consensusObject[index]		= matchs[j].pointScene;
                consensusReference[index]	= matchs[j].pointReference;
                index++;
            }
        }
        CvMat ObjectPointsMat = cvMat(1, consensusObject.size(), CV_32FC2, &(consensusObject[0]));
        CvMat ReferencePointsMat = cvMat(1, consensusReference.size(), CV_32FC2, &(consensusReference[0]));
        
        cvFindHomography(&ReferencePointsMat, &ObjectPointsMat, &_h);
        return true;
    }

    return false;
}

bool find_homography(vector<CvPoint2D32f> & src, vector<CvPoint2D32f> & des, vector<double> & h, int method, const vector<float> & feature_distances)
{
    CvMat _pt1, _pt2;
    const double tolerance = 5;
    int n = src.size();

    _pt1 = cvMat(1, n, CV_32FC2, &src[0] );
    _pt2 = cvMat(1, n, CV_32FC2, &des[0] );
    h.resize(9);
    CvMat _h = cvMat(3, 3, CV_64F, &h[0]);
    bool ret =false;

    //if (n<20 && method == MY_PROSAC)	method = CV_RANSAC; //no need to use ProSAC, when number of matches is low.

    switch (method) {
    case CV_RANSAC:
    case CV_LMEDS:
        ret = cvFindHomography( &_pt1, &_pt2, &_h,	method, tolerance );
        break;
		case MY_PROSAC:
			{
				//if (n<30) return false;
			
        FindPROSACHomography prosac;
        vector<MatchedPoint> matchedPoints;
        matchedPoints.resize(n);

        for(int k=0; k<n; k++ ) {
            CvPoint2D32f ref, scene;
            ref = src[k];
            scene = des[k];
            float dist = feature_distances[k];
            MatchedPoint pt(scene,ref,dist);
            matchedPoints[k] = pt;
        }
        prosac.confidence = 0.99;
        prosac.AttatchMatchedPoints(&matchedPoints);
				prosac.method = MY_PROSAC;
        ret = prosac.Calculate();
        for (int k = 0; k<9; k++) h[k] = (double)prosac.homography[k];
				}
        break;
		case MY_RANSAC:
			{
			FindPROSACHomography prosac;
			vector<MatchedPoint> matchedPoints;
			matchedPoints.resize(n);

			for(int k=0; k<n; k++ ) {
				CvPoint2D32f ref, scene;
				ref = src[k];
				scene = des[k];
				float dist = feature_distances[k];
				MatchedPoint pt(scene,ref,dist);
				matchedPoints[k] = pt;
			}
			prosac.confidence = 0.99;
			prosac.method = MY_RANSAC;
			prosac.AttatchMatchedPoints(&matchedPoints);
			ret = prosac.Calculate();
			for (int k = 0; k<9; k++) h[k] = (double)prosac.homography[k];
			}
			break;
			
    }

    return ret;

}

