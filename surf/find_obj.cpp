/*
 * A Demo to OpenCV Implementation of SURF
 * Further Information Refer to "SURF: Speed-Up Robust Feature"
 * Author: Liu Liu
 * liuliu.1987+opencv@gmail.com
 */
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <btl/Utility/Converters.hpp>


#include <iostream>
#include <vector>
using namespace btl; //for "<<" operator
using namespace utility;

using namespace std;

void findPairs( const vector< cv::KeyPoint >& vObjectKeyPoints_, const vector< float >& vObjectDescriptors_ , const vector< cv::KeyPoint >& vImageKeyPoints_, const vector< float >& vImageDescriptors_, vector< int >* pvPtPairs_ )
{
	//conver vector< float > to cv::Mat_<float> format of descriptors
	int nSizeImage = vImageKeyPoints_.size();
	int nLengthDescriptorImage = vImageDescriptors_.size() / nSizeImage;
	cv::Mat cvmImageDescriptors( nSizeImage, nLengthDescriptorImage, CV_32F );
	float* pImage = cvmImageDescriptors.ptr<float>(0);
    for(vector< float >::const_iterator cit_ImageDescriptor = vImageDescriptors_.begin(); cit_ImageDescriptor!=vImageDescriptors_.end(); cit_ImageDescriptor++)
    {
		*pImage++ = *cit_ImageDescriptor;
    }
	//construct reference
    cv::flann::Index cFlannIndex(cvmImageDescriptors, cv::flann::KDTreeIndexParams(4));  // using 4 randomized kdtrees

	//conver vector< float > to cv::Mat_<float> format of descriptors
	int nSizeObject = vObjectKeyPoints_.size();
	int nLengthDescriptorObject = vObjectDescriptors_.size() / nSizeObject;
	cv::Mat cvmObjectDescriptors( nSizeObject, nLengthDescriptorObject, CV_32F );
	float* pObject = cvmObjectDescriptors.ptr<float>(0);
    for(vector< float >::const_iterator cit_ObjectDescriptor = vObjectDescriptors_.begin(); cit_ObjectDescriptor!=vObjectDescriptors_.end(); cit_ObjectDescriptor++)
    {
		*pObject++ = *cit_ObjectDescriptor;
    }

	// find nearest neighbors using FLANN
    cv::Mat cvmIndices(nSizeObject, 2, CV_32S);
    cv::Mat cvmDists  (nSizeObject, 2, CV_32F);
    cFlannIndex.knnSearch(cvmObjectDescriptors, cvmIndices, cvmDists, 2, cv::flann::SearchParams(64) ); // maximum number of leafs checked
	cout << " new " << cvmIndices.rows << endl;
    int* pIndices = cvmIndices.ptr<int>(0);
    float* pDists = cvmDists.ptr<float>(0);
    for (int i=0;i<cvmIndices.rows;++i) {
    	if (pDists[2*i]<0.6*pDists[2*i+1]) {
    		pvPtPairs_->push_back(i);
    		pvPtPairs_->push_back(pIndices[2*i]);
    	}
    }
	return;
}

void findPairs(  const cv::Mat& cvmObjectDescriptors_ , const cv::Mat& cvmImageDescriptors_, vector< int >* pvPtPairs_ )
{
	//construct reference
    cv::flann::Index cFlannIndex(cvmImageDescriptors_, cv::flann::KDTreeIndexParams(4));  // using 4 randomized kdtrees

	// find nearest neighbors using FLANN
	int nSizeObject = cvmObjectDescriptors_.rows;
    cv::Mat cvmIndices(nSizeObject, 2, CV_32S);
    cv::Mat cvmDists  (nSizeObject, 2, CV_32F);
    cFlannIndex.knnSearch(cvmObjectDescriptors_, cvmIndices, cvmDists, 2, cv::flann::SearchParams(64) ); // maximum number of leafs checked
    int* pIndices = cvmIndices.ptr<int>(0);
    float* pDists = cvmDists.ptr<float>(0);
    for (int i=0;i<cvmIndices.rows;++i) {
    	if (pDists[2*i]<0.6*pDists[2*i+1]) {
    		pvPtPairs_->push_back(i); //obj index
    		pvPtPairs_->push_back(pIndices[2*i]); //img index>
    	}
    }
	return;
}


int main(int argc, char** argv)
{
    const char* object_filename = argc == 3 ? argv[1] : "box.png";
    const char* scene_filename = argc == 3 ? argv[2] : "box_in_scene.png";

    static CvScalar colors[] = 
    {
        {{0,0,255}},
        {{0,128,255}},
        {{0,255,255}},
        {{0,255,0}},
        {{255,128,0}},
        {{255,255,0}},
        {{255,0,0}},
        {{255,0,255}},
        {{255,255,255}}
    };
	cv::Mat cvmObject = cv::imread( object_filename );
	cv::Mat cvmImage  = cv::imread( scene_filename );
	cv::Mat cvmGrayObject;
	cv::Mat cvmGrayImage;
	PRINT( cvmObject.rows );
	PRINT( cvmObject.cols );
	cv::cvtColor( cvmObject, cvmGrayObject, CV_BGR2GRAY );
	cv::cvtColor( cvmImage,  cvmGrayImage,  CV_BGR2GRAY );
	cv::Mat cvmMask;
	/*
//surf
	vector<cv::KeyPoint> vObjectKeyPoints,  vImageKeyPoints;
	vector<float>        vObjectDescriptors,vImageDescriptors;

	cv::SURF cSurf( 500, 4, 2, true );
    cSurf( cvmGrayObject, cvmMask, vObjectKeyPoints, vObjectDescriptors );
	cSurf( cvmGrayImage,  cvmMask, vImageKeyPoints,  vImageDescriptors  );

	cout << "cSurf() Object:" << vObjectKeyPoints.size() << endl;
	cout << "cSurf() Image: " << vImageKeyPoints.size()  << endl;

	vector<int> PtPairs;
	findPairs( vObjectKeyPoints, vObjectDescriptors, vImageKeyPoints, vImageDescriptors, &PtPairs );
*/

//sift
	cv::SIFT cSift( 0.04, 10. );

	cv::Mat cvmObjectDescriptors, cvmImageDescriptors;
	vector<cv::KeyPoint> vObjectKeyPointsSift, vImageKeyPointsSift;

	cSift( cvmGrayObject, cvmMask, vObjectKeyPointsSift, cvmObjectDescriptors);
	cSift( cvmGrayImage,  cvmMask, vImageKeyPointsSift,  cvmImageDescriptors);

	PRINT( cvmObjectDescriptors.rows );
	PRINT( cvmImageDescriptors.rows );

	cv::namedWindow ( "myObj", 1 );
    while ( true )
    {
		for(int i = 0; i < vObjectKeyPointsSift.size(); i++ )
	    {
	        int radius = cvRound(vObjectKeyPointsSift[i].size*1.2/9.*2);
			cv::circle( cvmObject, vObjectKeyPointsSift[i].pt, radius, colors[0], 1, 8, 0 );
	    }	

        cv::imshow ( "myObj", cvmObject );
		int nKey = cv::waitKey ( 30 );
		if ( nKey == 27 )
		{
			break;
		}
    }

	vector<int> PtPairs;
	findPairs( cvmObjectDescriptors, cvmImageDescriptors, &PtPairs );

	cout << " PtPairs.size()/2 = " << PtPairs.size()/2 << endl;

    //for display
	cv::Mat cvmCorr(cvmImage.rows + cvmObject.rows,cvmImage.cols, CV_8UC1 );
	cv::Mat cvmCorr2(cvmImage.rows + cvmObject.rows,cvmImage.cols, CV_8UC1 );

	cv::Mat roi1(cvmCorr, cv::Rect(0,0,cvmObject.cols, cvmObject.rows));
	cvmGrayObject.copyTo(roi1);
	cv::Mat roi2(cvmCorr, cv::Rect(0,cvmObject.rows, cvmImage.cols, cvmImage.rows ));
	cvmGrayImage.copyTo(roi2);

	int i=0;
	int nKey;
	cv::namedWindow ( "myWindow", 1 );
    while ( true )
    {
		cvmCorr.copyTo( cvmCorr2 );
		//cv::line( cvmCorr2, vObjectKeyPoints[ PtPairs[i] ].pt, cv::Point(vImageKeyPoints [ PtPairs[i+1] ].pt.x,vImageKeyPoints [ PtPairs[i+1] ].pt.y+cvmGrayObject.rows ), colors[7] );
		cv::line( cvmCorr2, vObjectKeyPointsSift[ PtPairs[i] ].pt, cv::Point(vImageKeyPointsSift [ PtPairs[i+1] ].pt.x,vImageKeyPointsSift [ PtPairs[i+1] ].pt.y+cvmGrayObject.rows ), colors[7] );


        cv::imshow ( "myWindow", cvmCorr2 );
		nKey = cv::waitKey ( 30 );
		if ( nKey == 32 )
        {
            i+=2; 
			if( i > PtPairs.size() )
				break;
        }
		if ( nKey == 27 )
		{
			break;
		}
    }



    return 0;
}
