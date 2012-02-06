#define INFO

//#include "boost/date_time/posix_time/posix_time.hpp"
#include <iostream>

//using namespace boost::posix_time;
//using namespace boost::gregorian;
using namespace std;

#include "../cuda/CudaLib.h"
#include "../OtherUtil.hpp"

int testCuda()
{
	PRINTSTR("test Cuda: cudaDepth2Disparity() && cudaDisparity2Depth()");
	const int nRow = 4;
	const int nCol = 6;
	const int N = nRow*nCol;
    float *pDepth, *pDisparity, *pDepthResult;

    // allocate the memory on the CPU
	pDepth     = (float*)malloc( N * sizeof(float) );
	pDisparity = (float*)malloc( N * sizeof(float) );
	pDepthResult = (float*)malloc( N * sizeof(float) );

    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++) {
        pDepth[i] = i;
        pDisparity[i] = 0;
    }

    //time_duration cTD0;
    //ptime cT0 ( microsec_clock::local_time() );

    btl::cuda_util::cudaDepth2Disparity( pDepth, nRow, nCol, pDisparity ); 
	btl::cuda_util::cudaDisparity2Depth( pDisparity, nRow, nCol, pDepthResult ); 
    //ptime cT1 ( microsec_clock::local_time() );
    //time_duration cTDAll = cT1 - cT0 ;

    //cout << " Overall            = " << cTDAll << endl;
    
    // verify that the GPU did the work we requested
    bool success = true;
    for (int i=0; i<N; i++) {
		cout << pDepth[i] << " + " << pDisparity[i] <<  " + " << pDepthResult[i] << endl; 
    }
    
    //ptime cT2 ( microsec_clock::local_time() );
    //time_duration cTDP = cT2 - cT1 ;

    //cout << " Check result       = " << cTDP << endl;

    // free the memory we allocated on the CPU
    free( pDepth );
    free( pDisparity );

    return 0;
}
