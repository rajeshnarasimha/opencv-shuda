#include <vector>
#include <opencv2/gpu/gpu.hpp>
#include "Fast.h"
#include "Orb.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <npp.h>


namespace btl { namespace device {  namespace orb  {
	//use thrust library to sort fast corners according to their response
	int thrustSortFastCornersAndCull(int* pnLoc_, float* pfResponse_, const int nSize_, const int nPoints_);

	void cudaCalcHarrisResponses(cv::gpu::PtrStepSzb img, const short2* loc, float* response, const int npoints, int nBlockSize_, float harris_k, cudaStream_t stream);

	void loadUMax(const int* u_max, int count);

	void IC_Angle_gpu(cv::gpu::PtrStepSzb image, const short2* loc, float* angle, int npoints, int usHalfPatch_, cudaStream_t stream);

	void cudaComputeOrbDescriptor(cv::gpu::PtrStepb cvgmImg_, const short2* pLoc_, const float* pAngle_, const int nPoints_,
		const int* nPatternX_, const int* nPatternY_, cv::gpu::PtrStepb cvgmDescriptor_, int nDescriptorSize_, int WTA_K, cudaStream_t stream);

	void cudaMergeLocation(const short2* s2Loc_, float* pfX_, float* pfY_, const int nPoints_, const float fScale_, cudaStream_t stream);
} //namespace orb
} //namespace device
} //namespace btl

namespace btl{ namespace image{

const float HARRIS_K = 0.04f;
const int DESCRIPTOR_SIZE = 32;

const int bit_pattern_31_[256 * 4] =
{
	8,-3, 9,5/*mean (0), correlation (0)*/,
	4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
	-11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
	7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
	2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
	1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
	-2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
	-13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
	-13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
	10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
	-13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
	-11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
	7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
	-4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
	-13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
	-9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
	12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
	-3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
	-6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
	11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
	4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
	5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
	3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
	-8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
	-2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
	-13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
	-7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
	-4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
	-10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
	5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
	5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
	1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
	9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
	4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
	2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
	-4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
	-8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
	4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
	0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
	-13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
	-3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
	-6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
	8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
	0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
	7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
	-13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
	10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
	-6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
	10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
	-13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
	-13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
	3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
	5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
	-1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
	3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
	2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
	-13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
	-13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
	-13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
	-7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
	6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
	-9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
	-2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
	-12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
	3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
	-7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
	-3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
	2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
	-11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
	-1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
	5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
	-4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
	-9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
	-12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
	10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
	7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
	-7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
	-4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
	7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
	-7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
	-13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
	-3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
	7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
	-13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
	1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
	2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
	-4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
	-1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
	7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
	1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
	9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
	-1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
	-13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
	7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
	12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
	6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
	5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
	2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
	3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
	2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
	9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
	-8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
	-11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
	1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
	6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
	2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
	6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
	3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
	7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
	-11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
	-10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
	-5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
	-10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
	8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
	4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
	-10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
	4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
	-2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
	-5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
	7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
	-9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
	-5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
	8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
	-9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
	1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
	7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
	-2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
	11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
	-12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
	3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
	5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
	0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
	-9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
	0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
	-1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
	5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
	3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
	-13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
	-5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
	-4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
	6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
	-7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
	-13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
	1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
	4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
	-2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
	2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
	-2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
	4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
	-6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
	-3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
	7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
	4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
	-13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
	7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
	7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
	-7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
	-8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
	-13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
	2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
	10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
	-6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
	8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
	2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
	-11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
	-12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
	-11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
	5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
	-2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
	-1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
	-13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
	-10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
	-3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
	2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
	-9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
	-4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
	-4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
	-6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
	6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
	-13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
	11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
	7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
	-1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
	-4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
	-7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
	-13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
	-7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
	-8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
	-5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
	-13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
	1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
	1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
	9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
	5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
	-1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
	-9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
	-1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
	-13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
	8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
	2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
	7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
	-10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
	-10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
	4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
	3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
	-4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
	5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
	4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
	-9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
	0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
	-12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
	3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
	-10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
	8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
	-8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
	2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
	10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
	6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
	-7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
	-3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
	-1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
	-3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
	-8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
	4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
	2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
	6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
	3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
	11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
	-3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
	4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
	2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
	-10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
	-13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
	-13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
	6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
	0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
	-13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
	-9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
	-13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
	5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
	2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
	-1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
	9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
	11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
	3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
	-1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
	3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
	-13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
	5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
	8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
	7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
	-10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
	7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
	9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
	7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
	-1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};

void initializeOrbPattern(const cv::Point* pPointsPattern_, cv::Mat& cvmPattern_, int nTuples_, int nTupleSize_, int nPoolSize_)
{
	cv::RNG rng(0x12345678);

	cvmPattern_.create(2, nTuples_ * nTupleSize_, CV_32SC1);
	cvmPattern_.setTo(cv::Scalar::all(0));

	int* pattern_x_ptr = cvmPattern_.ptr<int>(0);
	int* pattern_y_ptr = cvmPattern_.ptr<int>(1);

	for (int i = 0; i < nTuples_; i++)
	{
		for (int k = 0; k < nTupleSize_; k++)
		{
			for(;;)
			{
				int idx = rng.uniform(0, nPoolSize_);
				cv::Point pt = pPointsPattern_[idx];

				int k1;
				for (k1 = 0; k1 < k; k1++)
					if (pattern_x_ptr[nTupleSize_ * i + k1] == pt.x && pattern_y_ptr[nTupleSize_ * i + k1] == pt.y)
						break;

				if (k1 == k)
				{
					pattern_x_ptr[nTupleSize_ * i + k] = pt.x;
					pattern_y_ptr[nTupleSize_ * i + k] = pt.y;
					break;
				}
			}
		}
	}
}

void makeRandomPattern(int patchSize, cv::Point* pattern, int npoints)
{
	// we always start with a fixed seed,
	// to make patterns the same on each run
	cv::RNG rng(0x34985739);

	for (int i = 0; i < npoints; i++)
	{
		pattern[i].x = rng.uniform(-patchSize / 2, patchSize / 2 + 1);
		pattern[i].y = rng.uniform(-patchSize / 2, patchSize / 2 + 1);
	}
}



CORB::CORB(int nFeatures, float scaleFactor, int nLevels, int edgeThreshold, int firstLevel, int WTA_K, int scoreType, int patchSize) :
_nFeatures(nFeatures), _fScaleFactor(scaleFactor), _nLevels(nLevels), _nEdgeThreshold(edgeThreshold), _nFirstLevel(firstLevel), _nWTA_K(WTA_K),
	_nScoreType(scoreType), _nPatchSize(patchSize),
	_fastDetector(DEFAULT_FAST_THRESHOLD)
{
	CV_Assert(_nPatchSize >= 2);

	// fill the extractors and pcvgmDescriptors_ for the corresponding scales
	float factor = 1.0f / _fScaleFactor;
	float n_desired_features_per_scale = _nFeatures * (1.0f - factor) / (1.0f - std::pow(factor, _nLevels));

	_vFeaturesPerLevel.resize(_nLevels);
	size_t sum_n_features = 0;
	for (int level = 0; level < _nLevels - 1; ++level)
	{
		_vFeaturesPerLevel[level] = cvRound(n_desired_features_per_scale);
		sum_n_features += _vFeaturesPerLevel[level];
		n_desired_features_per_scale *= factor;
	}
	_vFeaturesPerLevel[_nLevels - 1] = nFeatures - sum_n_features;

	// pre-compute the end of a row in a circular patch (1/4 of the circular patch)
	int half_patch_size = _nPatchSize / 2;
	std::vector<int> u_max(half_patch_size + 2);
	for (int v = 0; v <= half_patch_size * std::sqrt(2.f) / 2 + 1; ++v)
		u_max[v] = cvRound(std::sqrt(static_cast<float>(half_patch_size * half_patch_size - v * v)));

	// Make sure we are symmetric
	for (int v = half_patch_size, v_0 = 0; v >= half_patch_size * std::sqrt(2.f) / 2; --v)
	{
		while (u_max[v_0] == u_max[v_0 + 1])
			++v_0;
		u_max[v] = v_0;
		++v_0;
	}
	CV_Assert(u_max.size() < 32);
	btl::device::orb::loadUMax(&u_max[0], static_cast<int>(u_max.size()));

	// Calc cvmPattern_
	const int npoints = 512; // 256 tests and each test requires 2 points 256x2 = 512
	cv::Point pattern_buf[npoints];
	const cv::Point* pPointsPattern = (const cv::Point*)bit_pattern_31_;
	if (_nPatchSize != 31)
	{
		pPointsPattern = pattern_buf;
		makeRandomPattern(_nPatchSize, pattern_buf, npoints);
	}

	CV_Assert(_nWTA_K == 2 || _nWTA_K == 3 || _nWTA_K == 4);

	cv::Mat cvmPattern; //2 x n : 1st row is x and 2nd row is y; test point1, test point2;

	if (_nWTA_K == 2)
	{ //assign cvmPattern_ from precomputed patterns
		cvmPattern.create(2, npoints, CV_32SC1);

		int* pattern_x_ptr = cvmPattern.ptr<int>(0); //get the 1st row
		int* pattern_y_ptr = cvmPattern.ptr<int>(1); //get the 2nd row

		for (int i = 0; i < npoints; ++i)
		{
			pattern_x_ptr[i] = pPointsPattern[i].x;
			pattern_y_ptr[i] = pPointsPattern[i].y;
		}
	}
	else
	{ //calc the cvmPattern_ according to _nWTA_K, descriptor size
		int ntuples = descriptorSize() * 4; // 32 Byte x 4 = 128 
		initializeOrbPattern(pPointsPattern, cvmPattern, ntuples, _nWTA_K, npoints);
	}

	_cvgmPattern.upload(cvmPattern);//2 x n : 1st row is x and 2nd row is y; test point1, test point2;

	_pBlurFilter = cv::gpu::createGaussianFilter_GPU(CV_8UC1, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

	_bBlurForDescriptor = false;
}//COrb


namespace
{
	inline float getScale(float scaleFactor, int firstLevel, int level)
	{
		return pow(scaleFactor, level - firstLevel);
	}

	//takes cvgmKeyPoints_ and culls them by the response
	//nCornerBeforeCulling_ 
	void sortAndCull(cv::gpu::GpuMat& cvgmKeyPoints_, int nCornerAfterCulling_, int* pnCornerBeforeCulling_ )
	{
		using namespace btl::device::orb;

		//this is only necessary if the cvgmKeyPoints_ size is greater than the number of desired points.
		if (*pnCornerBeforeCulling_ > nCornerAfterCulling_){
			if (nCornerAfterCulling_ == 0){
				cvgmKeyPoints_.release();
				return;
			}

			*pnCornerBeforeCulling_ = thrustSortFastCornersAndCull(cvgmKeyPoints_.ptr<int>(cv::gpu::FAST_GPU::LOCATION_ROW), cvgmKeyPoints_.ptr<float>(cv::gpu::FAST_GPU::RESPONSE_ROW), *pnCornerBeforeCulling_, nCornerAfterCulling_);
		}
	}
}

void CORB::buildScalePyramids(const cv::gpu::GpuMat& image, const cv::gpu::GpuMat& mask)
{
	CV_Assert(image.type() == CV_8UC1);
	CV_Assert(mask.empty() || (mask.type() == CV_8UC1 && mask.size() == image.size()));

	_vcvgmImagePyr.resize(_nLevels);
	_vcvgmMaskPyr.resize(_nLevels);

	for (int level = 0; level < _nLevels; ++level)
	{
		float scale = 1.0f / getScale(_fScaleFactor, _nFirstLevel, level);

		cv::Size sz(cvRound(image.cols * scale), cvRound(image.rows * scale));

		ensureSizeIsEnough(sz, image.type(), _vcvgmImagePyr[level]);
		ensureSizeIsEnough(sz, CV_8UC1, _vcvgmMaskPyr[level]);
		_vcvgmMaskPyr[level].setTo(cv::Scalar::all(255));

		// Compute the resized image
		if (level != _nFirstLevel)
		{
			if (level < _nFirstLevel){
				resize(image, _vcvgmImagePyr[level], sz, 0, 0, cv::INTER_LINEAR);

				if (!mask.empty())
					resize(mask, _vcvgmMaskPyr[level], sz, 0, 0, cv::INTER_LINEAR);
			}
			else{
				resize(_vcvgmImagePyr[level - 1], _vcvgmImagePyr[level], sz, 0, 0, cv::INTER_LINEAR);

				if (!mask.empty()){
					resize(_vcvgmMaskPyr[level - 1], _vcvgmMaskPyr[level], sz, 0, 0, cv::INTER_LINEAR);
					threshold(_vcvgmMaskPyr[level], _vcvgmMaskPyr[level], 254, 0, cv::THRESH_TOZERO);
				}
			}//else
		}
		else{
			image.copyTo(_vcvgmImagePyr[level]);
			if (!mask.empty())	mask.copyTo(_vcvgmMaskPyr[level]);
		}

		// Filter pcvgmKeyPoints_ by cvgmImage_ border
		ensureSizeIsEnough(sz, CV_8UC1, _cvgmBuf);
		_cvgmBuf.setTo(cv::Scalar::all(0));
		cv::Rect inner(_nEdgeThreshold, _nEdgeThreshold, sz.width - 2 * _nEdgeThreshold, sz.height - 2 * _nEdgeThreshold);
		_cvgmBuf(inner).setTo(cv::Scalar::all(255));

		bitwise_and(_vcvgmMaskPyr[level], _cvgmBuf, _vcvgmMaskPyr[level]);
	}//for( int level = 0)
}//build scale cvgmImage_

void CORB::computeKeyPointsPyramid()
{
	using namespace btl::device::orb;

	int half_patch_size = _nPatchSize / 2;

	_vcvgmKeyPointsPyr.resize(_nLevels);
	_vKeyPointsCount.resize(_nLevels);

	for (int level = 0; level < _nLevels; ++level)
	{
		_vKeyPointsCount[level] = _fastDetector.calcKeyPointsLocation(_vcvgmImagePyr[level], _vcvgmMaskPyr[level]);

		if (_vKeyPointsCount[level] == 0) continue;

		ensureSizeIsEnough(3, _vKeyPointsCount[level], CV_32FC1, _vcvgmKeyPointsPyr[level]);

		cv::gpu::GpuMat cvgmFastKpRange = _vcvgmKeyPointsPyr[level].rowRange(0, 2);
		_vKeyPointsCount[level] = _fastDetector.getKeyPoints(&cvgmFastKpRange);

		if (_vKeyPointsCount[level] == 0) continue;

		int nFeatures = static_cast<int>(_vFeaturesPerLevel[level]);

		if (_nScoreType == cv::ORB::HARRIS_SCORE){
			// Keep more points than necessary as FAST does not give amazing corners
			sortAndCull(_vcvgmKeyPointsPyr[level], 2 * nFeatures, &(_vKeyPointsCount[level]));
			// Compute the Harris cornerness (better scoring than FAST)
			cudaCalcHarrisResponses(_vcvgmImagePyr[level], _vcvgmKeyPointsPyr[level].ptr<short2>(0), _vcvgmKeyPointsPyr[level].ptr<float>(1), _vKeyPointsCount[level], 7, HARRIS_K, 0);
		}

		//sortAndCull to the final desired level, using the new Harris scores or the original FAST scores.
		sortAndCull(_vcvgmKeyPointsPyr[level], nFeatures, &(_vKeyPointsCount[level]));

		// Compute orientation
		IC_Angle_gpu(_vcvgmImagePyr[level], _vcvgmKeyPointsPyr[level].ptr<short2>(0), _vcvgmKeyPointsPyr[level].ptr<float>(2), _vKeyPointsCount[level], half_patch_size, 0);
	}//for (int l = 0; l < _nLevels; ++l)
	return;
}//computeKeyPointsPyramid()

void CORB::computeDescriptors(cv::gpu::GpuMat* pcvgmDescriptors_)
{
	using namespace btl::device::orb;

	int nAllKeyPoints = 0;

	for (int l = 0; l < _nLevels; ++l)
		nAllKeyPoints += _vKeyPointsCount[l];

	if (nAllKeyPoints == 0)	{
		pcvgmDescriptors_->release();
		return;
	}

	ensureSizeIsEnough(nAllKeyPoints, descriptorSize(), CV_8UC1, *pcvgmDescriptors_);

	int nOffset = 0;

	for (int l = 0; l < _nLevels; ++l)
	{
		if (_vKeyPointsCount[l] == 0) continue;

		cv::gpu::GpuMat cvgmDesciptorRange = pcvgmDescriptors_->rowRange(nOffset, nOffset + _vKeyPointsCount[l]);

		if (_bBlurForDescriptor){
			// preprocess the resized cvgmImage_
			ensureSizeIsEnough(_vcvgmImagePyr[l].size(), _vcvgmImagePyr[l].type(), _cvgmBuf);
			_pBlurFilter->apply(_vcvgmImagePyr[l], _cvgmBuf, cv::Rect(0, 0, _vcvgmImagePyr[l].cols, _vcvgmImagePyr[l].rows));
		}

		cudaComputeOrbDescriptor(_bBlurForDescriptor ? _cvgmBuf : _vcvgmImagePyr[l], _vcvgmKeyPointsPyr[l].ptr<short2>(0), _vcvgmKeyPointsPyr[l].ptr<float>(2),
			_vKeyPointsCount[l], _cvgmPattern.ptr<int>(0), _cvgmPattern.ptr<int>(1), cvgmDesciptorRange, descriptorSize(), _nWTA_K, 0);

		nOffset += _vKeyPointsCount[l];
	}
}

void CORB::mergeKeyPoints(cv::gpu::GpuMat* pcvgmKeyPoints_)
{
	using namespace btl::device::orb;

	int nAllkeypoints = 0;

	for (int l = 0; l < _nLevels; ++l)
		nAllkeypoints += _vKeyPointsCount[l];

	if (nAllkeypoints == 0)
	{
		pcvgmKeyPoints_->release();
		return;
	}

	ensureSizeIsEnough(ROWS_COUNT, nAllkeypoints, CV_32FC1, *pcvgmKeyPoints_);

	int nOffset = 0;

	for (int l = 0; l < _nLevels; ++l)
	{
		if (_vKeyPointsCount[l] == 0)
			continue;

		float sf = getScale(_fScaleFactor, _nFirstLevel, l);

		cv::gpu::GpuMat keyPointsRange = pcvgmKeyPoints_->colRange(nOffset, nOffset + _vKeyPointsCount[l]);

		float fLocScale = l != _nFirstLevel ? sf : 1.0f;

		cudaMergeLocation(_vcvgmKeyPointsPyr[l].ptr<short2>(0), keyPointsRange.ptr<float>(0), keyPointsRange.ptr<float>(1), _vKeyPointsCount[l], fLocScale, 0);

		cv::gpu::GpuMat range = keyPointsRange.rowRange(2, 4);
		_vcvgmKeyPointsPyr[l](cv::Range(1, 3), cv::Range(0, _vKeyPointsCount[l])).copyTo(range);

		keyPointsRange.row(4).setTo(cv::Scalar::all(l));
		keyPointsRange.row(5).setTo(cv::Scalar::all(_nPatchSize * sf));

		nOffset += _vKeyPointsCount[l];
	}
}

void CORB::downloadKeyPoints(const cv::gpu::GpuMat &cvgmKeypoints_, std::vector<cv::KeyPoint>* pvKeypoints_)
{
	if (cvgmKeypoints_.empty())
	{
		pvKeypoints_->clear();
		return;
	}

	cv::Mat cvmKeypoints(cvgmKeypoints_);

	convertKeyPoints(cvmKeypoints, &*pvKeypoints_);
}

void CORB::convertKeyPoints(const cv::Mat &cvmKeypoints_, std::vector<cv::KeyPoint>* pvKeypoints_)
{
	if (cvmKeypoints_.empty())
	{
		pvKeypoints_->clear();
		return;
	}

	//CV_Assert(cvmKeypoints_.type() == CV_32FC1 && cvmKeypoints_.rows == ROWS_COUNT);

	const float* pX =		cvmKeypoints_.ptr<float>(X_ROW);
	const float* pY =		cvmKeypoints_.ptr<float>(Y_ROW);
	const float* pResponse =cvmKeypoints_.ptr<float>(RESPONSE_ROW);
	const float* pAngle =	cvmKeypoints_.ptr<float>(ANGLE_ROW);
	const float* pOctave =	cvmKeypoints_.ptr<float>(OCTAVE_ROW);
	const float* pSize =	cvmKeypoints_.ptr<float>(SIZE_ROW);

	pvKeypoints_->resize(cvmKeypoints_.cols);

	for (int i = 0; i < cvmKeypoints_.cols; ++i)
	{
		cv::KeyPoint kp;

		kp.pt.x = pX[i];
		kp.pt.y = pY[i];
		kp.response = pResponse[i];
		kp.angle = pAngle[i];
		kp.octave = static_cast<int>(pOctave[i]);
		kp.size = pSize[i];

		(*pvKeypoints_)[i] = kp;
	}
}

void CORB::operator()(const cv::gpu::GpuMat& cvgmImage_, const cv::gpu::GpuMat& cvgmMask_, cv::gpu::GpuMat* pcvgmKeypoints_)
{
	buildScalePyramids(cvgmImage_, cvgmMask_);
	computeKeyPointsPyramid();
	mergeKeyPoints(&*pcvgmKeypoints_);
}

void CORB::operator()(const cv::gpu::GpuMat& cvgmImage_, const cv::gpu::GpuMat& cvgmMask_, cv::gpu::GpuMat* pcvgmKeypoints, cv::gpu::GpuMat* pcvgmDescriptors_)
{
	buildScalePyramids(cvgmImage_, cvgmMask_);
	computeKeyPointsPyramid();
	computeDescriptors(&*pcvgmDescriptors_);
	mergeKeyPoints(&*pcvgmKeypoints);
}

void CORB::operator()(const cv::gpu::GpuMat& cvgmImage_, const cv::gpu::GpuMat& cvgmMask_, std::vector<cv::KeyPoint>* pvKeypoints_)
{
	(*this)(cvgmImage_, cvgmMask_, &_cvgmKeypoints);
	downloadKeyPoints(_cvgmKeypoints, &*pvKeypoints_);
}

void CORB::operator()(const cv::gpu::GpuMat& cvgmImage_, const cv::gpu::GpuMat& cvgmMask_, std::vector<cv::KeyPoint>* pvKeypoints_, cv::gpu::GpuMat* pcvgmDescriptors_)
{
	(*this)(cvgmImage_, cvgmMask_, &_cvgmKeypoints, &*pcvgmDescriptors_);
	downloadKeyPoints(_cvgmKeypoints, &*pvKeypoints_);
}

void CORB::release()
{
	_vcvgmImagePyr.clear();
	_vcvgmMaskPyr.clear();
	_vcvgmKeyPointsPyr.clear();

	_cvgmBuf.release();
	_fastDetector.release();
	_cvgmKeypoints.release();
}




}//namespace cvgmImage_
}//namespace btl