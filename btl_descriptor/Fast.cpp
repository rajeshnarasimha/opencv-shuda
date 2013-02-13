
#include <vector>
#include <opencv2/gpu/gpu.hpp>
#include "Fast.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <npp.h>

//using namespace cv;
//using namespace cv::gpu;
//using namespace std;

namespace btl { namespace device { 
	namespace fast
	{
		unsigned int cudaCalcKeypoints(const cv::gpu::PtrStepSzb cvgmImage_, const cv::gpu::PtrStepSzb cvgmMask_, const unsigned int uMaxKeyPoints_,const int nThreshold_, short2* ps2KeyPointLoc_, cv::gpu::GpuMat* pcvgmScore_);
		int cudaNonMaxSupression(const short2* ps2KeyPointLoc_, const int nCount_, cv::gpu::PtrStepSzi cvgmScore_, short2* ps2Locations_, float* pfResponse_);
	}
}//namespace device
}//namespace btl

namespace btl{ namespace image{

CFast::CFast(int nThreshold_, bool bNonMaxSupression_  /*= true*/, double dKeyPointsRatio_ /*= 0.05*/) :
_bNonMaxSupression(bNonMaxSupression_), _nThreshold(nThreshold_), _dKeyPointsRatio(dKeyPointsRatio_), _uCount(0)
{
}

void CFast::operator ()(const cv::gpu::GpuMat& cvgmImage_, const cv::gpu::GpuMat& cvgmMask_, std::vector<cv::KeyPoint>* pvKeyPoints_)
{
	if (cvgmImage_.empty())
		return;

	(*this)(cvgmImage_, cvgmMask_, &_cvgmdKeyPoints);
	downloadKeypoints(_cvgmdKeyPoints, &*pvKeyPoints_);
}

void CFast::operator ()(const cv::gpu::GpuMat& cvgmImg_, const cv::gpu::GpuMat& cvgmMask_, cv::gpu::GpuMat* pcvgmKeyPoints_)
{
	calcKeyPointsLocation(cvgmImg_, cvgmMask_);
	//perform non-max suppression
	pcvgmKeyPoints_->cols = getKeyPoints(&*pcvgmKeyPoints_);
}


void CFast::downloadKeypoints(const cv::gpu::GpuMat& cvgmKeyPoints_, std::vector<cv::KeyPoint>* pvKeyPoints_)
{
	if (cvgmKeyPoints_.empty())	return;
	cv::Mat cvmKeyPoints(cvgmKeyPoints_);//download from cvgm to cvm
	convertKeypoints(cvmKeyPoints, &*pvKeyPoints_);
}

void CFast::convertKeypoints(const cv::Mat& cvmKeyPoints_, std::vector<cv::KeyPoint>* pvKeyPoints_)
{
	if (cvmKeyPoints_.empty())	return;

	CV_Assert(cvmKeyPoints_.rows == ROWS_COUNT && cvmKeyPoints_.elemSize() == 4);

	int nPoints = cvmKeyPoints_.cols;
	pvKeyPoints_->resize(nPoints);

	const short2* ps2LocationRow = cvmKeyPoints_.ptr<short2>(LOCATION_ROW);
	const float* pfResponseRow = cvmKeyPoints_.ptr<float>(RESPONSE_ROW);

	for (int i = 0; i < nPoints; ++i){
		cv::KeyPoint kp(ps2LocationRow[i].x, ps2LocationRow[i].y, static_cast<float>(FEATURE_SIZE), -1, pfResponseRow[i]);
		(*pvKeyPoints_)[i] = kp;
	}
	return;
}
//! find keypoints and compute it's response if _bNonMaxSupression is true
//! return count of detected keypoints
//return the # of keypoints, stored in _uCount;
//store Key point locations into _cvgmKeyPointLocation;
//store the corner strength into _cvgmScore;
int CFast::calcKeyPointsLocation(const cv::gpu::GpuMat& cvgmImg_, const cv::gpu::GpuMat& cvgmMask_)
{
	//using namespace cv::gpu::device::fast;

	CV_Assert(cvgmImg_.type() == CV_8UC1);
	CV_Assert(cvgmMask_.empty() || (cvgmMask_.type() == CV_8UC1 && cvgmMask_.size() == cvgmImg_.size()));

	if (!cv::gpu::TargetArchs::builtWith(cv::gpu::GLOBAL_ATOMICS) || !cv::gpu::DeviceInfo().supports(cv::gpu::GLOBAL_ATOMICS))
		CV_Error(CV_StsNotImplemented, "The device doesn't support global atomics");

	unsigned int uMaxKeypoints = static_cast<unsigned int>(_dKeyPointsRatio * cvgmImg_.size().area());

	ensureSizeIsEnough(1, uMaxKeypoints, CV_16SC2, _cvgmKeyPointLocation);

	if (_bNonMaxSupression)
	{
		ensureSizeIsEnough(cvgmImg_.size(), CV_32SC1, _cvgmScore);
		_cvgmScore.setTo(cv::Scalar::all(0));
	}

	_uCount = btl::device::fast::cudaCalcKeypoints(cvgmImg_, cvgmMask_, uMaxKeypoints, _nThreshold, _cvgmKeyPointLocation.ptr<short2>(), _bNonMaxSupression ? &_cvgmScore : NULL);
	_uCount = std::min(_uCount, uMaxKeypoints);

	return _uCount;
}

int CFast::getKeyPoints(cv::gpu::GpuMat* pcvgmKeyPoints_)
{
	if (!cv::gpu::TargetArchs::builtWith(cv::gpu::GLOBAL_ATOMICS) || !cv::gpu::DeviceInfo().supports(cv::gpu::GLOBAL_ATOMICS))
		CV_Error(CV_StsNotImplemented, "The device doesn't support global atomics");

	if (_uCount == 0) return 0;

	ensureSizeIsEnough(ROWS_COUNT, _uCount, CV_32FC1, *pcvgmKeyPoints_);

	if (_bNonMaxSupression)
		return btl::device::fast::cudaNonMaxSupression(_cvgmKeyPointLocation.ptr<short2>(), _uCount, _cvgmScore, pcvgmKeyPoints_->ptr<short2>(LOCATION_ROW), pcvgmKeyPoints_->ptr<float>(RESPONSE_ROW));

	cv::gpu::GpuMat cvgmLocRow(1, _uCount, _cvgmKeyPointLocation.type(), pcvgmKeyPoints_->ptr(0));
	_cvgmKeyPointLocation.colRange(0, _uCount).copyTo(cvgmLocRow);
	pcvgmKeyPoints_->row(1).setTo(cv::Scalar::all(0));

	return _uCount;
}

void CFast::release()
{
	_cvgmKeyPointLocation.release();
	_cvgmdKeyPoints.release();
	_cvgmScore.release();
}

}//namespace cvgmImage_
}//namespace btl