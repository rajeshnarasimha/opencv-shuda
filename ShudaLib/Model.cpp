
#define INFO
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>
#include <fstream>
#include <list>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <gl/freeglut.h>

#include "OtherUtil.hpp"
#include "Converters.hpp"
#include "EigenUtil.hpp"
#include "Camera.h"
#include "Kinect.h"
#include "GLUtil.h"
#include "Histogram.h"
#include "KeyFrame.h"
#include "Model.h"


namespace btl{ namespace geometry
{

CModel::CModel()
{
	_cvgmXYxZVolContent.create(VOLUME_LEVEL,VOLUME_RESOL,CV_32FC1);
	_cvgmX
	_cvgmXYxZVolContent.download(_cvmXYxZVolContent);

}
CModel::~CModel(void)
{
}

void CModel::gpuIntegrate( btl::kinect::CKeyFrame& cFrame_, unsigned short usPyrLevel_ )
{
	BTL_ASSERT( btl::utility::BTL_CV == cFrame_._eConvention, "the frame depth data must be captured in cv-convention");
	btl::cuda_util::cudaIntegrate(*cFrame_._acvgmShrPtrPyrPts[usPyrLevel_],cFrame_._eimR.data(),cFrame_._eivT.data(),&_cvgmXYxZVolContent);
	_cvgmXYxZVolContent.download(_cvmXYxZVolContent);
}

}//geometry
}//btl
