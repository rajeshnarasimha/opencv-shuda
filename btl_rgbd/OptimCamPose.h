#ifndef BTL_OPTIMIZATION_CAMPOSE
#define BTL_OPTIMIZATION_CAMPOSE

namespace btl{ namespace utility{

class COptimCamPose: public COptim{
public:
	// override this to check if everything has been initialized and ready to start
	virtual bool isOK(); 

	// cost functions to be mininized
	// override this by the actual cost function,
	// default is the Rosenbrock's Function
	virtual double Func( const cv::Mat_<double>& cvmX_ );
	virtual void  dFunc(const cv::Mat_<double>& cvmX_, cv::Mat_<double>& cvmG_ );
	//| R' 0 |
	//| T' 1 | SE3
	cv::Mat setSE3( const cv::Mat& cvmR_, const cv::Mat& cvmT_ );
	void getRT(Eigen::Matrix3d* peimR_, Eigen::Vector3d* peivT_);
	cv::Mat_<double>	_cvmPlaneRef; //4 by # of planes
	cv::Mat_<double>	_cvmPlaneCur; //4 by # of planes
	cv::Mat_<double>    _cvmPlaneWeight;//1 by # of planes
};

}//utitlity
}//btl

#endif