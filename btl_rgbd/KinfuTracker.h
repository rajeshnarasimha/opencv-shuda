#ifndef BTL_GEOMETRY_KINFU_TRACKER
#define BTL_GEOMETRY_KINFU_TRACKER

namespace btl{ namespace geometry
{

	class CKinFuTracker
	{
	public:
		//type
		typedef boost::shared_ptr<CKinFuTracker> tp_shared_ptr;

		enum{ICP, ORBICP, ORB, SURF, SURFICP};
	public:
		//both pKeyFrame_ and pCubicGrids_ must be allocated before hand
		CKinFuTracker(btl::kinect::CKeyFrame::tp_ptr pKeyFrame_,CCubicGrids::tp_shared_ptr pCubicGrids_ /*ushort usVolumeResolution_,float fVolumeSizeM_*/ );
		~CKinFuTracker(){;}
		void setMethod(int nMethod_){ _nMethod = nMethod_;}
		void init(btl::kinect::CKeyFrame::tp_ptr pKeyFrame_);
		void track( btl::kinect::CKeyFrame::tp_ptr pCurFrame_ );
		void setNextView( Eigen::Matrix4f* pSystemPose_ );
		void setPrevView( Eigen::Matrix4f* pSystemPose_ ) const;

		btl::kinect::CKeyFrame::tp_ptr prevFrame() const{ return _pPrevFrameWorld.get();}
	protected:
		//ICP approach
		void initICP(btl::kinect::CKeyFrame::tp_ptr pKeyFrame_);
		void trackICP(btl::kinect::CKeyFrame::tp_ptr pCurFrame_);
		//ORB + ICP approach
		void initORBICP(btl::kinect::CKeyFrame::tp_ptr pKeyFrame_);
		void trackORBICP(btl::kinect::CKeyFrame::tp_ptr pCurFrame_);
		//ORB 
		void initORB( btl::kinect::CKeyFrame::tp_ptr pKeyFrame_ );
		void trackORB( btl::kinect::CKeyFrame::tp_ptr pCurFrame_ );
		//SURF
		void initSURF( btl::kinect::CKeyFrame::tp_ptr pKeyFrame_ );
		void trackSURF( btl::kinect::CKeyFrame::tp_ptr pCurFrame_ );
		//SURF + ICP
		void initSURFICP( btl::kinect::CKeyFrame::tp_ptr pKeyFrame_ );
		void trackSURFICP( btl::kinect::CKeyFrame::tp_ptr pCurFrame_ );
		//Brox optical flow
		void initBroxOpticalFlow( btl::kinect::CKeyFrame::tp_ptr pKeyFrame_ );
		CCubicGrids::tp_shared_ptr _pCubicGrids;
		btl::kinect::CKeyFrame::tp_scoped_ptr _pPrevFrameWorld;

		btl::kinect::SCamera::tp_ptr _pRGBCamera; //share the content of the RGBCamera with those from VideoKinectSource
		Eigen::Matrix4f _eimCurPose;
		unsigned int _uViewNO;
		std::vector<Eigen::Matrix4f> _veimPoses;
		int _nMethod;
	};//CKinFuTracker

}//geometry
}//btl

#endif