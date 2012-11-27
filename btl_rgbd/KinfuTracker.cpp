#define INFO
#include <GL/glew.h>
#include <gl/freeglut.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
//stl
#include <iostream>
#include <string>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
//boost
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
//openncv
#include <opencv2/gpu/gpumat.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <utility>
#include <boost/lexical_cast.hpp>
#include <gl/freeglut.h>
#include <XnCppWrapper.h>
#include "Converters.hpp"
#include "GLUtil.h"
#include "EigenUtil.hpp"
#include "Camera.h"
#include "GLUtil.h"
#include "PlaneObj.h"
#include "Histogram.h"
#include "KeyFrame.h"
#include "CyclicBuffer.h"
#include "VideoSourceKinect.hpp"
#include "CubicGrids.h"
#include "KinfuTracker.h"

namespace btl{ namespace geometry
{
	CKinFuTracker::CKinFuTracker(btl::kinect::CKeyFrame::tp_ptr pKeyFrame_,btl::geometry::CCubicGrids::tp_shared_ptr pCubicGrids_ /*ushort usVolumeResolution_,float fVolumeSizeM_*/ )
		:_pCubicGrids(pCubicGrids_)
	{
		_nMethod = CKinFuTracker::ICP;
	}

	void CKinFuTracker::init(const btl::kinect::CKeyFrame::tp_ptr pKeyFrame_){
		//Note: the point cloud of the pKeyFrame is located in camera coordinate system
		switch(_nMethod)
		{
		case ICP:
			initICP(pKeyFrame_);
			break;
		case ORB:
			initORB(pKeyFrame_);
			break;
		case ORBICP:
			initORBICP(pKeyFrame_);
			break;
		case SURF:
			initSURF(pKeyFrame_);
			break;
		}
		return;
	}

	void CKinFuTracker::track(btl::kinect::CKeyFrame::tp_ptr pCurFrame_){
		switch(_nMethod)
		{
		case ICP:
			trackICP(pCurFrame_);
			break;
		case ORB:
			trackORB(pCurFrame_);
			break;
		case ORBICP:
			trackORBICP(pCurFrame_);
			break;
		case SURF:
			trackSURF(pCurFrame_);
			break;
		}
		return;
	}

	void CKinFuTracker::initICP(const btl::kinect::CKeyFrame::tp_ptr pKeyFrame_)
	{
		//input key frame must be defined in local camera system
		_pCubicGrids->reset();
		_veimPoses.clear();
		_veimPoses.reserve(1000);
		//initialize pose
		pKeyFrame_->setView(&_eimCurPose);
		_veimPoses.push_back(_eimCurPose); // the first pose is initialize by the pKeyFrame_;
		//copy pKeyFrame_ to _pPrevFrameWorld
		_pPrevFrameWorld.reset(new btl::kinect::CKeyFrame(pKeyFrame_));	
		_pPrevFrameWorld->gpuTransformToWorldCVCV();//transform from camera to world
		//integrate the frame into the world
		_pCubicGrids->gpuIntegrateFrameIntoVolumeCVCV(*_pPrevFrameWorld);

	}

	void CKinFuTracker::trackICP(btl::kinect::CKeyFrame::tp_ptr pCurFrame_){
		//the current frame is defined in camera system
		//PRINTSTR("ICP tracking.");
		pCurFrame_->setRTTo(*_pPrevFrameWorld);//initialize the current un-calibrated frame as the previous frame
		pCurFrame_->gpuICP ( _pPrevFrameWorld.get(), false );//refine the R,T with w.r.t. previous key frame
		if( pCurFrame_->isMovedwrtReferencInRadiusM( _pPrevFrameWorld.get(),M_PI_4/45.,0.02) ){ //test if the current frame have been moving
			pCurFrame_->gpuTransformToWorldCVCV();
			_pCubicGrids->gpuIntegrateFrameIntoVolumeCVCV(*pCurFrame_);
			//refresh prev frame in world as the ray casted virtual frame
			_pPrevFrameWorld->setRTTo( *pCurFrame_ );
			_pCubicGrids->gpuRaycast( &*_pPrevFrameWorld ); //get virtual frame
			pCurFrame_->copyImageTo(&*_pPrevFrameWorld); //fill in the color info
			//store R t pose
			pCurFrame_->setView(&_eimCurPose);
			_veimPoses.push_back(_eimCurPose);
		}//if current frame moved
		return;
	} //trackICP()

	void CKinFuTracker::setNextView( Eigen::Matrix4f* pSystemPose_ )
	{
		_uViewNO = ++_uViewNO % _veimPoses.size(); 
		*pSystemPose_ = _veimPoses[_uViewNO];
	}

	void CKinFuTracker::initORBICP( btl::kinect::CKeyFrame::tp_ptr pKeyFrame_ )
	{
		//input key frame must be defined in local camera system
		_pCubicGrids->reset();
		_veimPoses.clear();
		_veimPoses.reserve(1000);
		//initialize pose
		pKeyFrame_->setView(&_eimCurPose);
		_veimPoses.push_back(_eimCurPose); // the first pose is initialize by the pKeyFrame_;
		//copy pKeyFrame_ to _pPrevFrameWorld
		_pPrevFrameWorld.reset(new btl::kinect::CKeyFrame(pKeyFrame_));	
		_pPrevFrameWorld->gpuTransformToWorldCVCV();//transform from camera to world
		//integrate the frame into the world
		_pCubicGrids->gpuIntegrateFrameIntoVolumeCVCV(*_pPrevFrameWorld);
		//extract ORB features
		_pPrevFrameWorld->extractOrbFeatures();
	}

	void CKinFuTracker::trackORBICP( btl::kinect::CKeyFrame::tp_ptr pCurFrame_ )
	{
		//the current frame is defined in camera system
		//PRINTSTR("ICP tracking.");
		pCurFrame_->setRTTo(*_pPrevFrameWorld);//initialize the current un-calibrated frame as the previous frame
		//trackICP camera motion
		pCurFrame_->extractOrbFeatures();
		ushort uInliers;
		double dError = pCurFrame_->calcRTOrb ( *_pPrevFrameWorld,.2,&uInliers ); //roughly estimate R,T w.r.t. last key frame,
		PRINT(uInliers)
		if ( uInliers > 60) {
			pCurFrame_->gpuICP ( _pPrevFrameWorld.get(), false );//refine the R,T with w.r.t. previous key frame
			if( pCurFrame_->isMovedwrtReferencInRadiusM( _pPrevFrameWorld.get(),M_PI_4/45.,0.02) ){ //test if the current frame have been moving
				pCurFrame_->gpuTransformToWorldCVCV();
				_pCubicGrids->gpuIntegrateFrameIntoVolumeCVCV(*pCurFrame_);
				//refresh prev frame in world as the ray casted virtual frame
				_pPrevFrameWorld->setRTTo( *pCurFrame_ );
				_pCubicGrids->gpuRaycast( &*_pPrevFrameWorld ); //get virtual frame
				//store R t pose
				pCurFrame_->setView(&_eimCurPose);
				_veimPoses.push_back(_eimCurPose);
				pCurFrame_->copyImageTo(&*_pPrevFrameWorld);
			}//if current frame moved
		}
		return;
	}//trackORBICP
		
	void CKinFuTracker::initORB( btl::kinect::CKeyFrame::tp_ptr pKeyFrame_ )
	{
		//input key frame must be defined in local camera system
		_pCubicGrids->reset();
		_veimPoses.clear();
		_veimPoses.reserve(1000);
		//initialize pose
		pKeyFrame_->setView(&_eimCurPose);
		_veimPoses.push_back(_eimCurPose); // the first pose is initialize by the pKeyFrame_;
		//copy pKeyFrame_ to _pPrevFrameWorld
		_pPrevFrameWorld.reset(new btl::kinect::CKeyFrame(pKeyFrame_));	
		_pPrevFrameWorld->gpuTransformToWorldCVCV();//transform from camera to world
		//integrate the frame into the world
		_pCubicGrids->gpuIntegrateFrameIntoVolumeCVCV(*_pPrevFrameWorld);

		_pPrevFrameWorld->extractOrbFeatures();
		return;
	}//initORB()
		
	void CKinFuTracker::trackORB( btl::kinect::CKeyFrame::tp_ptr pCurFrame_ )
	{
		//the current frame is defined in camera system
		//PRINTSTR("ICP tracking.");
		pCurFrame_->setRTTo(*_pPrevFrameWorld);//initialize the current un-calibrated frame as the previous frame
		//trackICP camera motion
		pCurFrame_->extractOrbFeatures();
		ushort uInliers;
		double dError = pCurFrame_->calcRTOrb ( *_pPrevFrameWorld, .2, &uInliers ); //roughly estimate R,T w.r.t. last key frame,
		if ( /*uInliers > 300 &&*/ pCurFrame_->isMovedwrtReferencInRadiusM( _pPrevFrameWorld.get(), M_PI_4/45., 0.02 )) {//test if the current frame have been moving
			pCurFrame_->gpuTransformToWorldCVCV();
			_pCubicGrids->gpuIntegrateFrameIntoVolumeCVCV(*pCurFrame_);
			//refresh prev frame in world as the ray casted virtual frame
			_pPrevFrameWorld->setRTTo( *pCurFrame_ );
			pCurFrame_->copyTo(&*_pPrevFrameWorld);
			//pCurFrame_->copyImageTo(&*_pPrevFrameWorld);
			//store R t pose
			pCurFrame_->setView(&_eimCurPose);
			_veimPoses.push_back(_eimCurPose);
		}
		return;
	}//trackORB

	void CKinFuTracker::initSURF( btl::kinect::CKeyFrame::tp_ptr pKeyFrame_ )
	{
		//input key frame must be defined in local camera system
		_pCubicGrids->reset();
		_veimPoses.clear();
		_veimPoses.reserve(1000);
		//initialize pose
		pKeyFrame_->setView(&_eimCurPose);
		_veimPoses.push_back(_eimCurPose); // the first pose is initialize by the pKeyFrame_;
		//copy pKeyFrame_ to _pPrevFrameWorld
		_pPrevFrameWorld.reset(new btl::kinect::CKeyFrame(pKeyFrame_));	
		_pPrevFrameWorld->gpuTransformToWorldCVCV();//transform from camera to world
		//integrate the frame into the world
		_pCubicGrids->gpuIntegrateFrameIntoVolumeCVCV(*_pPrevFrameWorld);

		_pPrevFrameWorld->extractSurfFeatures();
		return;
	}//initSURF()
	
	void CKinFuTracker::trackSURF( btl::kinect::CKeyFrame::tp_ptr pCurFrame_ )
	{
		//the current frame is defined in camera system
		//PRINTSTR("ICP tracking.");
		pCurFrame_->setRTTo(*_pPrevFrameWorld);//initialize the current un-calibrated frame as the previous frame
		//trackICP camera motion
		pCurFrame_->extractSurfFeatures();
		ushort uInliers;
		double dError = pCurFrame_->calcRT ( *_pPrevFrameWorld,0,.2,&uInliers ); //roughly estimate R,T w.r.t. last key frame,
		if ( /*uInliers > 300 && */pCurFrame_->isMovedwrtReferencInRadiusM( _pPrevFrameWorld.get(),M_PI_4/45.,0.02)) {//test if the current frame have been moving

			pCurFrame_->gpuTransformToWorldCVCV();
			_pCubicGrids->gpuIntegrateFrameIntoVolumeCVCV(*pCurFrame_);
			//refresh prev frame in world as the ray casted virtual frame
			_pPrevFrameWorld->setRTTo( *pCurFrame_ );
			pCurFrame_->copyTo(&*_pPrevFrameWorld);
			//store R t pose
			pCurFrame_->setView(&_eimCurPose);
			_veimPoses.push_back(_eimCurPose);
		}
		return;
	}//trackORB

	void CKinFuTracker::initSURFICP( btl::kinect::CKeyFrame::tp_ptr pKeyFrame_ )
	{
		//input key frame must be defined in local camera system
		_pCubicGrids->reset();
		_veimPoses.clear();
		_veimPoses.reserve(1000);
		//initialize pose
		pKeyFrame_->setView(&_eimCurPose);
		_veimPoses.push_back(_eimCurPose); // the first pose is initialize by the pKeyFrame_;
		//copy pKeyFrame_ to _pPrevFrameWorld
		_pPrevFrameWorld.reset(new btl::kinect::CKeyFrame(pKeyFrame_));	
		_pPrevFrameWorld->gpuTransformToWorldCVCV();//transform from camera to world
		//integrate the frame into the world
		_pCubicGrids->gpuIntegrateFrameIntoVolumeCVCV(*_pPrevFrameWorld);
		//extract ORB features
		_pPrevFrameWorld->extractSurfFeatures();
	}

	void CKinFuTracker::trackSURFICP( btl::kinect::CKeyFrame::tp_ptr pCurFrame_ )
	{
		//the current frame is defined in camera system
		//PRINTSTR("ICP tracking.");
		pCurFrame_->setRTTo(*_pPrevFrameWorld);//initialize the current un-calibrated frame as the previous frame
		//trackICP camera motion
		pCurFrame_->extractSurfFeatures();
		ushort uInliers;
		double dError = pCurFrame_->calcRT ( *_pPrevFrameWorld,0,.2,&uInliers ); //roughly estimate R,T w.r.t. last key frame,
		if ( uInliers > 300) {
			pCurFrame_->gpuICP ( _pPrevFrameWorld.get(), false );//refine the R,T with w.r.t. previous key frame
			if( pCurFrame_->isMovedwrtReferencInRadiusM( _pPrevFrameWorld.get(),M_PI_4/45.,0.02) ){ //test if the current frame have been moving
				pCurFrame_->gpuTransformToWorldCVCV();
				_pCubicGrids->gpuIntegrateFrameIntoVolumeCVCV(*pCurFrame_);
				//refresh prev frame in world as the ray casted virtual frame
				_pPrevFrameWorld->setRTTo( *pCurFrame_ );
				_pCubicGrids->gpuRaycast( &*_pPrevFrameWorld ); //get virtual frame
				//store R t pose
				pCurFrame_->setView(&_eimCurPose);
				_veimPoses.push_back(_eimCurPose);
				pCurFrame_->copyImageTo(&*_pPrevFrameWorld);
			}//if current frame moved
		}
		return;
	}

	void CKinFuTracker::initBroxOpticalFlow( btl::kinect::CKeyFrame::tp_ptr pKeyFrame_ )
	{
		//input key frame must be defined in local camera system
		_pCubicGrids->reset();
		_veimPoses.clear();
		_veimPoses.reserve(1000);
		//initialize pose
		pKeyFrame_->setView(&_eimCurPose);
		_veimPoses.push_back(_eimCurPose); // the first pose is initialize by the pKeyFrame_;
		//copy pKeyFrame_ to _pPrevFrameWorld
		_pPrevFrameWorld.reset(new btl::kinect::CKeyFrame(pKeyFrame_));	
		_pPrevFrameWorld->gpuTransformToWorldCVCV();//transform from camera to world
		//integrate the frame into the world
		_pCubicGrids->gpuIntegrateFrameIntoVolumeCVCV(*_pPrevFrameWorld);
	}



}//geometry
}//btl