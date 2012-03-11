#define INFO
//gl
#include <gl/glew.h>
#include <gl/freeglut.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
//boost
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
//stl
#include <vector>
#include <fstream>
#include <list>
#include <math.h>
//openncv
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "OtherUtil.hpp"
#include "Converters.hpp"
#include "EigenUtil.hpp"
#include "Camera.h"
#include "CVUtil.hpp"
#include "Kinect.h"
#include "GLUtil.h"
#include "PlaneObj.h"
#include "Histogram.h"
#include "KeyFrame.h"
#include "PlaneWorld.h"
#include "PlaneObj.h"
#include "Optim.hpp"
#include "OptimCamPose.h"

btl::geometry::CSinglePlaneSingleViewInWorld::CSinglePlaneSingleViewInWorld(const btl::geometry::SPlaneObj& sPlaneObj_, ushort usPyrLevel_, btl::kinect::CKeyFrame::tp_ptr pFrame_, ushort usPlaneIdx_) 
:_pFrame(pFrame_),_usPlaneIdx(usPlaneIdx_){
	_aeivAvgNormal[usPyrLevel_] = sPlaneObj_._eivAvgNormal;
	_adAvgPosition[usPyrLevel_] = sPlaneObj_._dAvgPosition;
	_avIdx[usPyrLevel_].reset( new std::vector<unsigned int>(sPlaneObj_._vIdx.begin(),sPlaneObj_._vIdx.end()) ); 
}//SPlaneW()

void btl::geometry::CSinglePlaneSingleViewInWorld::renderInWorldCVGL(btl::gl_util::CGLUtil::tp_ptr pGL_, const uchar* pColor_, const ushort usPyrLevel_ ) const {
	glPushMatrix();
	_pFrame->loadGLMVIn(); //setup the modelview matrix for opengl
	const float* pPt = (const float*) _pFrame->_acvmShrPtrPyrPts[usPyrLevel_]->data;
	const float* pNl = (const float*) _pFrame->_acvmShrPtrPyrNls[usPyrLevel_]->data;
	glBegin(GL_POINTS);
	_pFrame->renderASinglePlaneObjInLocalCVGL(pPt,pNl,*_avIdx[usPyrLevel_],pColor_);
	glEnd();
	glPopMatrix();
}//renderInWorldCVGL()
bool btl::geometry::CSinglePlaneSingleViewInWorld::identical( const Eigen::Vector3d& eivNormal_, const double dPosition_, const ushort usPyrLevel_ ) const {
	double dCos = eivNormal_.dot( _aeivAvgNormal[usPyrLevel_] );
	double dDif = fabs( dPosition_ - _adAvgPosition[usPyrLevel_] );
	if(dCos > 0.75 && dDif < 0.1 ) return true; 
	else return false;
}//identical()
btl::geometry::CSinglePlaneMultiViewsInWorld::CSinglePlaneMultiViewsInWorld( const btl::geometry::SPlaneObj& sPlaneObj_, const ushort usPyrLevel_, btl::kinect::CKeyFrame::tp_ptr pFrame_, ushort usPlaneIdx_ )
:_usPlaneIdx(usPlaneIdx_){
	//create SPSV
	btl::geometry::CSinglePlaneSingleViewInWorld::tp_shared_ptr pShrPtrSPSV( new btl::geometry::CSinglePlaneSingleViewInWorld(sPlaneObj_,usPyrLevel_,pFrame_,usPlaneIdx_) );
	_vShrPtrSPSV.push_back(pShrPtrSPSV);
	_aeivAvgNormal[usPyrLevel_] = sPlaneObj_._eivAvgNormal;
	_adAvgPosition[usPyrLevel_] = sPlaneObj_._dAvgPosition;
}//CSinglePlaneMultiViewsInWorld()
bool btl::geometry::CSinglePlaneMultiViewsInWorld::identical( const Eigen::Vector3d& eivNormal_, const double dPosition_, const ushort usPyrLevel_ ) const {
	double dCos = eivNormal_.dot( _aeivAvgNormal[usPyrLevel_] );
	double dDif = fabs( dPosition_ - _adAvgPosition[usPyrLevel_] );
	if(dCos > 0.80 && dDif < 0.1 ) return true; 
	else return false;
}
void btl::geometry::CSinglePlaneMultiViewsInWorld::integrateFrameIntoPlanesWorldCVCV( btl::kinect::CKeyFrame::tp_ptr pFrame_, btl::geometry::tp_plane_obj_list& lPlanes_, const ushort usPyrLevel_, CMultiPlanesSingleViewInWorld::tp_ptr pMPSV_){
	//find the close plane objects
	btl::geometry::CSinglePlaneSingleViewInWorld::tp_shared_ptr pShrPtrSPSV;
	btl::geometry::tp_plane_obj_list::iterator itErase;
	bool bErase = false;
	for (btl::geometry::tp_plane_obj_list::iterator itPlaneObj = lPlanes_.begin();itPlaneObj != lPlanes_.end();itPlaneObj++){
		if(bErase){
			lPlanes_.erase(itErase);
			bErase = false;
		}
		if ( identical( itPlaneObj->_eivAvgNormal,itPlaneObj->_dAvgPosition, usPyrLevel_ ) ){
			if(!pShrPtrSPSV.get()){
				pShrPtrSPSV.reset( new btl::geometry::CSinglePlaneSingleViewInWorld(*itPlaneObj,usPyrLevel_,pFrame_,_usPlaneIdx) );
			}//pShrPtrSPSV not yet assigned
			else{
				pShrPtrSPSV->_avIdx[usPyrLevel_]->insert(pShrPtrSPSV->_avIdx[usPyrLevel_]->end(), itPlaneObj->_vIdx.begin(),itPlaneObj->_vIdx.end() );
			}//else add index to exiting pShrPtrSPSV
			{
				//set erase marker
				itErase = itPlaneObj;
				bErase = true;
			}
		}// if identical to the current SPMV
	}//for each plane world in pFrame_
	if(bErase){
		lPlanes_.erase(itErase);
		bErase = false;
	}
	if(pShrPtrSPSV.get()){
		_vShrPtrSPSV.push_back(pShrPtrSPSV);
		//update MPSV
		pMPSV_->_vPtrSPSV.push_back( pShrPtrSPSV.get() );
	}//if identical plane found
	return;
}//integrateFrameIntoPlanesWorldCVCV()
void btl::geometry::CSinglePlaneMultiViewsInWorld::renderSinglePlaneInAllViewsWorldGL(btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_,const ushort usPyrLevel_ /*= 3*/) const {
	
	const unsigned char* pColor = btl::utility::__aColors[_usPlaneIdx+usColorIdx_%BTL_NUM_COLOR];
	if( pGL_ && pGL_->_bEnableLighting ){glEnable(GL_LIGHTING);}
	else                            	{glDisable(GL_LIGHTING);}
	for (tp_shr_spsv_vec::const_iterator citShrPtrSPSV = _vShrPtrSPSV.begin(); citShrPtrSPSV!= _vShrPtrSPSV.end(); citShrPtrSPSV++){
		(*citShrPtrSPSV)->renderInWorldCVGL(pGL_,pColor,usPyrLevel_);
	}//for each plane
}//renderPlaneInWorldGL()

void btl::geometry::CSinglePlaneMultiViewsInWorld::renderSinglePlaneInSingleViewWorldGL( btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_,const ushort usView_, const ushort usPyrLevel_ /*= 3*/ ) const
{
	const unsigned char* pColor = btl::utility::__aColors[_usPlaneIdx+usColorIdx_%BTL_NUM_COLOR];
	if( pGL_ && pGL_->_bEnableLighting ){glEnable(GL_LIGHTING);}
	else                            	{glDisable(GL_LIGHTING);}
	_vShrPtrSPSV[usView_]->renderInWorldCVGL(pGL_,pColor,usPyrLevel_);
}

btl::geometry::CMultiPlanesSingleViewInWorld::CMultiPlanesSingleViewInWorld( btl::kinect::CKeyFrame::tp_ptr pFrame_ )
:_pFrame(pFrame_){}

void btl::geometry::CMultiPlanesSingleViewInWorld::renderAllPlanesInSingleViewWorldCVGL( btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_, const ushort usPyrLevel_ ) const {
	//setup the lighting
	if( pGL_ && pGL_->_bEnableLighting ){glEnable(GL_LIGHTING);}
	else                            	{glDisable(GL_LIGHTING);}
	glPushMatrix();
	_pFrame->loadGLMVIn(); //setup the modelview matrix for opengl
	const float* pPt = (const float*) _pFrame->_acvmShrPtrPyrPts[usPyrLevel_]->data;
	const float* pNl = (const float*) _pFrame->_acvmShrPtrPyrNls[usPyrLevel_]->data;
	glBegin(GL_POINTS);
	for (tp_ptr_spsv_vec::const_iterator citPlane = _vPtrSPSV.begin(); citPlane != _vPtrSPSV.end(); citPlane++ ) {
		const unsigned char* pColor = btl::utility::__aColors[(*citPlane)->_usPlaneIdx +usColorIdx_%BTL_NUM_COLOR];
		_pFrame->renderASinglePlaneObjInLocalCVGL(pPt,pNl,*(*citPlane)->_avIdx[usPyrLevel_],pColor);
	}//for each plane
	glEnd();
	glPopMatrix();
}//renderAllPlanesInWorldCVGL()

btl::geometry::CMultiPlanesMultiViewsInWorld::CMultiPlanesMultiViewsInWorld( btl::kinect::CKeyFrame::tp_ptr pFrame_ ){
	//store original key frame
	btl::kinect::CKeyFrame::tp_shared_ptr pShrKeyFrameStoredLocally (new btl::kinect::CKeyFrame(pFrame_) );
	_vShrPtrKFrs.push_back(pShrKeyFrameStoredLocally);
	//construct vector of MPSV
	CMultiPlanesSingleViewInWorld::tp_shared_ptr pShrPtrMPSV ( new CMultiPlanesSingleViewInWorld(pShrKeyFrameStoredLocally.get()) );
	_vShrPtrMPSV.push_back(pShrPtrMPSV);
	//construct vector of SPMV
	ushort usIdx = 0;
	for ( btl::geometry::tp_plane_obj_list::iterator itPlaneObj = pFrame_->_vPlaneObjsDistanceNormal[3].begin();itPlaneObj != pFrame_->_vPlaneObjsDistanceNormal[3].end(); itPlaneObj++, usIdx++ ){
		//construct a SPMV
		btl::geometry::CSinglePlaneMultiViewsInWorld::tp_shared_ptr pShrPtrSPMV( new btl::geometry::CSinglePlaneMultiViewsInWorld(*itPlaneObj,3,pShrKeyFrameStoredLocally.get(),usIdx ) ); 
		_vShrPtrSPMV.push_back(pShrPtrSPMV);
		//get SPSV from SPMV and store it in MPSV
		btl::geometry::CSinglePlaneSingleViewInWorld::tp_ptr pSPSV = pShrPtrSPMV->_vShrPtrSPSV[0].get();
		pShrPtrMPSV->_vPtrSPSV.push_back( pSPSV );
	}//for each plane obj
}//CMultiPlanesMultiViewsInWorld()
void btl::geometry::CMultiPlanesMultiViewsInWorld::fuse(CMultiPlanesSingleViewInWorld::tp_ptr pMPSV_, const ushort usPyrLevel_){
	btl::utility::COptimCamPose cOpt;
	ushort usPlaneCounts = pMPSV_->_vPtrSPSV.size();
	cOpt._cvmPlaneCur.create(4,usPlaneCounts);
	cOpt._cvmPlaneRef.create(4,usPlaneCounts);
	ushort usPlane = 0;
	//merge planes into previous SPMV
	for (tp_ptr_spsv_vec::iterator itSPSV = pMPSV_->_vPtrSPSV.begin(); itSPSV != pMPSV_->_vPtrSPSV.end(); itSPSV++, usPlane++ ) {
		BTL_ASSERT( _vShrPtrSPMV[(*itSPSV)->_usPlaneIdx]->_usPlaneIdx == (*itSPSV)->_usPlaneIdx, "incorrect correspondence established." );
		cOpt._cvmPlaneRef.at<double>(0,usPlane) = _vShrPtrSPMV[(*itSPSV)->_usPlaneIdx]->_aeivAvgNormal[3](0);
		cOpt._cvmPlaneRef.at<double>(1,usPlane) = _vShrPtrSPMV[(*itSPSV)->_usPlaneIdx]->_aeivAvgNormal[3](1);
		cOpt._cvmPlaneRef.at<double>(2,usPlane) = _vShrPtrSPMV[(*itSPSV)->_usPlaneIdx]->_aeivAvgNormal[3](2);
		cOpt._cvmPlaneRef.at<double>(3,usPlane) = _vShrPtrSPMV[(*itSPSV)->_usPlaneIdx]->_adAvgPosition[3];

		cOpt._cvmPlaneCur.at<double>(0,usPlane) = (*itSPSV)->_aeivAvgNormal[3](0);
		cOpt._cvmPlaneCur.at<double>(1,usPlane) = (*itSPSV)->_aeivAvgNormal[3](1);
		cOpt._cvmPlaneCur.at<double>(2,usPlane) = (*itSPSV)->_aeivAvgNormal[3](2);
		cOpt._cvmPlaneCur.at<double>(3,usPlane) = (*itSPSV)->_adAvgPosition[3];
	}//for eacn plane
	cOpt.Go();
	cOpt.getRT(&pMPSV_->_pFrame->_eimRw,&pMPSV_->_pFrame->_eivTw);
	pMPSV_->_pFrame->applyRelativePose(*_vShrPtrMPSV[0]->_pFrame);
}//fuse()
void btl::geometry::CMultiPlanesMultiViewsInWorld::integrateFrameIntoPlanesWorldCVCV( btl::kinect::CKeyFrame::tp_ptr pFrame_ ) {
	//store key frame
	btl::kinect::CKeyFrame::tp_shared_ptr pShrPtrFrameStoredLocally = btl::kinect::CKeyFrame::tp_shared_ptr(new btl::kinect::CKeyFrame(pFrame_));
	_vShrPtrKFrs.push_back(pShrPtrFrameStoredLocally);
	//construct new MPSV
	CMultiPlanesSingleViewInWorld::tp_shared_ptr pShrPtrMPSV( new CMultiPlanesSingleViewInWorld(pShrPtrFrameStoredLocally.get()) );
	//merge planes into previous SPMV
	for(tp_shr_spmv_vec::iterator itSPMV = _vShrPtrSPMV.begin(); itSPMV != _vShrPtrSPMV.end(); itSPMV++ ){
		//integrateFrameIntoPlanesWorldCVCV() will erase integrated plane objs
		(*itSPMV)->integrateFrameIntoPlanesWorldCVCV(pShrPtrFrameStoredLocally.get(),pFrame_->_vPlaneObjsDistanceNormal[3],3,pShrPtrMPSV.get());//3 special case
	}//for each SPMV
	
	//1.fuse the newly added frame 2.update SPMV: AvgNormal and AvgPosition
	//fuse();
	fuse(pShrPtrMPSV.get(),3);

	//insert the rest of plane objs as new SPMV
	if (pFrame_->_vPlaneObjsDistanceNormal[3].size()>0)	{
		ushort usPlaneIdx = _vShrPtrSPMV.size();
		for ( btl::geometry::tp_plane_obj_list::iterator itPlaneObj = pFrame_->_vPlaneObjsDistanceNormal[3].begin();itPlaneObj != pFrame_->_vPlaneObjsDistanceNormal[3].end(); itPlaneObj++, usPlaneIdx++ ){
			btl::geometry::CSinglePlaneMultiViewsInWorld::tp_shared_ptr pShrPtrSPMV( new btl::geometry::CSinglePlaneMultiViewsInWorld(*itPlaneObj,3,pShrPtrFrameStoredLocally.get(),usPlaneIdx ) ); 
			_vShrPtrSPMV.push_back(pShrPtrSPMV);
			//store the SPSV into the MPSV
			//pShrPtrMPSV->_vPtrSPSV.push_back(pShrPtrSPMV->_vShrPtrSPSV[0].get());
		}
	}//if other planes undetected left
	if (pShrPtrMPSV->_vPtrSPSV.size()>0){
		_vShrPtrMPSV.push_back(pShrPtrMPSV);
	}//if the new view has more than one plane detected
	return;
}//integrateFrameIntoPlanesWorldCVCV()

void btl::geometry::CMultiPlanesMultiViewsInWorld::renderAllPlanesInGivenViewWorldCVGL( btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_, const ushort usPyrLevel_, const ushort usView_ ){
	//if citMPSV hits end set as the beginning otherwise increase it by 1
	if (_vShrPtrMPSV.empty()) return;
	ushort usTmp = usView_%_vShrPtrMPSV.size();
	_vShrPtrMPSV[usTmp]->renderAllPlanesInSingleViewWorldCVGL(pGL_,usColorIdx_,usPyrLevel_);
}

void btl::geometry::CMultiPlanesMultiViewsInWorld::renderGivenPlaneInGivenViewWorldCVGL( btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_, const ushort usPyrLevel_, const ushort usView_, const ushort usPlane_ ){
	//if citMPSV hits end set as the beginning otherwise increase it by 1
	if (_vShrPtrMPSV.empty()) return;
	ushort usPlaneNOSafe = usPlane_ % _vShrPtrSPMV.size();
	ushort usViewNoSafe  = usView_  % _vShrPtrSPMV[usPlaneNOSafe]->_vShrPtrSPSV.size();
	_vShrPtrSPMV[usPlaneNOSafe]->renderSinglePlaneInSingleViewWorldGL(pGL_,usColorIdx_,usViewNoSafe,usPyrLevel_);
}

void btl::geometry::CMultiPlanesMultiViewsInWorld::renderGivenPlaneInAllViewWorldCVGL( btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_, const ushort usPyrLevel_, const ushort usPlane_ ){
	if (_vShrPtrMPSV.empty()) return;
	ushort usPlaneNOSafe = usPlane_ % _vShrPtrSPMV.size();
	_vShrPtrSPMV[usPlaneNOSafe]->renderSinglePlaneInAllViewsWorldGL(pGL_,usColorIdx_,usPyrLevel_);
}

void btl::geometry::CMultiPlanesMultiViewsInWorld::renderAllCamrea(btl::gl_util::CGLUtil::tp_ptr pGL_,bool bBW_, bool bRenderDepth_, float fSize_ ){
	for (tp_shr_kfrm_vec::const_iterator citFrame = _vShrPtrKFrs.begin(); citFrame != _vShrPtrKFrs.end(); citFrame++){
		(*citFrame)->renderCameraInGLWorld(pGL_->_bDisplayCamera,bBW_,bRenderDepth_,fSize_,pGL_->_uLevel);
	}//for each keyframe
}





