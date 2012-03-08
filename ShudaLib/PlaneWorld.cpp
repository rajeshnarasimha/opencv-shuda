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


btl::geometry::SSinglePlaneSingleViewInWorld::SSinglePlaneSingleViewInWorld(const btl::geometry::SPlaneObj& sPlaneObj_, ushort usPyrLevel_, btl::kinect::CKeyFrame::tp_ptr pFrame_ ) 
:_pFrame(pFrame_){
	_aeivAvgNormal[usPyrLevel_] = sPlaneObj_._eivAvgNormal;
	_adAvgPosition[usPyrLevel_] = sPlaneObj_._dAvgPosition;
	_avIdx[usPyrLevel_].reset( new std::vector<unsigned int>(sPlaneObj_._vIdx.begin(),sPlaneObj_._vIdx.end()) ); 
}//SPlaneW()
void btl::geometry::SSinglePlaneSingleViewInWorld::renderInWorldCVGL(btl::gl_util::CGLUtil::tp_ptr pGL_, const uchar* pColor_, const ushort usPyrLevel_ ) const {
	glPushMatrix();
	_pFrame->loadGLMVIn(); //setup the modelview matrix for opengl
	const float* pPt = (const float*) _pFrame->_acvmShrPtrPyrPts[usPyrLevel_]->data;
	const float* pNl = (const float*) _pFrame->_acvmShrPtrPyrNls[usPyrLevel_]->data;
	glBegin(GL_POINTS);
	_pFrame->renderASinglePlaneObjInLocalCVGL(pPt,pNl,*_avIdx[usPyrLevel_],pColor_);
	glEnd();
	glPopMatrix();
}//renderInWorldCVGL()

bool btl::geometry::SSinglePlaneSingleViewInWorld::identical( const Eigen::Vector3d& eivNormal_, const double dPosition_, const ushort usPyrLevel_ ) const {
	double dCos = eivNormal_.dot( _aeivAvgNormal[usPyrLevel_] );
	double dDif = fabs( dPosition_ - _adAvgPosition[usPyrLevel_] );
	if(dCos > 0.80 && dDif < 0.1 ) return true; 
	else return false;
}

btl::geometry::CSinglePlaneMultiViewsInWorld::CSinglePlaneMultiViewsInWorld( const btl::geometry::SPlaneObj& sPlaneObj_, const ushort usPyrLevel_, btl::kinect::CKeyFrame::tp_ptr pFrame_ ){
		btl::geometry::SSinglePlaneSingleViewInWorld::tp_shared_ptr pShrPtrSPSV( new btl::geometry::SSinglePlaneSingleViewInWorld(sPlaneObj_,usPyrLevel_,pFrame_) );
		_vShrPtrSPSV.push_back(pShrPtrSPSV);
		for (ushort u=0; u<4; u++){
			_aeivAvgNormal[u] = sPlaneObj_._eivAvgNormal;
			_adAvgPosition[u] = sPlaneObj_._dAvgPosition;
		}
}//CSinglePlaneMultiViewsInWorld()
bool btl::geometry::CSinglePlaneMultiViewsInWorld::identical( const Eigen::Vector3d& eivNormal_, const double dPosition_, const ushort usPyrLevel_ ) const {
	double dCos = eivNormal_.dot( _aeivAvgNormal[usPyrLevel_] );
	double dDif = fabs( dPosition_ - _adAvgPosition[usPyrLevel_] );
	if(dCos > 0.80 && dDif < 0.1 ) return true; 
	else return false;
}
void btl::geometry::CSinglePlaneMultiViewsInWorld::integrateFrameIntoPlanesWorldCVCV( btl::kinect::CKeyFrame::tp_ptr pFrame_, btl::geometry::tp_plane_obj_list& lPlanes_, const ushort usPyrLevel_){
	//find the close plane objects
	btl::geometry::SSinglePlaneSingleViewInWorld::tp_shared_ptr pShrPtrSPSV;
	for (btl::geometry::tp_plane_obj_list::iterator itPlaneObj = lPlanes_.begin();itPlaneObj != lPlanes_.end();itPlaneObj++){
		if ( !itPlaneObj->_bCorrespondetFound && identical( itPlaneObj->_eivAvgNormal,itPlaneObj->_dAvgPosition, usPyrLevel_ ) ){
			if(!pShrPtrSPSV.get()){
				pShrPtrSPSV.reset( new btl::geometry::SSinglePlaneSingleViewInWorld(*itPlaneObj,usPyrLevel_,pFrame_) );
			}//pShrPtrSPSV not yet assigned
			else{
				pShrPtrSPSV->_avIdx[usPyrLevel_]->insert(pShrPtrSPSV->_avIdx[usPyrLevel_]->end(), itPlaneObj->_vIdx.begin(),itPlaneObj->_vIdx.end() );
			}//else add index to exiting pShrPtrSPSV
			//erase detected plane
			itPlaneObj->_bCorrespondetFound = true;
		}// if identical to the current plane
	}//for each plane world in pFrame_
	if(pShrPtrSPSV.get()){
		_vShrPtrSPSV.push_back(pShrPtrSPSV);
	}//if identical plane found
	return;
}//integrateFrameIntoPlanesWorldCVCV()
void btl::geometry::CSinglePlaneMultiViewsInWorld::renderPlaneInAllViewsWorldGL(btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_,const ushort usPyrLevel_ /*= 3*/) const {
	
	const unsigned char* pColor = btl::utility::__aColors[_usIdx+usColorIdx_%BTL_NUM_COLOR];
	if( pGL_ && pGL_->_bEnableLighting ){glEnable(GL_LIGHTING);}
	else                            	{glDisable(GL_LIGHTING);}
	for (tp_shr_spsv_vec::const_iterator citShrPtrSPSV = _vShrPtrSPSV.begin(); citShrPtrSPSV!= _vShrPtrSPSV.end(); citShrPtrSPSV++){
		(*citShrPtrSPSV)->renderInWorldCVGL(pGL_,pColor,usPyrLevel_);
	}//for each plane
}//renderPlaneInWorldGL()

void btl::geometry::CSinglePlaneMultiViewsInWorld::renderPlaneInSingleViewWorldGL( btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_,const ushort usView_, const ushort usPyrLevel_ /*= 3*/ ) const
{
	const unsigned char* pColor = btl::utility::__aColors[_usIdx+usColorIdx_%BTL_NUM_COLOR];
	if( pGL_ && pGL_->_bEnableLighting ){glEnable(GL_LIGHTING);}
	else                            	{glDisable(GL_LIGHTING);}
	_vShrPtrSPSV[usView_]->renderInWorldCVGL(pGL_,pColor,usPyrLevel_);
}

btl::geometry::CMultiPlanesMultiViewsInWorld::CMultiPlanesMultiViewsInWorld( btl::kinect::CKeyFrame::tp_ptr pFrame_ ){
	btl::kinect::CKeyFrame::tp_shared_ptr pShrKeyFrameStoredLocally (new btl::kinect::CKeyFrame(pFrame_) );
	_vShrPtrKeyFrames.push_back(pShrKeyFrameStoredLocally);
	ushort usIdx = 0;
	for ( btl::geometry::tp_plane_obj_list::iterator itPlaneObj = pFrame_->_vPlaneObjsDistanceNormal[3].begin();itPlaneObj != pFrame_->_vPlaneObjsDistanceNormal[3].end(); itPlaneObj++, usIdx++ ){
		btl::geometry::CSinglePlaneMultiViewsInWorld::tp_shared_ptr pShrPtrSPMV( new btl::geometry::CSinglePlaneMultiViewsInWorld(*itPlaneObj,3,pShrKeyFrameStoredLocally.get() ) ); 
		pShrPtrSPMV->_usIdx = usIdx;
		_vShrPtrSPMV.push_back(pShrPtrSPMV);
	}
	
}//CMultiPlanesMultiViewsInWorld()

void btl::geometry::CMultiPlanesMultiViewsInWorld::integrateFrameIntoPlanesWorldCVCV( btl::kinect::CKeyFrame::tp_ptr pFrame_ ) {
	btl::kinect::CKeyFrame::tp_shared_ptr pShrFrameStoredLocally = btl::kinect::CKeyFrame::tp_shared_ptr(new btl::kinect::CKeyFrame(pFrame_));
	_vShrPtrKeyFrames.push_back(pShrFrameStoredLocally);
	//make _bCorrespondetFound false
	for( btl::geometry::tp_plane_obj_list::iterator it = pFrame_->_vPlaneObjsDistanceNormal[3].begin(); it != pFrame_->_vPlaneObjsDistanceNormal[3].end(); it++ ) it->_bCorrespondetFound = false;
		
	for(tp_shr_spmv_vec::iterator itSPMV = _vShrPtrSPMV.begin(); itSPMV != _vShrPtrSPMV.end(); itSPMV++ ){
		//integrateFrameIntoPlanesWorldCVCV() will erase integrated plane objs
		(*itSPMV)->integrateFrameIntoPlanesWorldCVCV(pShrFrameStoredLocally.get(),pFrame_->_vPlaneObjsDistanceNormal[3],3);//3 special case
	}//for each plane
	return;
}//integrateFrameIntoPlanesWorldCVCV()

void btl::geometry::CMultiPlanesMultiViewsInWorld::renderASinglePlaneInWorldGL(btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_, const ushort usPyrLevel_ ) const {

}
void btl::geometry::CMultiPlanesMultiViewsInWorld::renderAllPlanesInSingleViewWorldGL(btl::gl_util::CGLUtil::tp_ptr pGL_, const ushort usColorIdx_, const ushort usPyrLevel_, const ushort usView_ ) const {
	for (tp_shr_spmv_vec::const_iterator citPlane= _vShrPtrSPMV.begin(); citPlane != _vShrPtrSPMV.end(); citPlane++ ) {
		(*citPlane)->renderPlaneInAllViewsWorldGL(pGL_,usColorIdx_,usPyrLevel_);
	}//for each plane
	
}


