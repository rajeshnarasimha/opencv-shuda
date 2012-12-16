#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

float dL1(const int4& n4Descriptor1_, const int4& n4Descriptor2_){
	float fDist = 0.f;
	for (uchar u=0; u < 4; u++){
		uchar uD1 = (n4Descriptor1_ >> u*8) & 0xFF;
		uchar uD2 = (n4Descriptor2_ >> u*8) & 0xFF;
		 fDist += abs(uD2 - uD2); 
	}
	fDist /= 4;
	return fDist;
}

void devGetFastDescriptor(const cv::Mat& cvgmImage_, const int r, const int c, int4* pDescriptor_ ){
	pDescriptor_->x = pDescriptor_->y = pDescriptor_->z = pDescriptor_->w = 0;
	uchar3 Color;
	Color = cvgmImage_.ptr<uchar3>(r-3)[c  ];//1
	pDescriptor_->x += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->x = pDescriptor_->x << 8;
	Color = cvgmImage_.ptr<uchar3>(r-3)[c+1];//2
	pDescriptor_->x += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->x = pDescriptor_->x << 8;
	Color = cvgmImage_.ptr<uchar3>(r-2)[c+2];//3
	pDescriptor_->x += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->x = pDescriptor_->x << 8;
	Color = cvgmImage_.ptr<uchar3>(r-1)[c+3];//4
	pDescriptor_->x += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	
	Color = cvgmImage_.ptr<uchar3>(r  )[c+3];//5
	pDescriptor_->y += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->y = pDescriptor_->y << 8;
	Color = cvgmImage_.ptr<uchar3>(r+1)[c+3];//6
	pDescriptor_->y += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->y = pDescriptor_->y << 8;
	Color = cvgmImage_.ptr<uchar3>(r+2)[c+2];//7
	pDescriptor_->y += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->y = pDescriptor_->y << 8;
	Color = cvgmImage_.ptr<uchar3>(r+3)[c+1];//8
	pDescriptor_->y += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 

	Color = cvgmImage_.ptr<uchar3>(r+3)[c  ];//9
	pDescriptor_->z += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->z = pDescriptor_->z << 8;
	Color= cvgmImage_.ptr<uchar3>(r+3)[c-1];//10
	pDescriptor_->z += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->z = pDescriptor_->z << 8;
	Color= cvgmImage_.ptr<uchar3>(r+2)[c-2];//11
	pDescriptor_->z += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->z = pDescriptor_->z << 8;
	Color= cvgmImage_.ptr<uchar3>(r+1)[c-3];//12
	pDescriptor_->z += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 

	Color= cvgmImage_.ptr<uchar3>(r  )[c-3];//13
	pDescriptor_->w += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->w = pDescriptor_->w << 8;
	Color= cvgmImage_.ptr<uchar3>(r-1)[c-3];//14
	pDescriptor_->w += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->w = pDescriptor_->w << 8;
	Color= cvgmImage_.ptr<uchar3>(r-2)[c-2];//15
	pDescriptor_->w += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	pDescriptor_->w = pDescriptor_->w << 8;
	Color= cvgmImage_.ptr<uchar3>(r-3)[c-1];//16
	pDescriptor_->w += static_cast<uchar>((Color.x + Color.y + Color.z)/3.f); 
	return;
}


void main()
{
	//opencv cpp style
#ifdef WEB_CAM
	cv::VideoCapture cap ( 1 ); // 0: open the default camera
	// 1: open the integrated webcam
#else
	cv::VideoCapture cap ( "VTreeTrunk.avi" ); 
#endif

	if ( !cap.isOpened() ) return;
	cv::Mat cvmColorFrame;
	cap >> cvmColorFrame;
	int4 n4Des1;
	devGetFastDescriptor(cvmColorFrame,5,5,&n4Des1);
	int4 n4Des2;
	devGetFastDescriptor(cvmColorFrame,5,5,&n4Des2);
	float fDist = dL1(n4Des1, n4Des2);
}