#ifndef BTL_KINECT
#define BTL_KINECT


namespace btl{
namespace kinect{

#define KINECT_WIDTH 640
#define KINECT_WIDTH_L1 320
#define KINECT_WIDTH_L2 160
#define KINECT_WIDTH_L3 80
#define KINECT_HEIGHT 480
#define KINECT_HEIGHT_L1 240
#define KINECT_HEIGHT_L2 120
#define KINECT_HEIGHT_L3 60

#define KINECT_WxH 307200
#define KINECT_WxH_L1 76800 //320*240
#define KINECT_WxH_L2 19200 //160*120
#define KINECT_WxH_L3 4800  // 80*60

#define KINECT_WxHx3 921600
#define KINECT_WxHx3_L1 230400 
#define KINECT_WxHx3_L2 57600

static unsigned int __aKinectWxH[4] = {KINECT_WxH,   KINECT_WxH_L1,   KINECT_WxH_L2,   KINECT_WxH_L3};
static unsigned short __aKinectW[4] = {KINECT_WIDTH, KINECT_WIDTH_L1, KINECT_WIDTH_L2, KINECT_WIDTH_L3};
static unsigned short __aKinectH[4] = {KINECT_HEIGHT,KINECT_HEIGHT_L1,KINECT_HEIGHT_L2,KINECT_HEIGHT_L3};



}//kinect
}//btl
#endif
