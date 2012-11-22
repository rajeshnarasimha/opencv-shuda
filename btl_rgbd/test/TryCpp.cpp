#include "../Converters.hpp"
#include "../CVUtil.hpp"
#include "../EigenUtil.hpp"
#include "TryCpp.h"

void tryStdVectorResize()
{
	PRINTSTR("try std::vector::resize() whether it allocate memory");
	std::vector<int> vInt;
	vInt.resize(3);
	PRINT(vInt);
	vInt[2]=10;
	PRINT(vInt);

	PRINTSTR("try std::vector< <> >::resize() whether it allocate memory");
	std::vector< std::vector< int > > vvIdx;
	vvIdx.resize(3);
	vvIdx[2].push_back(1);
	PRINT(vvIdx);
}
void tryStdVectorConstructor()
{
	PRINTSTR("try std::vector< <> >::vector()");
	std::vector< int > vInt(5,1);
	PRINT( vInt );
}

enum tp_flag { NO_MERGE, MERGE_WITH_LEFT, MERGE_WITH_RIGHT, MERGE_WITH_BOTH };
void tryStdVectorEnum()
{
	PRINTSTR("try std::vector< enum >");
	std::vector<tp_flag > vMergeFlags(2,NO_MERGE);
	vMergeFlags[1] = MERGE_WITH_BOTH;
	PRINT(vMergeFlags);
}
void tryStdVector()
{
	tryStdVectorResize();
	tryStdVectorConstructor();
	tryStdVectorEnum();
}
void tryStdLimits(){
	PRINTSTR("try std::limits");
	float fQNaN = std::numeric_limits<float>::quiet_NaN();
	PRINT(fQNaN);
	float fSNaN = std::numeric_limits<float>::signaling_NaN();
	PRINT(fSNaN);
	float fInf  = std::numeric_limits<float>::infinity();
	PRINT(fInf);
	PRINT(fSNaN<10.f);
	PRINT(fSNaN>10.f);
	PRINT(fInf>10.f);
	PRINT(fInf<10.f);
	PRINT(-fInf<10.f);
	PRINT(fQNaN - 1);
}
void tryStdList(){
	PRINTSTR("try std::list");
	typedef unsigned int uint;
	std::list<uint> lTmp;
	for (unsigned int i=0; i<10; i++){
		lTmp.push_back(i);
	}
	PRINT(lTmp);
	std::list<uint>::iterator itErase;
	bool bErase = false;
	for (std::list<uint>::iterator itNum = lTmp.begin(); itNum != lTmp.end(); itNum++ ){
		if( bErase ){
			lTmp.erase(itErase);
			bErase = false;
		}//remove after itNum increased
		if ((*itNum%2)==1)	{
			itErase= itNum;
			bErase = true;
		}//store 
	}
	if( bErase ){
		lTmp.erase(itErase);
		bErase = false;
	}//remove after itNum increased
	PRINT(lTmp);
}

void tryCppSizeof()
{
	PRINTSTR("try sizeof() operator");
	PRINT(sizeof(long double));
	PRINT(sizeof(double));
	PRINT(sizeof(short));
}
void tryCppTypeDef()
{
	PRINTSTR("try cpp keyword typedef");
	{
		typedef int tp_int;
		tp_int n;
		n = 1;
		PRINT( n );
		{
			tp_int m;
			m = 2;
			PRINT( m );
		}
	}
	//tp_int o; compile error
}
void tryCppBitwiseShift()
{
	PRINTSTR("tryCppBitwiseShift():")
		int n=10;
	int m=3;
	PRINT(n);
	PRINTSTR("n << m");
	n = n << m;
	PRINT(n);

	unsigned short usSamples=3;
	const unsigned short usSamplesElevationZ = 1<<usSamples; //2^usSamples
	const unsigned short usSamplesAzimuthX = usSamplesElevationZ<<1;   //usSamplesElevationZ*2
	const unsigned short usSamplesAzimuthY = usSamplesElevationZ<<1;   //usSamplesElevationZ*2
	const unsigned short usWidth = usSamplesAzimuthX;				    //
	const unsigned short usLevel = usSamplesAzimuthX<<(usSamples+1);	//usSamplesAzimuthX*usSamplesAzimuthX
	const unsigned short usTotal = usLevel<<(usSamples);  //usSamplesAzimuthX*usSamplesAzimuthY*usSamplesElevationZ
	PRINT(usSamples);
	PRINT(usSamplesElevationZ);
	PRINT(usSamplesAzimuthX);
	PRINT(usSamplesAzimuthY);
	PRINT(usLevel);
	PRINT(usSamplesAzimuthX*usSamplesAzimuthY);
	PRINT(usTotal);
	PRINT(usSamplesAzimuthX*usSamplesAzimuthY*usSamplesElevationZ);
	unsigned short usX = 3;
	unsigned short usY = 7;
	unsigned short usZ = 7;
	PRINT(usX);
	PRINT(usY);
	PRINT(usZ);
	PRINT(usZ*usLevel+usY*usWidth+usX);

}

void tryCppLongDouble()
{
	PRINTSTR("try long double and double type the effective digits");
	PRINT( std::numeric_limits<long double>::digits10 );
	PRINT( std::numeric_limits<double>::digits10);
}

void tryCppOperator()
{
	std::cout << "try: >> / <<" << std::endl;
	int nL = 1; 
	PRINT(nL);
	PRINT(nL<<1);
	PRINT(nL<<2);
	PRINT(nL<<3);
}



struct event_cb;
typedef void (*event_cb_t)(const struct event_cb *evt, void *user_data);

struct event_cb
{
	event_cb_t cb;
	void *data;
};

static struct event_cb saved = { 0, 0 };

void event_cb_register(event_cb_t cb, void *user_data)
{
	saved.cb = cb;
	saved.data = user_data;
}
class CEvent{
public:
static void my_event_cb(const struct event_cb *evt, void *data)
{
	//printf("in %s\n", __func__);
	//std::cout <<" in "<< __func__ << std::endl; 
	//printf("data1: %s\n", (const char *)data);
	std::cout <<"data1: " << (const char *)data << std::endl;
	//printf("data2: %s\n", (const char *)evt->data);
	std::cout <<"data2: " << (const char *)evt->data << std::endl;
}
};

void tryCppCallbackFunction()
{
	//http://stackoverflow.com/questions/631273/function-pointers-callbacks-c

	char my_custom_data[40] = "Hello!";
	event_cb_register(CEvent::my_event_cb, my_custom_data);

	saved.cb(&saved, saved.data);
	return;
}



void tryCpp()
{
	/*tryStdList();
	tryStdLimits();
	tryStdVector();
	tryStdVectorEnum();

	tryCppBitwiseShift();
	tryCppOperator();
	tryCppLongDouble();
	tryCppSizeof();
	tryCppTypeDef();*/
	tryCppCallbackFunction();
}