#ifndef OTHER_BTL
#define OTHER_BTL

namespace btl{ namespace other{


template <class T>
void increase(const T nCycle_, T* pnIdx_ ){
	++*pnIdx_;
	*pnIdx_ = *pnIdx_ < nCycle_? *pnIdx_: *pnIdx_-nCycle_;
}
template <class T>
void decrease(const T nCycle_, T* pnIdx_ ){
	--*pnIdx_;
	*pnIdx_ = *pnIdx_ < 0?       *pnIdx_+nCycle_: *pnIdx_;
}




}//other
}//btl




#endif