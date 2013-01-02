
#include <cuda.h>
#include <cuda_runtime.h>
#include "Helper.hpp"

#define __float2int_rn short

__device__ short2 operator + (const short2 s2O1_, const short2 s2O2_){
	return make_short2(s2O1_.x + s2O2_.x,s2O1_.y + s2O2_.y);
}
__device__ short2 operator - (const short2 s2O1_, const short2 s2O2_){
	return make_short2(s2O1_.x - s2O2_.x,s2O1_.y - s2O2_.y);
}
__device__ short2 operator * (const float fO1_, const short2 s2O2_){
	return make_short2( __float2int_rn(fO1_* s2O2_.x),__float2int_rn( fO1_ * s2O2_.y));
}