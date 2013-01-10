
#include <cuda.h>
#include <cuda_runtime.h>
#include "Helper.hpp"

#define __float2int_rn short

__device__ __host__ short2 operator + (const short2 s2O1_, const short2 s2O2_){
	return make_short2(s2O1_.x + s2O2_.x,s2O1_.y + s2O2_.y);
}
__device__ __host__ short2 operator - (const short2 s2O1_, const short2 s2O2_){
	return make_short2(s2O1_.x - s2O2_.x,s2O1_.y - s2O2_.y);
}
__device__ float2 operator * (const float fO1_, const short2 s2O2_){
	return make_float2( fO1_* s2O2_.x, fO1_ * s2O2_.y );
}
__device__ __host__ float2 operator + (const float2 f2O1_, const float2 f2O2_){ //can be called from host and device
	return make_float2(f2O1_.x + f2O2_.x,f2O1_.y + f2O2_.y);
}
__device__ __host__ float2 operator - (const float2 f2O1_, const float2 f2O2_){ //can be called from host and device
	return make_float2(f2O1_.x - f2O2_.x,f2O1_.y - f2O2_.y);
}
__device__  short2 convert2s2(const float2 f2O1_){ //can be called from host and device
	return make_short2(__float2int_rn(f2O1_.x), __float2int_rn(f2O1_.y));
}