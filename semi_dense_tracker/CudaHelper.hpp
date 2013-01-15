#ifndef CUDA_HELPER_BTL
#define CUDA_HELPER_BTL



__device__ unsigned int _devuCounter = 0;

__device__ unsigned int _devuNewlyAddedCounter = 0;

__device__ unsigned int _devuOther = 0;

__device__ unsigned int _devuTest1 = 0;

  __device__ __host__ __forceinline__ short2 operator + (const short2 s2O1_, const short2 s2O2_){
	return make_short2(s2O1_.x + s2O2_.x,s2O1_.y + s2O2_.y);
}
  __device__ __host__ __forceinline__ short2 operator - (const short2 s2O1_, const short2 s2O2_){ //can be called from host and device
	return make_short2(s2O1_.x - s2O2_.x,s2O1_.y - s2O2_.y);
}
  __device__ __forceinline__ float2 operator * (const float fO1_, const short2 s2O2_){
	return make_float2( fO1_* s2O2_.x, fO1_ * s2O2_.y);
}
  __device__ __host__ __forceinline__ float2 operator + (const float2 f2O1_, const float2 f2O2_){ //can be called from host and device
	  return make_float2(f2O1_.x + f2O2_.x,f2O1_.y + f2O2_.y);
}
  __device__ __host__ __forceinline__ float2 operator - (const float2 f2O1_, const float2 f2O2_){ //can be called from host and device
	  return make_float2(f2O1_.x - f2O2_.x,f2O1_.y - f2O2_.y);
}
  __device__ __host__ __forceinline__ int4 operator + (const int4 n4O1_, const int4 n4O2_){
	  return make_int4(n4O1_.x + n4O2_.x, n4O1_.y + n4O2_.y, n4O1_.z+n4O2_.z, n4O1_.w+n4O2_.w);
  }
  __device__ __host__ __forceinline__ int4 operator - (const int4 n4O1_, const int4 n4O2_){
	  return make_int4(n4O1_.x - n4O2_.x, n4O1_.y - n4O2_.y, n4O1_.z-n4O2_.z, n4O1_.w-n4O2_.w);
  }

  __device__  __forceinline__ short2 convert2s2(const float2 f2O1_){ //can be called from host and device
	  return make_short2(__float2int_rn(f2O1_.x), __float2int_rn(f2O1_.y));
  }
 





#endif