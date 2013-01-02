#ifndef CUDA_HELPER_BTL
#define CUDA_HELPER_BTL



__device__ unsigned int _devuCounter = 0;

__device__ unsigned int _devuNewlyAddedCounter = 0;

__device__ unsigned int _devuOther = 0;

__device__ unsigned int _devuTest1 = 0;

  __device__ __host__ __forceinline__ short2 operator + (const short2 s2O1_, const short2 s2O2_){
	return make_short2(s2O1_.x + s2O2_.x,s2O1_.y + s2O2_.y);
}
  __device__ __host__ __forceinline__ short2 operator - (const short2 s2O1_, const short2 s2O2_){
	return make_short2(s2O1_.x - s2O2_.x,s2O1_.y - s2O2_.y);
}
  __device__ __forceinline__ short2 operator * (const float fO1_, const short2 s2O2_){
	return make_short2( __float2int_rn(fO1_* s2O2_.x),__float2int_rn( fO1_ * s2O2_.y));
}




#endif