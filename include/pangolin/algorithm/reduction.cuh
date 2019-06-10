#pragma once

namespace pangolin {

template <typename T> __device__ __forceinline__ T warp_sum(T val) {

#if __CUDACC_VER_MAJOR__ >= 9
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
#else
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down(0xffffffff, val, offset);
  }
#endif

  return val;
}

} // namespace pangolin