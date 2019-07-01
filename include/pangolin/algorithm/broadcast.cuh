#pragma once

namespace pangolin {

/*! Broadcast value from threadIdx.x root to all threads in the warp
 */
template <size_t WARPS_PER_BLOCK, typename T> __device__ __forceinline__ T warp_broadcast(const T val, const int root) {

  static_assert(sizeof(T) % sizeof(int) == 0, "need to zero out some bytes of ints");

  // int lx = threadIdx.x % 32;
  // int wx = threadIdx.x / 32;

  union Ints {
    T t;
    int ints[sizeof(T) / sizeof(int)];
  };

  Ints ints;
  ints.t = val;

  // shuffle one int at a time

#if __CUDACC_VER_MAJOR__ >= 9
#pragma unroll
  for (size_t i = 0; i < sizeof(val) / sizeof(int); i++) {
    ints.ints[i] = __shfl_sync(0xffffffff /* all threads */, ints.ints[i], root);
  }
#else
#pragma unroll
  for (size_t i = 0; i < sizeof(val) / sizeof(int); i++) {
    ints.ints[i] = __shfl(ints.ints[i], root);
  }
#endif

  return ints.t;
}

/*! Broadcast value from threadIdx.x root to all threads in the warp
 */
template <typename T> __device__ __forceinline__ T warp_broadcast2(T val, const int root) {

#if __CUDACC_VER_MAJOR__ >= 9
  val = __shfl_sync(0xffffffff /* all threads */, val, root);
#else
  val = __shfl(val, root);
#endif

  return val;
}

/*! Broadcast value from threadIdx.x root to all threads in the warp
 */
template <typename T> __device__ __forceinline__ T *warp_broadcast2(T *val, const int root) {
  uintptr_t upval = reinterpret_cast<uintptr_t>(val);
  return reinterpret_cast<T *>(warp_broadcast2(upval, root));
}

/*! Broadcast value from threadIdx.x root to all threads in the block
 */
template <typename T> __device__ __forceinline__ T block_broadcast(const T val, const int root) {
  __shared__ T sharedVal;

  if (threadIdx.x == root) {
    sharedVal = val;
  }
  __syncthreads();
  return sharedVal;
}

/*! Broadcast value from threadIdx.x root to all threads in the block
 */
template <typename T> __device__ T block_broadcast2(const T val, const int root) {
  __shared__ T sharedVal;

  if (threadIdx.x == root) {
    sharedVal = val;
  }
  __syncthreads();
  return warp_broadcast2(sharedVal, 0);
}

} // namespace pangolin