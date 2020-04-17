#pragma once

template <typename T, typename Index> __device__ __forceinline__ void bitmap_set_atomic(T *bitmap, Index i) {
    Index field = i / (sizeof(T) * CHAR_BIT);
    Index bit = i % (sizeof(T) * CHAR_BIT);
    T bits = T(1) << bit;
    atomicOr(&bitmap[field], bits);
  }
  
  template <typename T, typename Index> __device__ __forceinline__ bool bitmap_get(T *bitmap, Index i) {
    Index fieldIdx = i / Index(sizeof(T) * CHAR_BIT);
    Index bitIdx = i % Index(sizeof(T) * CHAR_BIT);
    T bits = bitmap[fieldIdx];
    return (bits >> bitIdx) & T(1);
  }
  
  // clear all bits between [first, second]
  // may reset more bits than that
  template <typename T, typename Index> __device__ void block_bitmap_clear(T *bitmap, Index first, Index second) {
    const Index firstIdx = first / Index(sizeof(T) * CHAR_BIT);
    const Index secondIdx = second / Index(sizeof(T) * CHAR_BIT);
    for (Index i = firstIdx + threadIdx.x; i <= secondIdx; i += blockDim.x) {
      bitmap[i] = 0;
    }
  }
  
  // clear all bits between [first, second]
  // may reset more bits than that
  template <typename T, typename Index>
  __device__ void warp_bitmap_clear(T *bitmap, Index first, Index second, const size_t lx) {
    const Index firstIdx = first / Index(sizeof(T) * CHAR_BIT);
    const Index secondIdx = second / Index(sizeof(T) * CHAR_BIT);
    for (Index i = firstIdx + lx; i <= secondIdx; i += 32) {
      bitmap[i] = 0;
    }
  }
  
  // clear bit i
  // may clear more bits than that
  template <typename T, typename Index> __device__ __forceinline__ void bitmap_clear(T *bitmap, Index i) {
    Index fieldIdx = i / Index(sizeof(T) * CHAR_BIT);
    bitmap[fieldIdx] = 0;
  }