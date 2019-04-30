#pragma once

#include <cstdlib>
#include <new>

#include <cuda_runtime.h>

template <class T> struct CUDAZeroCopyAllocator {
  typedef T value_type;
  CUDAZeroCopyAllocator() = default;
  template <class U>
  constexpr CUDAZeroCopyAllocator(const CUDAZeroCopyAllocator<U> &) noexcept {}
  T *allocate(std::size_t n) {
    if (n > std::size_t(-1) / sizeof(T))
      throw std::bad_alloc();
    T *p;
    cudaError_t err = cudaHostAlloc(
        &p, n * sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) {
      throw std::bad_alloc();
    }
    return p;
  }
  void deallocate(T *p, std::size_t) noexcept { cudaFreeHost(p); }
};
template <class T, class U>
bool operator==(const CUDAZeroCopyAllocator<T> &,
                const CUDAZeroCopyAllocator<U> &) {
  return true;
}
template <class T, class U>
bool operator!=(const CUDAZeroCopyAllocator<T> &,
                const CUDAZeroCopyAllocator<U> &) {
  return false;
}