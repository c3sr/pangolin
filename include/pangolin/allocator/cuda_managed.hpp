#pragma once

#include <cstdlib>
#include <new>

#include <cuda_runtime.h>

template <class T> struct CUDAManagedAllocator {
  typedef T value_type;
  CUDAManagedAllocator() = default;
  template <class U>
  constexpr CUDAManagedAllocator(const CUDAManagedAllocator<U> &) noexcept {}
  T *allocate(std::size_t n) {
    if (n > std::size_t(-1) / sizeof(T))
      throw std::bad_alloc();
    T *p;
    cudaError_t err = cudaMallocManaged(&p, n * sizeof(T));
    if (err != cudaSuccess) {
      throw std::bad_alloc();
    }
    return p;
  }
  void deallocate(T *p, std::size_t) noexcept { cudaFree(p); }
};
template <class T, class U>
bool operator==(const CUDAManagedAllocator<T> &,
                const CUDAManagedAllocator<U> &) {
  return true;
}
template <class T, class U>
bool operator!=(const CUDAManagedAllocator<T> &,
                const CUDAManagedAllocator<U> &) {
  return false;
}