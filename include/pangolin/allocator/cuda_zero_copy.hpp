#pragma once

#include <cstdlib>
#include <new>

namespace pangolin {

namespace allocator {

template <class T> struct CUDAZeroCopy {
  typedef T value_type;
  CUDAZeroCopy() = default;
  template <class U> constexpr CUDAZeroCopy(const CUDAZeroCopy<U> &) noexcept {}
  T *allocate(std::size_t n) {
    if (n > std::size_t(-1) / sizeof(T))
      throw std::bad_alloc();
    T *p;
    cudaError_t err = cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) {
      throw std::bad_alloc();
    }
    return p;
  }
  void deallocate(T *p, std::size_t) noexcept { cudaFreeHost(p); }
};
template <class T, class U> bool operator==(const CUDAZeroCopy<T> &, const CUDAZeroCopy<U> &) { return true; }
template <class T, class U> bool operator!=(const CUDAZeroCopy<T> &, const CUDAZeroCopy<U> &) { return false; }

} // namespace allocator

} // namespace pangolin