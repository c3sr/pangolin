#pragma once

#include <cstdlib>
#include <new>

namespace pangolin {

namespace allocator {

template <class T> struct CUDAMalloc {
  typedef T value_type;
  CUDAMalloc() = default;
  template <class U>
  constexpr CUDAMalloc(const CUDAMalloc<U> &) noexcept {}
  T *allocate(std::size_t n) {
    if (n > std::size_t(-1) / sizeof(T))
      throw std::bad_alloc();
    T *p;
    cudaError_t err = cudaMalloc(&p, n * sizeof(T));
    if (err != cudaSuccess) {
      throw std::bad_alloc();
    }
    return p;
  }
  void deallocate(T *p, std::size_t) noexcept { cudaFree(p); }
};
template <class T, class U>
bool operator==(const CUDAMalloc<T> &,
                const CUDAMalloc<U> &) {
  return true;
}
template <class T, class U>
bool operator!=(const CUDAMalloc<T> &,
                const CUDAMalloc<U> &) {
  return false;
}

}

} // namespace pangolin