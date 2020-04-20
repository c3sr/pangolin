// github.com/cwpearson/cuda-memory-management

#pragma once

#include <cstdlib>
#include <new>

namespace cmm {

namespace detail {

/* round x up to nearest multiple of `up`.
Up must not be 0
*/
inline size_t round_up(size_t x, size_t up) { return (x + up - 1) / up * up; }
} // namespace detail

template <class T> struct ZeroCopy {
  typedef T value_type;
  ZeroCopy() = default;
  template <class U> constexpr ZeroCopy(const ZeroCopy<U> &) noexcept {}
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
template <class T, class U> bool operator==(const ZeroCopy<T> &, const ZeroCopy<U> &) { return true; }
template <class T, class U> bool operator!=(const ZeroCopy<T> &, const ZeroCopy<U> &) { return false; }

template <class T> struct Malloc {
  typedef T value_type;
  Malloc() = default;
  template <class U> constexpr Malloc(const Malloc<U> &) noexcept {}
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
template <class T, class U> bool operator==(const Malloc<T> &, const Malloc<U> &) { return true; }
template <class T, class U> bool operator!=(const Malloc<T> &, const Malloc<U> &) { return false; }

template <class T> struct Managed {
  typedef T value_type;
  Managed() = default;
  template <class U> constexpr Managed(const Managed<U> &) noexcept {}
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
template <class T, class U> bool operator==(const Managed<T> &, const Managed<U> &) { return true; }
template <class T, class U> bool operator!=(const Managed<T> &, const Managed<U> &) { return false; }

} // namespace cmm