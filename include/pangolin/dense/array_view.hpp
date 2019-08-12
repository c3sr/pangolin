#pragma once

#ifdef __CUDACC__
#define __PANGOLIN_HOST_DEVICE__ __host__ __device__
#else
#define __PANGOLIN_HOST_DEVICE__
#endif

template <typename T> class ArrayView {
  T *data_;
  size_t size_;

public:
  typedef T *pointer;
  typedef const T *const_pointer;
  typedef T *iterator;
  typedef const T *const_iterator;
  typedef T &reference;
  typedef const T &const_reference;

  __PANGOLIN_HOST_DEVICE__ ArrayView(T *data, size_t count) : data_(data), size_(count) {}
  __PANGOLIN_HOST_DEVICE__ ArrayView() : ArrayView(nullptr, 0) {}

  __PANGOLIN_HOST_DEVICE__ __forceinline__ reference operator[](size_t i) noexcept { return data_[i]; }
  __PANGOLIN_HOST_DEVICE__ __forceinline__ const_reference &operator[](size_t i) const noexcept { return data_[i]; }

  __PANGOLIN_HOST_DEVICE__ __forceinline__ iterator begin() noexcept { return data_; }
  __PANGOLIN_HOST_DEVICE__ __forceinline__ iterator end() noexcept { return data_ + size_; }

  __PANGOLIN_HOST_DEVICE__ __forceinline__ const_iterator begin() const noexcept { return data_; }
  __PANGOLIN_HOST_DEVICE__ __forceinline__ const_iterator end() const noexcept { return data_ + size_; }

  __PANGOLIN_HOST_DEVICE__ __forceinline__ size_t size() const noexcept { return size_; }

  __PANGOLIN_HOST_DEVICE__ __forceinline__ pointer data() const noexcept { return data_; }
};

#undef __PANGOLIN_HOST_DEVICE__