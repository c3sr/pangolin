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
  __PANGOLIN_HOST_DEVICE__ ArrayView(T *data, size_t size) : data_(data), size_(size) {}
  __PANGOLIN_HOST_DEVICE__ ArrayView() : ArrayView(nullptr, 0) {}

  __PANGOLIN_HOST_DEVICE__ __forceinline__ T &operator[](size_t i) noexcept { return data_[i]; }
  __PANGOLIN_HOST_DEVICE__ __forceinline__ const T &operator[](size_t i) const noexcept { return data_[i]; }

  __PANGOLIN_HOST_DEVICE__ __forceinline__ T *begin() const noexcept { return data_; }
  __PANGOLIN_HOST_DEVICE__ __forceinline__ T *end() const noexcept { return data_ + size_; }

  __PANGOLIN_HOST_DEVICE__ __forceinline__ size_t size() const noexcept { return size_; }
};

#undef __PANGOLIN_HOST_DEVICE__