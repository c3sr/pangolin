#pragma once

#include "pangolin/macro.h"

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

  PANGOLIN_HOST_DEVICE ArrayView(T *data, size_t count) : data_(data), size_(count) {}
  PANGOLIN_HOST_DEVICE ArrayView() : ArrayView(nullptr, 0) {}

  PANGOLIN_HOST_DEVICE __forceinline__ reference operator[](size_t i) noexcept { return data_[i]; }
  PANGOLIN_HOST_DEVICE __forceinline__ const_reference &operator[](size_t i) const noexcept { return data_[i]; }

  PANGOLIN_HOST_DEVICE __forceinline__ iterator begin() noexcept { return data_; }
  PANGOLIN_HOST_DEVICE __forceinline__ iterator end() noexcept { return data_ + size_; }

  PANGOLIN_HOST_DEVICE __forceinline__ const_iterator begin() const noexcept { return data_; }
  PANGOLIN_HOST_DEVICE __forceinline__ const_iterator end() const noexcept { return data_ + size_; }

  PANGOLIN_HOST_DEVICE __forceinline__ size_t size() const noexcept { return size_; }

  PANGOLIN_HOST_DEVICE __forceinline__ pointer data() const noexcept { return data_; }
};

