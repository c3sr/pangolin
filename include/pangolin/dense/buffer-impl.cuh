#pragma once

#include <cassert>
#include <cstring>
#include <iostream>
#include <new>

#include "buffer.cuh"

#include "pangolin/logger.hpp"
#include "pangolin/utilities.hpp"

#ifdef __CUDACC__
#define PANGOLIN_HOST_DEVICE __host__ __device__
#define PANGOLIN_HOST __host__
#else
#define PANGOLIN_HOST_DEVICE
#define PANGOLIN_HOST
#endif

namespace pangolin {

/*! empty container constructor (default constructor)
 */
template <typename T> PANGOLIN_HOST Buffer<T>::Buffer(void) : capacity_(0), data_(nullptr) {}

/*! fill constructor

Constructs a container with n elements
*/
template <typename T> PANGOLIN_HOST Buffer<T>::Buffer(size_t n) : capacity_(0), data_(nullptr) { resize(n); }

/*! copy constructor

Constructs a container with a copy of each of the elements in other, in the same order
*/
template <typename T> PANGOLIN_HOST Buffer<T>::Buffer(const Buffer &other) : capacity_(0), data_(nullptr) {
  SPDLOG_TRACE(logger::console(), "copy ctor");
  resize(other.capacity_);
  std::memcpy(data_, other.data_, other.capacity_ * sizeof(value_type));
}

/*! move constructor

    Constructs a container that acquires teh elemetns of x
    x is left in an unspecified by valid state
*/
template <typename T> PANGOLIN_HOST Buffer<T>::Buffer(Buffer &&other) : capacity_(other.capacity_), data_(other.data_) {
  SPDLOG_TRACE(logger::console(), "move ctor");
  other.capacity_ = 0;
  other.data_ = nullptr;
}

// destructor
template <typename T> PANGOLIN_HOST Buffer<T>::~Buffer() {
  SPDLOG_TRACE(logger::console(), "dtor");
  if (data_) {
    CUDA_RUNTIME(cudaFree(data_));
    data_ = nullptr;
    capacity_ = 0;
  }
}

/*! copy-assignment copies all elements from other into the container
 */
template <typename T> PANGOLIN_HOST Buffer<T> &Buffer<T>::operator=(const Buffer &other) {
  Buffer<T> temp(other);
  temp.swap(*this);
  return *this;
}

/*! move-assignment moves the elements of other into the container

    other is left in an unspecified but valid state
*/
template <typename T> PANGOLIN_HOST Buffer<T> &Buffer<T>::operator=(Buffer &&other) noexcept {
  SPDLOG_TRACE(logger::console(), "move assignment");

  /* We just swap other and this, which has the following benefits:
     Don't call delete on other (maybe faster)
     Opportunity for data to be reused since it was not deleted
     No exceptions thrown.
  */

  other.swap(*this);
  return *this;
}

/*! Compare for equality
 */
template <typename T> PANGOLIN_HOST bool Buffer<T>::operator==(const Buffer &other) const noexcept {
  if (capacity_ == other.capacity_) {
    for (size_t i = 0; i < capacity_; ++i) {
      if (data_[i] != other.data_[i]) {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

/*! swap the state of the container with other
 */
template <typename T> PANGOLIN_HOST void Buffer<T>::swap(Buffer &other) noexcept {
  std::swap(other.capacity_, capacity_);
  std::swap(other.data_, data_);
}

template <typename T> PANGOLIN_HOST void Buffer<T>::resize(size_t n) {
  if (n != capacity_) {
    CUDA_RUNTIME(cudaFree(data_));
    data_ = nullptr;
    if (n > 0) {
      CUDA_RUNTIME(cudaMallocManaged(&data_, n * sizeof(value_type)));
      capacity_ = n;
    }
  }
}

/*! call cudaMemAdvise(..., cudaMemAdviseSetReadMostly, 0) on Buffer data
 */
template <typename T> PANGOLIN_HOST void Buffer<T>::read_mostly() {
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge37112fc1ac88d0f6bab7a945e48760a
  SPDLOG_TRACE(logger::console(), "cudaMemAdviseSetReadMostly {}B on device", size() * sizeof(T));
  CUDA_RUNTIME(cudaMemAdvise(data_, size() * sizeof(T), cudaMemAdviseSetReadMostly, 0 /* ignored */));
}

/*! call cudaMemAdvise(..., cudaMemAdviseSetAccessedBy, dev) on Buffer data
 */
template <typename T> PANGOLIN_HOST void Buffer<T>::accessed_by(const int dev) {
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge37112fc1ac88d0f6bab7a945e48760a
  CUDA_RUNTIME(cudaMemAdvise(data_, size() * sizeof(T), cudaMemAdviseSetAccessedBy, dev));
}

/*! call cudaMemPrefetchAsync(..., dev, stream) on Buffer data
 */
template <typename T> PANGOLIN_HOST void Buffer<T>::prefetch_async(const int dev, cudaStream_t stream) {
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge8dc9199943d421bc8bc7f473df12e42
  SPDLOG_TRACE(logger::console(), "cudaMemPrefetchAsync {}B to device {} stream {}", size() * sizeof(T), dev,
               uintptr_t(stream));
  CUDA_RUNTIME(cudaMemPrefetchAsync(data_, size() * sizeof(T), dev, stream));
}

template <typename T> PANGOLIN_HOST_DEVICE inline size_t Buffer<T>::size() const noexcept { return capacity_; }

template <typename T> PANGOLIN_HOST_DEVICE inline typename Buffer<T>::reference Buffer<T>::operator[](size_t n) {
  return data_[n];
}

template <typename T>
PANGOLIN_HOST_DEVICE inline typename Buffer<T>::const_reference Buffer<T>::operator[](size_t n) const {
  return data_[n];
}

} // namespace pangolin

#undef PANGOLIN_HOST_DEVICE
#undef PANGOLIN_HOST