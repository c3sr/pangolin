#pragma once

#include <cassert>
#include <cstring>
#include <iostream>
#include <new>

#include "device_buffer.cuh"

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

template <typename T> PANGOLIN_HOST void DeviceBuffer<T>::set_device(void) { CUDA_RUNTIME(cudaSetDevice(device_)); }

/*! empty container constructor (default constructor)

  device 0
 */
template <typename T> PANGOLIN_HOST DeviceBuffer<T>::DeviceBuffer(void) : device_(0), capacity_(0), data_(nullptr) {}

/*! fill constructor

Constructs a container with n elements
*/
template <typename T>
PANGOLIN_HOST DeviceBuffer<T>::DeviceBuffer(size_t n, const device_id_type device //<! [in] the CUDA device ID
                                            )
    : device_(device), capacity_(0), data_(nullptr) {
  resize(n);
}

/*! copy constructor

Constructs a container with a copy of each of the elements in other, in the same order
*/
template <typename T>
PANGOLIN_HOST DeviceBuffer<T>::DeviceBuffer(const DeviceBuffer &other)
    : device_(other.device_), capacity_(0), data_(nullptr) {
  SPDLOG_TRACE(logger::console(), "copy ctor");
  resize(other.capacity_);
  CUDA_RUNTIME(cudaMemcpy(data_, other.data_, other.capacity_ * sizeof(value_type), cudaMemcpyDefault));
}

/*! move constructor

    Constructs a container that acquires teh elemetns of x
    x is left in an unspecified by valid state
*/
template <typename T>
PANGOLIN_HOST DeviceBuffer<T>::DeviceBuffer(DeviceBuffer &&other) : capacity_(other.capacity_), data_(other.data_) {
  SPDLOG_TRACE(logger::console(), "move ctor");
  other.capacity_ = 0;
  other.data_ = nullptr;
  other.device_ = 0;
}

// destructor
template <typename T> PANGOLIN_HOST DeviceBuffer<T>::~DeviceBuffer() {
  SPDLOG_TRACE(logger::console(), "dtor");
  if (data_) {
    set_device();
    CUDA_RUNTIME(cudaFree(data_));
    data_ = nullptr;
    capacity_ = 0;
  }
}

/*! copy-assignment copies all elements from other into the container
 */
template <typename T> PANGOLIN_HOST DeviceBuffer<T> &DeviceBuffer<T>::operator=(const DeviceBuffer &other) {
  DeviceBuffer<T> temp(other);
  temp.swap(*this);
  return *this;
}

/*! move-assignment moves the elements of other into the container

    other is left in an unspecified but valid state
*/
template <typename T> PANGOLIN_HOST DeviceBuffer<T> &DeviceBuffer<T>::operator=(DeviceBuffer &&other) noexcept {
  SPDLOG_TRACE(logger::console(), "move assignment");

  /* We just swap other and this, which has the following benefits:
     Don't call delete on other (maybe faster)
     Opportunity for data to be reused since it was not deleted
     No exceptions thrown.
  */

  other.swap(*this);
  return *this;
}

/*! swap the state of the container with other
 */
template <typename T> PANGOLIN_HOST void DeviceBuffer<T>::swap(DeviceBuffer &other) noexcept {
  std::swap(other.capacity_, capacity_);
  std::swap(other.data_, data_);
  std::swap(other.device_, device_);
}

template <typename T> PANGOLIN_HOST void DeviceBuffer<T>::resize(size_t n) {
  if (n != capacity_) {
    set_device();
    CUDA_RUNTIME(cudaFree(data_));
    data_ = nullptr;
    if (n > 0) {
      set_device();
      CUDA_RUNTIME(cudaMalloc(&data_, n * sizeof(value_type)));
      capacity_ = n;
    }
  }
}

template <typename T> PANGOLIN_HOST_DEVICE inline size_t DeviceBuffer<T>::size() const noexcept { return capacity_; }

template <typename T>
PANGOLIN_HOST_DEVICE inline typename DeviceBuffer<T>::reference DeviceBuffer<T>::operator[](size_t n) {
  return data_[n];
}

template <typename T>
PANGOLIN_HOST_DEVICE inline typename DeviceBuffer<T>::const_reference DeviceBuffer<T>::operator[](size_t n) const {
  return data_[n];
}

} // namespace pangolin

#undef PANGOLIN_HOST_DEVICE
#undef PANGOLIN_HOST