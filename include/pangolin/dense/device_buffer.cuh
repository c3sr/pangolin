#pragma once

#include <cstdlib>

#include "pangolin/macro.h"

namespace pangolin {

/*! An uninitialized memory space for Ts on the device
 */
template <typename T> class DeviceBuffer {
public:
  typedef T value_type;
  typedef value_type &reference;
  typedef const value_type &const_reference;
  typedef int device_id_type;

private:
  device_id_type device_; //!< the cuda device this buffer is on
  size_t capacity_;       //!< the number of elements the underlying allocation can hold
  value_type *data_;      //!< the underlying allocation
  PANGOLIN_HOST void set_device();

public:
  PANGOLIN_HOST DeviceBuffer();
  PANGOLIN_HOST DeviceBuffer(size_t n, int device);
  PANGOLIN_HOST DeviceBuffer(DeviceBuffer &&other);
  PANGOLIN_HOST DeviceBuffer(const DeviceBuffer &other);
  PANGOLIN_HOST ~DeviceBuffer();

  PANGOLIN_HOST DeviceBuffer &operator=(DeviceBuffer &&other) noexcept;
  PANGOLIN_HOST DeviceBuffer &operator=(const DeviceBuffer &other);

  PANGOLIN_HOST void swap(DeviceBuffer &other) noexcept;

  PANGOLIN_HOST_DEVICE inline value_type *data() noexcept { return data_; }
  PANGOLIN_HOST_DEVICE inline const value_type *data() const noexcept { return data_; }

  PANGOLIN_HOST void resize(size_t n);

  PANGOLIN_HOST_DEVICE inline size_t size() const noexcept;

  PANGOLIN_HOST_DEVICE inline reference operator[](size_t n);
  PANGOLIN_HOST_DEVICE inline const_reference operator[](size_t n) const;

  PANGOLIN_HOST_DEVICE inline value_type *begin() noexcept { return data_; }
  PANGOLIN_HOST_DEVICE inline value_type *end() noexcept { return data_ + capacity_; }
  PANGOLIN_HOST_DEVICE inline const value_type *begin() const noexcept { return data_; }
  PANGOLIN_HOST_DEVICE inline const value_type *end() const noexcept { return data_ + capacity_; }
};

template <typename T> class DeviceBufferView {
public:
  typedef T value_type;
  typedef value_type &reference;
  typedef const value_type &const_reference;
  typedef int device_id_type;

private:
  device_id_type device_; //!< the device of the underlying allocation
  size_t capacity_;       //!< the number of elements the underlying allocation can hold
  value_type *data_;      //!< the underlying allocation

public:
  /*! Construct from a DeviceBuffer
   */
  PANGOLIN_HOST explicit DeviceBufferView(const DeviceBuffer<T> &buffer)
      : device_(buffer.device_), capacity_(buffer.capacity_), data_(buffer.data_) {}

  PANGOLIN_HOST_DEVICE inline value_type *data() noexcept { return data_; }
  PANGOLIN_HOST_DEVICE inline const value_type *data() const noexcept { return data_; }

  PANGOLIN_HOST_DEVICE inline size_t size() const noexcept { return capacity_; }

  PANGOLIN_HOST_DEVICE inline reference operator[](size_t n) { return data_[n]; }
  PANGOLIN_HOST_DEVICE inline const_reference operator[](size_t n) const { return data_[n]; }

  PANGOLIN_HOST_DEVICE inline value_type *begin() noexcept { return data_; }
  PANGOLIN_HOST_DEVICE inline value_type *end() noexcept { return data_ + capacity_; }
  PANGOLIN_HOST_DEVICE inline const value_type *begin() const noexcept { return data_; }
  PANGOLIN_HOST_DEVICE inline const value_type *end() const noexcept { return data_ + capacity_; }
};

} // namespace pangolin

#include "device_buffer-impl.cuh"
