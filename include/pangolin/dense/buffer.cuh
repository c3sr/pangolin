#pragma once

#include <cstdlib>

#include "pangolin/macro.h"

namespace pangolin {

/*! An uninitialized memory space for Ts
 */
template <typename T> class Buffer {
public:
  typedef T value_type;
  typedef value_type &reference;
  typedef const value_type &const_reference;

private:
  size_t capacity_;  //!< the number of elements the underlying allocation can hold
  value_type *data_; //!< the underlying allocation

public:
  PANGOLIN_HOST explicit Buffer(void);
  PANGOLIN_HOST explicit Buffer(size_t n);
  PANGOLIN_HOST Buffer(Buffer &&other);
  PANGOLIN_HOST Buffer(const Buffer &other);
  PANGOLIN_HOST ~Buffer();

  PANGOLIN_HOST Buffer &operator=(Buffer &&other) noexcept;
  PANGOLIN_HOST Buffer &operator=(const Buffer &other);

  PANGOLIN_HOST bool operator==(const Buffer &other) const noexcept;

  PANGOLIN_HOST void swap(Buffer &other) noexcept;

  PANGOLIN_HOST_DEVICE inline value_type *data() noexcept { return data_; }
  PANGOLIN_HOST_DEVICE inline const value_type *data() const noexcept { return data_; }

  PANGOLIN_HOST void resize(size_t n);

  PANGOLIN_HOST void read_mostly();
  PANGOLIN_HOST void accessed_by(const int dev);
  PANGOLIN_HOST void prefetch_async(const int dev, cudaStream_t stream = 0);

  PANGOLIN_HOST_DEVICE inline size_t size() const noexcept;

  PANGOLIN_HOST_DEVICE inline reference operator[](size_t n);
  PANGOLIN_HOST_DEVICE inline const_reference operator[](size_t n) const;

  PANGOLIN_HOST_DEVICE inline value_type *begin() noexcept { return data_; }
  PANGOLIN_HOST_DEVICE inline value_type *end() noexcept { return data_ + capacity_; }
  PANGOLIN_HOST_DEVICE inline const value_type *begin() const noexcept { return data_; }
  PANGOLIN_HOST_DEVICE inline const value_type *end() const noexcept { return data_ + capacity_; }
};

template <typename T> class BufferView {
public:
  typedef T value_type;
  typedef value_type &reference;
  typedef const value_type &const_reference;

private:
  size_t capacity_;  //!< the number of elements the underlying allocation can hold
  value_type *data_; //!< the underlying allocation

public:
  /*! Construct from a Buffer
   */
  PANGOLIN_HOST explicit BufferView(const Buffer<T> &buffer) : capacity_(buffer.capacity_), data_(buffer.data_) {}

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


#include "buffer-impl.cuh"
