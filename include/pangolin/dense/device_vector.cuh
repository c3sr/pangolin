#pragma once

#include "cmm/cmm.hpp"

#include "pangolin/utilities.hpp"
#include "vector.cuh"

#ifdef __CUDACC__
#define PANGOLIN_HOST_DEVICE __host__ __device__
#define PANGOLIN_HOST __host__
#else
#define PANGOLIN_HOST_DEVICE
#define PANGOLIN_HOST
#endif

namespace pangolin {

template <typename T> class DeviceVector : public Vector<T, cmm::Malloc<T>> {
private:
  typedef Vector<T, cmm::Malloc<T>> Parent;
  using Parent::capacity_;
  using Parent::data_;
  using Parent::reserve;
  using Parent::size_;
  using Parent::value_type;

protected:
  cudaStream_t stream_;

public:
  // FIXME: HostVector's value_type must be the same
  template <typename HostVector>
  PANGOLIN_HOST DeviceVector(const HostVector &other, cudaStream_t stream = 0) : stream_(stream) {
    resize(other.size());
    CUDA_RUNTIME(cudaMemcpyAsync(data_, other.data(), size_ * sizeof(T), cudaMemcpyHostToDevice, stream_));
  }

  // FIXME: HostVector's value_type must be the same
  template <typename HostVector> PANGOLIN_HOST DeviceVector &operator=(const HostVector &other) {
    resize(other.size());
    CUDA_RUNTIME(cudaMemcpyAsync(data_, other.data(), size_ * sizeof(value_type), cudaMemcpyHostToDevice));
    return this;
  }

  /*! Conversion to std::vector<T>
   */
  explicit operator std::vector<T>() const {
    std::vector<T> ret;
    ret.resize(size_);
    CUDA_RUNTIME(cudaMemcpyAsync(ret.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    return ret;
  }

  PANGOLIN_HOST void resize(size_t n) {
    if (n < size_) {
      // FIXME: call destructor of some elements
      size_ = n;
    } else if (n > size_) {
      reserve(n);
      // FIXME: value_initialize new elements
      size_ = n;
    }
  }

  PANGOLIN_HOST void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }
};

#undef PANGOLIN_HOST_DEVICE
#undef PANGOLIN_HOST

} // namespace pangolin