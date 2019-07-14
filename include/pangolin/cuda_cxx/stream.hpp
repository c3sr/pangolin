#pragma once

#include <cassert>
#include <memory>

#include <fmt/ostream.h>

#include "pangolin/logger.hpp"
#include "pangolin/utilities.hpp"

namespace pangolin {

/*! A reference to a cudaStream, with the same interface as Stream
 */
class StreamRef {
private:
  cudaStream_t stream_;
  int dev_;

public:
  StreamRef(cudaStream_t stream, int device) : stream_(stream), dev_(device) {}

  /*! move constructor
   */
  StreamRef(StreamRef &&other) : stream_(other.stream_), dev_(other.dev_) { other.stream_ = nullptr; }

  /*! copy ctor
   */
  StreamRef(const StreamRef &other) : stream_(other.stream_), dev_(other.dev_) {}

  void swap(StreamRef &other) {
    std::swap(stream_, other.stream_);
    std::swap(dev_, other.dev_);
  }

  /*! move-assignment moves the elements of other into the container

      other is left in an unspecified but valid state
  */
  StreamRef &operator=(StreamRef &&other) noexcept {
    other.swap(*this);
    return *this;
  }

  bool operator==(const StreamRef &other) const noexcept { return stream_ == other.stream_; }

  inline int device() const noexcept { return dev_; }
  inline cudaStream_t stream() const noexcept { return stream_; }
  StreamRef ref() const noexcept { return *this; }
  void sync() const noexcept { CUDA_RUNTIME(cudaStreamSynchronize(stream_));}

  explicit inline operator cudaStream_t() const { return stream_; }

  friend std::ostream &operator<<(std::ostream &os, const StreamRef &s) {
    return os << "stream:" << uintptr_t(s.stream_);
  }
};
class Stream {
private:
  cudaStream_t stream_;
  int dev_;

public:
  Stream() : stream_(nullptr) {
    SPDLOG_TRACE(logger::console(), "default ctor");
    CUDA_RUNTIME(cudaGetDevice(&dev_));
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
  }
  explicit Stream(int dev) : stream_(nullptr), dev_(dev) {
    SPDLOG_TRACE(logger::console(), "device ctor");
    CUDA_RUNTIME(cudaSetDevice(dev));
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
  }

  ~Stream() {
    if (stream_) {
      CUDA_RUNTIME(cudaStreamDestroy(stream_));
    }
  }

  /*! copy ctor

      Only one of each stream exists
   */
  Stream(const Stream &other) = delete;

  /*! move constructor
   */
  Stream(Stream &&other) : stream_(other.stream_), dev_(other.dev_) { other.stream_ = nullptr; }

  void swap(Stream &other) {
    std::swap(stream_, other.stream_);
    std::swap(dev_, other.dev_);
  }

  /*! move-assignment moves the elements of other into the container

      other is left in an unspecified but valid state
  */
  Stream &operator=(Stream &&other) noexcept {
    SPDLOG_TRACE(logger::console(), "move assignment");
    other.swap(*this);
    return *this;
  }

  int device() const noexcept { return dev_; }
  inline cudaStream_t stream() const noexcept { return stream_; }
  StreamRef ref() const noexcept { return StreamRef(stream_, dev_); }
  void sync() const noexcept { CUDA_RUNTIME(cudaStreamSynchronize(stream_));}

  bool operator==(const Stream &other) const noexcept { return stream_ == other.stream_; }

  friend std::ostream &operator<<(std::ostream &os, const Stream &s) { return os << "stream:" << uintptr_t(s.stream_); }

  explicit inline operator cudaStream_t() const noexcept { return stream_; }
};

} // namespace pangolin