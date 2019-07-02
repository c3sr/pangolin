/*! A reference-counted cudaStream_t, which can operate in two modes:



*/

#pragma once

#include <cassert>
#include <memory>

#include <fmt/ostream.h>

#include "pangolin/logger.hpp"
#include "pangolin/utilities.hpp"

namespace pangolin {

/*! A reference-counted CUDA stream

  As a reference to an existing cudaStream_t, in which case the RcStream will not manage the lifetime of the stream.
  Or, by internally creating and managing the lifetime of a cudaStream_t. The stream is destroyed when all references to
  it are gone.

  Not thread-safe
 */
class RcStream {
private:
  int dev_;
  cudaStream_t stream_;
  std::shared_ptr<size_t> count_;

public:
  /*! Create an RcStream that owns a stream
   */
  RcStream() : stream_(nullptr) {
    SPDLOG_TRACE(logger::console(), "default ctor");
    count_ = std::make_shared<size_t>(1);
    assert(count_);
    CUDA_RUNTIME(cudaGetDevice(&dev_));
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
  }

  explicit RcStream(int dev) : dev_(dev), stream_(nullptr) {
    SPDLOG_TRACE(logger::console(), "device ctor {}", dev);
    count_ = std::make_shared<size_t>(1);
    assert(count_);
    CUDA_RUNTIME(cudaSetDevice(dev));
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
  }

  /*! Create an RcStream that refers to an existing stream

      The RcStream will not destroy the stream
  */
  explicit RcStream(int dev, cudaStream_t stream) : dev_(dev), stream_(stream) {
    SPDLOG_TRACE(logger::console(), "stream ctor {}", dev);
    count_ = std::make_shared<size_t>(2); // start with a count of 2 so stream_ is never destroyed
    assert(count_);
    CUDA_RUNTIME(cudaSetDevice(dev));
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
  }

  explicit inline operator cudaStream_t() const { return stream_; }

  explicit inline operator bool() const { return bool(stream_); }

  ~RcStream() {
    if (count_) {
      assert(*count_);
      *count_ -= 1;
      if (0 == *count_) {
        CUDA_RUNTIME(cudaStreamDestroy(stream_));
        stream_ = nullptr;
      }
    }
  }

  /*! move constructor
   */
  RcStream(RcStream &&other) : stream_(other.stream_), dev_(other.dev_), count_(std::move(other.count_)) {
    other.stream_ = nullptr;
  }

  /*! Copy ctor

      Increase reference count by 1
   */
  RcStream(const RcStream &other) : dev_(other.dev_), stream_(other.stream_), count_(other.count_) { *count_ += 1; }

  void swap(RcStream &other) {
    std::swap(stream_, other.stream_);
    std::swap(dev_, other.dev_);
    std::swap(count_, other.count_);
  }

  /*! move-assignment moves the elements of other into the container

      other is left in an unspecified but valid state
  */
  RcStream &operator=(RcStream &&other) noexcept {
    SPDLOG_TRACE(logger::console(), "move assignment");

    /* We just swap other and this, which has the following benefits:
       Don't call delete on other (maybe faster)
       Opportunity for data to be reused since it was not deleted
       No exceptions thrown.
    */

    other.swap(*this);
    return *this;
  }

  bool operator==(const RcStream &other) const noexcept {
    assert(count_ == other.count_);
    return dev_ == other.dev_ && stream_ == other.stream_;
  }

  inline int device() const noexcept { return dev_; }
  inline cudaStream_t stream() const noexcept { return stream_; }
  inline void sync() const noexcept { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }
  size_t count() const noexcept { return *count_; }

  friend std::ostream &operator<<(std::ostream &os, const RcStream &r) {
    return os << "stream:" << uintptr_t(r.stream_);
  }
};

} // namespace pangolin