#pragma once

#include <cassert>
#include <memory>

#include <fmt/ostream.h>

#include "pangolin/logger.hpp"
#include "pangolin/utilities.hpp"

namespace pangolin {

/*!
 */
class Event {
private:
  cudaEvent_t event_;
  int dev_;

public:
  Event() : event_(nullptr) {
    SPDLOG_TRACE(logger::console(), "default ctor");
    count_ = std::make_shared<size_t>(1);
    assert(count_);
    CUDA_RUNTIME(cudaGetDevice(&dev));
    CUDA_RUNTIME(cudaEventCreate(&event_));
  }

  explicit inline operator cudaEvent_t() const { return event_; }

  ~Event() {
    if (event_) {
      CUDA_RUNTIME(cudaEventDestroy(event_));
    }
  }

  /*! move constructor
   */
  Event(Event &&other) : event_(other.event_), dev_(other.dev_) { other.event_ = nullptr; }

  /*! Copy ctor

      Increase reference count by 1
   */
  Event(const Event &other) : event_(other.event_), dev_(other.dev_) {}

  void swap(Event &other) {
    std::swap(event_, other.event_);
    std::swap(dev_, other.dev_);
  }

  /*! move-assignment moves the elements of other into the container

      other is left in an unspecified but valid state
  */
  Event &operator=(Event &&other) noexcept {
    SPDLOG_TRACE(logger::console(), "move assignment");

    /* We just swap other and this, which has the following benefits:
       Don't call delete on other (maybe faster)
       Opportunity for data to be reused since it was not deleted
       No exceptions thrown.
    */

    other.swap(*this);
    return *this;
  }

  bool operator==(const RcStream &other) const noexcept { return event_ == other.event_; }

  inline int device() const noexcept { return dev_; }
  inline cudaEvent_t event() const noexcept { return event_; }

  friend std::ostream &operator<<(std::ostream &os, const RcStream &r) {
    return os << "event:" << uintptr_t(r.stream_);
  }
};

} // namespace pangolin