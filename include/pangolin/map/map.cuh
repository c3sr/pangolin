#pragma once

namespace pangolin {

class WarpIdx {
private:
  size_t index_;

public:
  __device__ WarpIdx(size_t i) : index_(i) {}
  __device__ size_t lane_idx() const { return threadIdx.x % 32; }
  __device__ size_t warp_idx() const { return threadIdx.x / 32; }
  __device__ size_t idx() const { return index_; }

  __device__ WarpIdx &operator+=(size_t s) {
    index_ += s;
    return *this;
  }

  __device__ bool operator>=(const WarpIdx &wi) const { return index_ >= wi.index_; }
  __device__ bool operator==(const WarpIdx &wi) const { return index_ == wi.index_; }
};

/*!
 */
class WarpIter {
private:
  WarpIdx index_;
  size_t maxIndex_;
  size_t stride_;

public:
  __device__ WarpIter(const size_t i, const size_t maxI, const size_t stride)
      : index_(i), maxIndex_(maxI), stride_(stride) {}

  __device__ int lane_idx() const { return threadIdx.x % 32; }
  __device__ int warp_idx() const { return threadIdx.x / 32; }

  /*! postfix
   */
  __device__ WarpIter operator++(int) {
    WarpIter i(*this);
    ++(*this);
    return i;
  }

  /*! prefix
   */
  __device__ WarpIter &operator++() {
    index_ += stride_;
    return *this;
  }

  __device__ WarpIdx operator*() const { return index_; }

  __device__ WarpIdx *operator->() { return &index_; }

  /*! equality
   */
  __device__ bool operator==(const WarpIter &rhs) const {
    if (index_ >= maxIndex_ && rhs.index_ >= rhs.maxIndex_) {
      return true;
    } else {
      return index_ == rhs.index_;
    }
  }

  /* inequality
   */
  __device__ bool operator!=(const WarpIter &rhs) const { return !(*this == rhs); }
};

/*!
 */
class WarpRange {
private:
  size_t start_;
  size_t stop_;
  size_t warpsPerGrid_;

public:
  __device__ WarpRange(const size_t start, const size_t stop) : start_(start), stop_(stop) {
    size_t warpsPerBlock = (blockDim.x + 32 - 1) / 32;
    warpsPerGrid_ = gridDim.x * warpsPerBlock;
  }

  WarpIter __device__ begin() const { return WarpIter(start_, stop_, warpsPerGrid_); }
  WarpIter __device__ end() const { return WarpIter(stop_, stop_, warpsPerGrid_); }
};

} // namespace pangolin