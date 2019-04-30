

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
  size_t stop_;

public:
  __device__ WarpIter(const size_t start, const size_t stop)
      : index_(start + blockDim.x * blockIdx.x + warp_idx()), stop_(stop) {}

  __device__ int lane_idx() const { return threadIdx.x % 32; }
  __device__ int warp_idx() const { return threadIdx.x / 32; }
  __device__ size_t stride() const { return gridDim.x * (blockDim.x + 32 - 1) / 32; }

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
    index_ += stride();
    return *this;
  }

  __device__ WarpIdx operator*() const { return index_; }

  __device__ WarpIdx *operator->() { return &index_; }

  /*! equality
   */
  __device__ bool operator==(const WarpIter &rhs) const {
    if (index_ >= stop_ && rhs.index_ >= rhs.stop_) {
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

public:
  __device__ WarpRange(const size_t start, const size_t stop) : start_(start), stop_(stop) {}

  WarpIter __device__ begin() const { return WarpIter(start_, stop_); }
  WarpIter __device__ end() const { return WarpIter(stop_, stop_); }
};

/*!

\tparam InputIterator
\tparam OutputIterator
\tparam UnaryOperation Unary __device__ function that accepts one element of the type pointed to by InputIterator as
argument, and returns some result value convertible to the type pointed to by OutputIterator. This can either be a
function pointer or a function object.
*/
template <typename InputIterator, typename OutputIterator, typename UnaryOperation>
__global__ void kernel_mapper(InputIterator first1, InputIterator last1, OutputIterator result, UnaryOperation op) {
  while (first1 != last1) {
    *result = op(*first1);
    ++result;
    ++first1;
  }
  return result;
}

/* similar to C++ stl transform (1)
 */
template <class InputIterator, class OutputIterator, class UnaryOperation>
OutputIterator transform(InputIterator first1, InputIterator last1, OutputIterator result, UnaryOperation op) {}

} // namespace pangolin