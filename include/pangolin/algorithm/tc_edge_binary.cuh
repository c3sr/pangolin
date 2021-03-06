#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.cuh"
#include "search.cuh"

template <size_t BLOCK_DIM_X, size_t C, typename CsrCooView>
__global__ void __launch_bounds__(BLOCK_DIM_X)
    kernel(uint64_t *__restrict__ count,                //!< [inout] the count, caller should zero
           const CsrCooView mat, const size_t numEdges, //!< the number of edges this kernel will count
           const size_t edgeStart                       //!< the edge this kernel will start counting at
    ) {

  typedef typename CsrCooView::index_type Index;

  static_assert(BLOCK_DIM_X % 32 == 0, "block size should be multiple of 32");
  constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;

  const size_t lx = threadIdx.x % 32;
  const size_t gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;
  uint64_t warpCount = 0;

  for (size_t i = gwx + edgeStart; i < edgeStart + numEdges; i += BLOCK_DIM_X * gridDim.x / 32) {
    const Index src = mat.rowInd_[i];
    const Index dst = mat.colInd_[i];

    const Index srcStart = mat.rowPtr_[src];
    const Index srcStop = mat.rowPtr_[src + 1];
    const Index dstStart = mat.rowPtr_[dst];
    const Index dstStop = mat.rowPtr_[dst + 1];
    const Index dstLen = dstStop - dstStart;
    const Index srcLen = srcStop - srcStart;

    // only thread 0 will return the full count
    // search in parallel through the smaller array into the larger array

    // FIXME: remove warp reduction from this function call
    if (dstLen > srcLen) {
      warpCount += pangolin::warp_sorted_count_binary<C, warpsPerBlock>(&mat.colInd_[srcStart], srcLen,
                                                                        &mat.colInd_[dstStart], dstLen);
    } else {
      warpCount += pangolin::warp_sorted_count_binary<C, warpsPerBlock>(&mat.colInd_[dstStart], dstLen,
                                                                        &mat.colInd_[srcStart], srcLen);
    }
  }

  // Add to total count
  if (0 == lx) {
    atomicAdd(count, warpCount);
  }
}

namespace pangolin {

class BinaryTC {
private:
  int dev_;
  cudaStream_t stream_;
  uint64_t *count_;

  // events for measuring time
  cudaEvent_t kernelStart_;
  cudaEvent_t kernelStop_;

public:
  /*! Device constructor

      Create a counter on device dev
  */
  BinaryTC(int dev, cudaStream_t stream = 0) : dev_(dev), stream_(stream), count_(nullptr) {
    SPDLOG_TRACE(logger::console(), "device ctor");
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting
    CUDA_RUNTIME(cudaGetLastError());

    CUDA_RUNTIME(cudaEventCreate(&kernelStart_));
    CUDA_RUNTIME(cudaEventCreate(&kernelStop_));
  }

  /*! default ctor - counter on device 0
   */
  BinaryTC() : BinaryTC(0) { SPDLOG_TRACE(logger::console(), "default ctor"); }

  /*! copy ctor - create a new counter on the same device

  All fields are reset
   */
  BinaryTC(const BinaryTC &other) : BinaryTC(other.dev_, other.stream_) { SPDLOG_TRACE(logger::console(), "copy ctor"); }

  ~BinaryTC() {
    SPDLOG_TRACE(logger::console(), "dtor");
    CUDA_RUNTIME(cudaEventDestroy(kernelStart_));
    CUDA_RUNTIME(cudaEventDestroy(kernelStop_));
  }

  BinaryTC &operator=(BinaryTC &&other) noexcept {
    SPDLOG_TRACE(logger::console(), "move assignment");

    /* We just swap other and this, which has the following benefits:
       Don't call delete on other (maybe faster)
       Opportunity for data to be reused since it was not deleted
       No exceptions thrown.
    */

    other.swap(*this);
    return *this;
  }

  void swap(BinaryTC &other) noexcept {
    std::swap(other.dev_, dev_);
    std::swap(other.kernelStart_, kernelStart_);
    std::swap(other.kernelStop_, kernelStop_);
    std::swap(other.stream_, stream_);
  }

  /* Async count triangle on device. May return before count is complete.

    Call sync() to block until count is complete.
    Call count() to retrieve count

  */
  template <typename CsrCoo>
  void count_async(const CsrCoo &mat, const size_t numEdges, const size_t edgeOffset = 0, const size_t dimBlock = 256,
                   const size_t c = 1) {

    zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting

    // create one warp per edge
    const int dimGrid = (32 * numEdges + (dimBlock * c) - 1) / (dimBlock * c);
    assert(edgeOffset + numEdges <= mat.nnz());
    assert(count_);
    LOG(debug, "device = {}, blocks = {}, threads = {}", dev_, dimGrid, dimBlock);
    CUDA_RUNTIME(cudaSetDevice(dev_));

#define IF_CASE(const_dimBlock, const_c)                                                                               \
  if (dimBlock == const_dimBlock && c == const_c) {                                                                    \
    CUDA_RUNTIME(cudaEventRecord(kernelStart_, stream_));                                                              \
    kernel<const_dimBlock, const_c><<<dimGrid, const_dimBlock, 0, stream_>>>(count_, mat, numEdges, edgeOffset);       \
    CUDA_RUNTIME(cudaEventRecord(kernelStop_, stream_));                                                               \
  }

#define ELSE_IF_CASE(const_dimBlock, const_c)                                                                          \
  else if (dimBlock == const_dimBlock && c == const_c) {                                                               \
    CUDA_RUNTIME(cudaEventRecord(kernelStart_, stream_));                                                              \
    kernel<const_dimBlock, const_c><<<dimGrid, const_dimBlock, 0, stream_>>>(count_, mat, numEdges, edgeOffset);       \
    CUDA_RUNTIME(cudaEventRecord(kernelStop_, stream_));                                                               \
  }

    IF_CASE(32, 1)
    ELSE_IF_CASE(64, 1)
    ELSE_IF_CASE(128, 1)
    ELSE_IF_CASE(256, 1)
    ELSE_IF_CASE(512, 1)
    ELSE_IF_CASE(1024, 1)
    ELSE_IF_CASE(32, 2)
    ELSE_IF_CASE(64, 2)
    ELSE_IF_CASE(128, 2)
    ELSE_IF_CASE(256, 2)
    ELSE_IF_CASE(512, 2)
    ELSE_IF_CASE(1024, 2)
    else {
      LOG(critical, "unsupported coarsening factor {} or block dimension {}", c, dimBlock);
      exit(-1);
    }

#undef IF_CASE
#undef ELSE_IF_CASE
  }

  template <typename CsrCoo> uint64_t count_sync(const CsrCoo &mat, const size_t edgeOffset, const size_t n) {
    count_async(mat, edgeOffset, n);
    sync();
    return count();
  }

  void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

  uint64_t count() const { return *count_; }
  int device() const { return dev_; }



  /*! return the number of ms the GPU spent in the triangle counting kernel

    After this call, the kernel will have been completed, though the count may not be available.
   */
  float kernel_time() {
    float ms;
    CUDA_RUNTIME(cudaEventSynchronize(kernelStop_));
    CUDA_RUNTIME(cudaEventElapsedTime(&ms, kernelStart_, kernelStop_));
    return ms / 1e3;
  }
};

} // namespace pangolin
