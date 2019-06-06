#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.hu"
#include "search.cuh"

template <size_t BLOCK_DIM_X, size_t C, typename CsrCooView>
__launch_bounds__(BLOCK_DIM_X) __global__ void __launch_bounds__(BLOCK_DIM_X)
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

public:
  BinaryTC(int dev) : dev_(dev), count_(nullptr) {
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
  }

  BinaryTC() : BinaryTC(0) {}

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
    kernel<const_dimBlock, const_c><<<dimGrid, const_dimBlock, 0, stream_>>>(count_, mat, numEdges, edgeOffset);       \
  }

#define ELSE_IF_CASE(const_dimBlock, const_c)                                                                          \
  else if (dimBlock == const_dimBlock && c == const_c) {                                                               \
    kernel<const_dimBlock, const_c><<<dimGrid, const_dimBlock, 0, stream_>>>(count_, mat, numEdges, edgeOffset);       \
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

    CUDA_RUNTIME(cudaGetLastError());
  }

  template <typename CsrCoo> uint64_t count_sync(const CsrCoo &mat, const size_t edgeOffset, const size_t n) {
    count_async(mat, edgeOffset, n);
    sync();
    return count();
  }

  void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

  uint64_t count() const { return *count_; }
  int device() const { return dev_; }
};

} // namespace pangolin
