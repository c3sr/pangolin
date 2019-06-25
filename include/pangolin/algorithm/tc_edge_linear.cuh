#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.cuh"
#include "search.cuh"
#include "pangolin/cuda_cxx/rc_stream.hpp"

template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void __launch_bounds__(BLOCK_DIM_X)
    tc_edge_linear_kernel(uint64_t *count, //!< [inout] the count, caller should zero
           const CsrCooView mat, const size_t numEdges, const size_t edgeStart) {

  typedef typename CsrCooView::index_type Index;

  size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
  uint64_t threadCount = 0;

  for (size_t i = gx + edgeStart; i < edgeStart + numEdges; i += BLOCK_DIM_X * gridDim.x) {
    const Index src = mat.rowInd_[i];
    const Index dst = mat.colInd_[i];

    const Index *srcBegin = &mat.colInd_[mat.rowPtr_[src]];
    const Index *srcEnd = &mat.colInd_[mat.rowPtr_[src + 1]];
    const Index *dstBegin = &mat.colInd_[mat.rowPtr_[dst]];
    const Index *dstEnd = &mat.colInd_[mat.rowPtr_[dst + 1]];

    threadCount += pangolin::serial_sorted_count_linear(srcBegin, srcEnd, dstBegin, dstEnd);
  }

  // Block-wide reduction of threadCount
  typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);

  // Add to total count
  if (0 == threadIdx.x) {
    atomicAdd(count, aggregate);
  }
}

namespace pangolin {

class LinearTC {
private:
  int dev_;
  RcStream stream_;
  uint64_t *count_;

public:
  LinearTC(int dev) : dev_(dev), count_(nullptr) {
    SPDLOG_TRACE(logger::console(), "set dev {}", dev_);
    CUDA_RUNTIME(cudaSetDevice(dev_));
    SPDLOG_TRACE(logger::console(), "mallocManaged");
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    zero_async<1>(count_, dev_, cudaStream_t(stream_)); // zero on the device that will do the counting
    // CUDA_RUNTIME(cudaHostAlloc(&count_, sizeof(*count_), cudaHostAllocPortable | cudaHostAllocMapped));
    // *count_ = 0;
  }

  LinearTC(int dev, RcStream stream) : dev_(dev), stream_(stream), count_(nullptr) {
    if (dev != stream.device()) {
      LOG(critical, "stream device {} and counter device {} mismatch", stream.device(), dev);
    }
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    zero_async<1>(count_, dev_, cudaStream_t(stream_)); // zero on the device that will do the counting
    // error may be deferred to a cudaHostGetDevicePointer
    // CUDA_RUNTIME(cudaHostAlloc(&count_, sizeof(*count_), cudaHostAllocPortable | cudaHostAllocMapped));
    // *count_ = 0;
  }

  /*! move constructor
   */
  LinearTC(LinearTC &&other)
      : dev_(other.dev_), stream_(std::move(other.stream_)), count_(other.count_) {
    other.count_ = nullptr;
  }

  LinearTC() : LinearTC(0) {}
  ~LinearTC() {
    CUDA_RUNTIME(cudaFree(count_));
  }

  template <typename CsrCoo>
  void count_async(const CsrCoo &mat, const size_t edgeOffset, const size_t numEdges, const size_t dimBlock = 256) {
    assert(count_);
    CUDA_RUNTIME(cudaSetDevice(dev_));
    zero_async<1>(count_, dev_, cudaStream_t(stream_));
    CUDA_RUNTIME(cudaGetLastError());
    // zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting
    // uint64_t *devCount = nullptr;
    // CUDA_RUNTIME(cudaHostGetDevicePointer(&devCount, count_, 0));
    // LOG(debug, "zero {}", uintptr_t(count_));
    // *count_ = 0;
    // LOG(debug, "did zero");
    const int dimGrid = (numEdges + dimBlock - 1) / dimBlock;
    assert(edgeOffset + numEdges <= mat.nnz());
    LOG(debug, "tc_edge_linear_kernel device = {}, <<<{}, {}, 0, {}>>>", dev_, dimGrid, dimBlock, stream_);

#define CASE(const_dimBlock)                                                                                           \
  case const_dimBlock:                                                                                                 \
    tc_edge_linear_kernel<const_dimBlock><<<dimGrid, const_dimBlock, 0, stream_.stream()>>>(count_, mat, numEdges, edgeOffset);                \
    break;

    switch (dimBlock) {
      CASE(32)
      CASE(64)
      CASE(128)
      CASE(256)
      CASE(512)
    default:
      LOG(critical, "unsupported block dimension {}", dimBlock);
      exit(1);
    }

#undef CASE
    CUDA_RUNTIME(cudaGetLastError());
  }

  template <typename CsrCoo> uint64_t count_sync(const CsrCoo &adj, const size_t edgeOffset, const size_t n) {
    count_async(adj, edgeOffset, n);
    sync();
    return count();
  }

  template <typename CsrCoo> uint64_t count_sync(const CsrCoo &adj) {
    count_async(adj, 0, adj.nnz());
    sync();
    return count();
  }

  void sync() { stream_.sync(); }

  /*! return the most recent count
   */
  uint64_t count() { 
    return *count_;
  }
  int device() const { return dev_; }
};

} // namespace pangolin