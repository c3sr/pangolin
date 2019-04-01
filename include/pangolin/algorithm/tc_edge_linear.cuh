#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.hu"
#include "search.cuh"

template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void kernel(uint64_t *count, //!< [inout] the count, caller should zero
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
  cudaStream_t stream_;
  uint64_t *count_;
  bool destroyStream_;

public:
  LinearTC(int dev) : dev_(dev), count_(nullptr), destroyStream_(true) {
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
    // CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    // zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting
    // CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    CUDA_RUNTIME(cudaHostAlloc(&count_, sizeof(*count_), cudaHostAllocMapped));
    *count_ = 0;
  }

  LinearTC(int dev, cudaStream_t stream) : dev_(dev), stream_(stream), count_(nullptr), destroyStream_(false) {
    CUDA_RUNTIME(cudaSetDevice(dev_));
    // CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    // zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting
    // CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    // error may be deferred to a cudaHostGetDevicePointer
    CUDA_RUNTIME(cudaHostAlloc(&count_, sizeof(*count_), cudaHostAllocMapped));

    *count_ = 0;
  }

  LinearTC() : LinearTC(0) {}
  ~LinearTC() {
    if (destroyStream_ && stream_) {
      CUDA_RUNTIME(cudaStreamDestroy(stream_));
    }
    CUDA_RUNTIME(cudaFreeHost(count_));
  }

  template <typename CsrCoo> void count_async(const CsrCoo &mat, const size_t numEdges, const size_t edgeOffset = 0) {
    // zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting
    uint64_t *devCount = nullptr;
    assert(count_);
    CUDA_RUNTIME(cudaHostGetDevicePointer(&devCount, count_, 0));
    SPDLOG_DEBUG(logger::console, "zero {}", uintptr_t(count_));
    *count_ = 0;
    SPDLOG_DEBUG(logger::console, "did zero");
    constexpr int dimBlock = 512;
    const int dimGrid = (numEdges + dimBlock - 1) / dimBlock;
    assert(edgeOffset + numEdges <= mat.nnz());
    CUDA_RUNTIME(cudaSetDevice(dev_));
    SPDLOG_DEBUG(logger::console, "device = {}, blocks = {}, threads = {}", dev_, dimGrid, dimBlock);
    kernel<dimBlock><<<dimGrid, dimBlock, 0, stream_>>>(devCount, mat, numEdges, edgeOffset);
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