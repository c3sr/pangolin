#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/dense/vector.hu"
#include "search.cuh"

template <size_t BLOCK_DIM_X, typename CsrCoo>
__global__ void kernel(uint64_t *count, //!< [inout] the count, caller should zero
                       const CsrCoo mat, const size_t numEdges, const size_t edgeStart) {

  typedef typename CsrCoo::index_type Index;

  size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
  uint64_t threadCount = 0;

  for (size_t i = gx + edgeStart; i < edgeStart + numEdges; i += BLOCK_DIM_X * gridDim.x) {
    const Index src = mat.device_row_ind()[i];
    const Index dst = mat.device_col_ind()[i];

    const Index *srcBegin = &mat.device_col_ind()[mat.device_row_ptr()[src]];
    const Index *srcEnd = &mat.device_col_ind()[mat.device_row_ptr()[src + 1]];
    const Index *dstBegin = &mat.device_col_ind()[mat.device_row_ptr()[dst]];
    const Index *dstEnd = &mat.device_col_ind()[mat.device_row_ptr()[dst + 1]];

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

public:
  LinearTC(int dev) : dev_(dev), count_(nullptr) {
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    *count_ = 0;
  }

  LinearTC() : LinearTC(0) {}

  template <typename CsrCoo> void count_async(const CsrCoo &mat, const size_t numEdges, const size_t edgeOffset = 0) {
    *count_ = 0;
    constexpr int dimBlock = 512;
    const int dimGrid = (numEdges + dimBlock - 1) / dimBlock;
    assert(edgeOffset + numEdges <= mat.nnz());
    assert(count_);
    SPDLOG_DEBUG(logger::console, "device = {}, blocks = {}, threads = {}", dev_, dimGrid, dimBlock);
    CUDA_RUNTIME(cudaSetDevice(dev_));
    kernel<dimBlock><<<dimGrid, dimBlock, 0, stream_>>>(count_, mat, numEdges, edgeOffset);
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