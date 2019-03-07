#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/dense/vector.hu"
#include "search.cuh"

namespace pangolin {

class LinearTC {
private:
  int dev_;
  cudaStream_t stream_;
  uint64_t *count_;

public:
  LinearTC(int dev) : dev_(dev) {
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
  }

  LinearTC() {
    CUDA_RUNTIME(cudaStreamDestroy(stream_));
    CUDA_RUNTIME(cudaFree(count_));
  }

  template <typename CsrCoo> void count_async(const CsrCoo &mat, const size_t edgeOffset, const size_t numEdges) {

    constexpr int dimBlock = 512;
    dim3 dimGrid = (numEdges + dimBlock - 1) / dimBlock;
    // pangolin::kernel_sorted_count_linear<dimBlock><<<dimGrid, dimBlock>>>()
  }

  template <typename CsrCoo> uint64_t count_sync(const CsrCoo &mat, const size_t edgeOffset, const size_t n) {
    count_async(mat, edgeOffset, n);
    sync();
    return count();
  }

  void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

  uint64_t count() const { return *count_; }
};

} // namespace pangolin