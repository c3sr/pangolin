#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.hu"
#include "search.cuh"

template <size_t BLOCK_DIM_X, size_t C, typename CsrCoo>
__global__ void kernel(uint64_t *count,                         //!< [inout] the count, caller should zero
                       const CsrCoo mat, const size_t numEdges, //!< the number of edges this kernel will count
                       const size_t edgeStart                   //<! the edge this kernel will start counting at
) {

  typedef typename CsrCoo::index_type Index;

  static_assert(BLOCK_DIM_X % 32 == 0); // block is multiple of 32
  constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;

  const size_t lx = threadIdx.x % 32;
  const size_t gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;
  uint64_t warpCount = 0;

  for (size_t i = gwx + edgeStart; i < edgeStart + numEdges; i += BLOCK_DIM_X * gridDim.x / 32) {
    const Index src = mat.device_row_ind()[i];
    const Index dst = mat.device_col_ind()[i];

    const Index srcStart = mat.device_row_ptr()[src];
    const Index srcStop = mat.device_row_ptr()[src + 1];
    const Index dstStart = mat.device_row_ptr()[dst];
    const Index dstStop = mat.device_row_ptr()[dst + 1];

    // only thread 0 will return the full count
    warpCount += pangolin::warp_sorted_count_binary<C, warpsPerBlock>(
        &mat.device_col_ind()[srcStart], srcStop - srcStart, &mat.device_col_ind()[dstStart], dstStop - dstStart);
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

  template <typename CsrCoo>
  void count_async(const CsrCoo &mat, const size_t numEdges, const size_t edgeOffset = 0, const size_t c = 1) {
    zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting
    // create one warp per edge
    constexpr int dimBlock = 512;
    const int dimGrid = (32 * numEdges + dimBlock - 1) / (dimBlock * c);
    assert(edgeOffset + numEdges <= mat.nnz());
    assert(count_);
    SPDLOG_DEBUG(logger::console, "device = {}, blocks = {}, threads = {}", dev_, dimGrid, dimBlock);
    CUDA_RUNTIME(cudaSetDevice(dev_));
    switch (c) {
    case 1:
      kernel<dimBlock, 1><<<dimGrid, dimBlock, 0, stream_>>>(count_, mat, numEdges, edgeOffset);
      break;
    case 2:
      kernel<dimBlock, 2><<<dimGrid, dimBlock, 0, stream_>>>(count_, mat, numEdges, edgeOffset);
      break;
    case 4:
      kernel<dimBlock, 4><<<dimGrid, dimBlock, 0, stream_>>>(count_, mat, numEdges, edgeOffset);
      break;
    case 8:
      kernel<dimBlock, 8><<<dimGrid, dimBlock, 0, stream_>>>(count_, mat, numEdges, edgeOffset);
      break;
    case 16:
      kernel<dimBlock, 16><<<dimGrid, dimBlock, 0, stream_>>>(count_, mat, numEdges, edgeOffset);
      break;
    default:
      LOG(critical, "unsupported coarsening factor, try 1,2,4,8,16");
      exit(-1);
    }
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