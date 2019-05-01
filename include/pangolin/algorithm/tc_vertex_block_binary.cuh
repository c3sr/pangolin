#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.hu"

/*!
Each threadblock handles a row of the adjacency matrix
If the row fits into shared memory, it is loaded.
Each threadblock handles a chunk of the src row and puts it in shared memory.
Each thread handles a column index and does a binary search of the dst row into the src row
*/
template <size_t BLOCK_DIM_X, typename CsrView>
__global__ void row_block_kernel(uint64_t *count,        //!<[out] the count will be accumulated into here
                                 const CsrView adj,      //<! [in] the CSR adjacency matrix to operate on
                                 const size_t rowOffset, //<! [in] the row that this kernel should start at
                                 const size_t numRows    //<! [in] the number of rows this kernel should operate on
) {
  typedef typename CsrView::index_type Index;
  __shared__ Index srcShared[BLOCK_DIM_X];

  uint64_t threadCount = 0;
  for (Index src = rowOffset + blockIdx.x; src < numRows; src += gridDim.x) {

    const size_t srcStart = adj.rowPtr_[src];
    const size_t srcStop = adj.rowPtr_[src + 1];
    const size_t srcLen = srcStop - srcStart;
    const bool useShared = (srcLen) < BLOCK_DIM_X;

    const Index *srcBegin = nullptr;
    if (useShared) {
      for (size_t i = 0; i < srcLen; i += blockDim.x) {
        srcShared[i] = adj.colInd_[srcStart + i];
      }
      __syncthreads();
      srcBegin = srcShared;
    } else {
      srcBegin = &adj.colInd_[srcStart];
    }

    // each thread looks at one destination row and does a binary search into the source row
    for (size_t i = threadIdx.x; i < srcLen; i += blockDim.x) {
      Index dst = srcBegin[i]; // FIXME: already loaded once
      const size_t dstStart = adj.rowPtr_[dst];
      const size_t dstStop = adj.rowPtr_[dst + 1];
      const Index *dstBegin = &adj.colInd_[dstStart];
      const Index *dstEnd = &adj.colInd_[dstStop];
      for (const Index *dstPtr = dstBegin; dstPtr < dstEnd; ++dstPtr) {
        threadCount += pangolin::serial_sorted_count_binary(srcBegin, 0, srcLen, *dstPtr);
      }
    }
  }

  // FIXME: block reduction first
  atomicAdd(count, threadCount);
}

template <size_t BLOCK_DIM_X, typename CsrView>
__global__ void launcher(uint64_t *count,   //!< [inout] the count, caller should zero
                         const CsrView csr, //!< [in] a CSR view
                         const size_t numRows, const size_t rowStart) {

  typedef typename CsrView::index_type Index;

  size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;

  // launch one kernel per row
  for (Index row = gx + rowStart; row < rowStart + numRows; row += BLOCK_DIM_X * gridDim.x) {
    const Index rowStart = csr.rowPtr_[row];
    const Index *rowBegin = &csr.colInd_[rowStart];
    const size_t rowLen = csr.rowPtr_[row + 1] - rowStart;

    constexpr int dimBlock = 512;
    const int dimGrid = (rowLen + dimBlock - 1) / dimBlock;
    row_block_kernel<dimBlock><<<dimGrid, dimBlock>>>(count, row, rowBegin, rowLen, csr);
  }
}

namespace pangolin {

class VertexBlockBinaryTC {
private:
  int dev_;
  cudaStream_t stream_;
  uint64_t *count_;

public:
  VertexBlockBinaryTC(int dev) : dev_(dev), count_(nullptr) {
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
  }

  VertexBlockBinaryTC() : VertexBlockBinaryTC(0) {}

  /*! count triangles in mat. May return before count is complete
   */
  template <typename CsrView>
  void count_async(const CsrView &mat, const size_t numRows, //!< [in] the number of rows this count will handle
                   const size_t rowOffset = 0) {
    zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting

    // block per row
    constexpr int dimBlock = 512;
    const int dimGrid = (numRows + dimBlock - 1) / dimBlock;
    assert(rowOffset + numRows <= mat.num_rows());
    assert(count_);
    LOG(debug, "row_block_kernel: device = {}, blocks = {}, threads = {}", dev_, dimGrid, dimBlock);
    CUDA_RUNTIME(cudaSetDevice(dev_));
    row_block_kernel<dimBlock><<<dimGrid, dimBlock, 0, stream_>>>(count_, mat, rowOffset, numRows);
    CUDA_RUNTIME(cudaGetLastError());
  }

  template <typename CsrView>
  uint64_t count_sync(const CsrView &mat, const size_t numRows, const size_t rowOffset = 0) {
    count_async(mat, numRows, rowOffset);
    sync();
    return count();
  }

  template <typename CsrView> uint64_t count_sync(const CsrView &mat) { return count_sync(mat, mat.num_rows(), 0); }

  /*! make the triangle count available in count()
   */
  void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

  uint64_t count() const { return *count_; }
  int device() const { return dev_; }
};

} // namespace pangolin