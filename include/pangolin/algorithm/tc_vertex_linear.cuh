#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.hu"

template <size_t BLOCK_DIM_X, typename Index>
__global__ void row_kernel(uint64_t *count,      //!<[out] the count will be accumulated into here
                           const Index src,      //!< [in] the src node
                           const Index *rowBegin //!< [in] the beginning of nonzeros in CSR row src
                           const size_t rowLen,  //!< [in] the number of nonzeros in row src
                           const CsrView mat,    //!< [in] the CSR matrix
) {

  __shared__ Index rowShared[BLOCK_DIM_X];

  // some threads will not have a non-zero in the row
  const size_t rowSharedLen =
      BLOCK_DIM_X * (blockIdx.x + 1) < rowLen ? BLOCK_DIM_X : (rowLen - BLOCK_DIM_X * blockIdx.x);

  // each block loads a chunk of the row into shared memory
  const size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
  if (gx < rowLen) {
    rowShared[threadIdx.x] = row[gx];
  }
  __syncthreads();

  // each thread does a linear search of this piece of the src neighbor list with the dst neighbor list
  if (gx < rowSz) {
    Index dst = row[gx];
    const size_t dstStart = mat.rowPtr_[dst];
    const size_t dstStop = mat.rowPtr_[dst + 1];
    const Index *dstBegin = mat.colInd_[dstStart];
    const Index *dstEnd = mat.colInd_[dstStop];
    threadCount = pangolin::serial_sorted_count_linear(rowShared, &rowShared[rowSharedLen], dstBegin, dstEnd);
    atomicAdd(count, threadCount);
  }
}

template <size_t BLOCK_DIM_X, typename CsrView>
__global__ void launcher(uint64_t *count,   //!< [inout] the count, caller should zero
                         const CsrView csr, //!< [in] a CSR view
                         const size_t numRows, const size_t rowStart) {

  typedef typename CsrView::index_type Index;

  size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;

  // launch one kernel per row
  for (size_t row = gx + rowStart; row < rowStart + numRows; row += BLOCK_DIM_X * gridDim.x) {
    const Index rowStart = csr.rowPtr_[row];
    const Index *rowBegin = csr.colInd_[rowStart];
    const size_t rowLen = csr.rowPtr_[row + 1] - rowStart;

    constexpr int dimBlock = 512;
    const int dimGrid = (rowLen + dimBlock - 1) / dimBlock;
    row_kernel<dimBlock><<<dimGrid, dimBlock>>>(count, row, rowBegin, rowLen, mat);
  }
}

namespace pangolin {

class VertexLinearTC {
private:
  int dev_;
  cudaStream_t stream_;
  uint64_t *count_;

public:
  VertexLinearTC(int dev) : dev_(dev), count_(nullptr) {
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
  }

  VertexLinearTC() : VertexLinearTC(0) {}

  /*! count triangles in mat. May return before count is complete
   */
  template <typename CsrView>
  void count_async(const CsrView &mat, const size_t numRows, //!< [in] the number of rows this count will handle
                   const size_t rowOffset = 0) {
    zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting

    // one thread per row
    constexpr int dimBlock = 512;
    const int dimGrid = (numRows + dimBlock - 1) / dimBlock;
    assert(rowOffset + numRows <= mat.num_rows());
    assert(count_);
    SPDLOG_DEBUG(logger::console, "device = {}, blocks = {}, threads = {}", dev_, dimGrid, dimBlock);
    CUDA_RUNTIME(cudaSetDevice(dev_));
    launcher<dimBlock><<<dimGrid, dimBlock, 0, stream_>>>(count_, mat, numRows, rowOffset);
    CUDA_RUNTIME(cudaGetLastError());
  }

  template <typename CsrView> uint64_t count_sync(const CsrView &mat, const size_t rowOffset, const size_t n) {
    count_async(mat, rowOffset, n);
    sync();
    return count();
  }

  /*! make the triangle count available in count()
   */
  void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

  uint64_t count() const { return *count_; }
  int device() const { return dev_; }
};

} // namespace pangolin