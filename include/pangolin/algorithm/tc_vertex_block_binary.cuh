#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.cuh"

/*!
Each threadblock handles a row of the adjacency matrix
If the row fits into shared memory, it is loaded.
Each threadblock handles a chunk of the src row and puts it in shared memory.
Each thread handles a column index and does a binary search of the dst row into the src row

\tparam BLOCK_DIM_X the number of threads in a block
\tparam CsrView A CSR adjacency matrix
*/
template <size_t BLOCK_DIM_X, typename CsrView>
__global__ void
row_block_kernel(uint64_t *count,        //<! [out] the count will be accumulated into here
                 const CsrView adj,      //<! [in] the CSR adjacency matrix to operate on
                 const size_t rowOffset, //<! [in] the row that this kernel should start at
                 const size_t numRows,   //<! [in] the number of rows this kernel should operate on
                 const size_t shmemSz    //<! [in] the number of row elements that can be cached in shared memory
) {
  typedef typename CsrView::index_type Index;
  extern __shared__ Index srcShared[];

  uint64_t threadCount = 0;
  for (Index src = rowOffset + blockIdx.x; src < numRows; src += gridDim.x) {

    const size_t srcStart = adj.rowPtr_[src];
    const size_t srcStop = adj.rowPtr_[src + 1];
    const size_t srcLen = srcStop - srcStart;

    // if the row fits in shared memory, put in in there
    // srcBegin will point at the beginning of the row's nonzeros, whether it's in shared memory or not
    const Index *srcBegin = nullptr;
    if (srcLen < shmemSz) {
      for (size_t i = threadIdx.x; i < srcLen; i += blockDim.x) {
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

namespace pangolin {

/*! A triangle counter

   One block per vertex (row in the CSR)
   Short rows cached in shared memory
   One thread per non-zero in the row. Each thread loads another row and compares all non-zeros to the source row with
   binary search.

 */
class VertexBlockBinaryTC {
private:
  int dev_;             //<! the CUDA device used by this counter
  cudaStream_t stream_; //<! a stream used by this counter
  uint64_t *count_;     //<! the triangle count
  dim3 maxGridSize_;    //<! the maximum grid size allowed by this device
  size_t rowCacheSize_; //<! the size of the kernel's shared memory row cache

public:
  VertexBlockBinaryTC(int dev, size_t rowCacheSize) : dev_(dev), count_(nullptr), rowCacheSize_(rowCacheSize) {
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));

    {
      cudaDeviceProp prop;
      CUDA_RUNTIME(cudaGetDeviceProperties(&prop, dev));
      maxGridSize_.x = prop.maxGridSize[0];
    }
  }

  /*! default constructor on GPU0 with a row cache size of 512 elements.
   */
  VertexBlockBinaryTC() : VertexBlockBinaryTC(0, 512) {}

  /*! count triangles in mat. May return before count is complete
   */
  template <typename CsrView>
  void count_async(const CsrView &adj,    //<! [in] a CSR adjacency matrix to count
                   const size_t numRows,  //!< [in] the number of rows this count will handle
                   const size_t rowOffset //<! [in] the first row to count
  ) {

    typedef typename CsrView::index_type Index;

    zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting

    // one block per row
    constexpr int dimBlock = 512;
    const int dimGrid = std::min(numRows, static_cast<uint64_t>(maxGridSize_.x));
    LOG(debug, "counting rows [{}, {}), adj has {} rows", rowOffset, rowOffset + numRows, adj.num_rows());
    assert(rowOffset + numRows <= adj.num_rows());
    assert(count_);
    CUDA_RUNTIME(cudaSetDevice(dev_));
    const size_t shmemBytes = rowCacheSize_ * sizeof(Index);
    LOG(debug, "row_block_kernel: device = {}, blocks = {}, threads = {} shmem = {}", dev_, dimGrid, dimBlock,
        shmemBytes);
    row_block_kernel<dimBlock>
        <<<dimGrid, dimBlock, shmemBytes, stream_>>>(count_, adj, rowOffset, numRows, rowCacheSize_);
    CUDA_RUNTIME(cudaGetLastError());
  }

  /*! Synchronous triangle count

      Counts triangles for rows [rowOffset, rowOffset + numRows)
  */
  template <typename CsrView>
  uint64_t count_sync(const CsrView &adj,    //<! [in] a CSR adjacency matrix to count
                      const size_t numRows,  //<! [in] the number of rows to count
                      const size_t rowOffset //<! [in] the first row to count
  ) {
    count_async(adj, numRows, rowOffset);
    sync();
    return count();
  }

  /*! Synchronous triangle count
   */
  template <typename CsrView> uint64_t count_sync(const CsrView &adj) { return count_sync(adj, adj.num_rows(), 0); }

  /*! make the triangle count available in count()
   */
  void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

  uint64_t count() const { return *count_; }
  int device() const { return dev_; }
};

} // namespace pangolin