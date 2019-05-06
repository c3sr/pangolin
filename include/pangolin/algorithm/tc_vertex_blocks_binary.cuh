#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/load_balance.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.hu"

/*!
Each row of the adjacency matrix is covered by multiple thread blocks
Each thread in the block handles an edge

Each row is sliced into BLOCK_DIM_X slices of contiguous non-zeros - each of these slices is a work-item.
A fixed number of thread blocks are mapped across all work items.
Each thread block can look up which row and rank (slice within the row) it is.


\tparam BLOCK_DIM_X the number of threads in a block
\tparam CsrView A CSR adjacency matrix
*/
template <size_t BLOCK_DIM_X, typename CsrView>
__global__ void row_block_kernel(uint64_t *count,            //<! [out] the count will be accumulated into here
                                 const CsrView adj,          //<! [in] the CSR adjacency matrix to operate on
                                 const size_t *workItemRow,  //<! [in] the row associated with this work item
                                 const size_t *workItemRank, //<! [in] the rank within the row for this work item
                                 const size_t numWorkItems   //<! [in] the total number of work items
) {
  typedef typename CsrView::index_type Index;
  // __shared__ Index srcShared[BLOCK_DIM_X];

  uint64_t threadCount = 0;

  // one thread-block per work-item
  for (size_t i = blockIdx.x; i < numWorkItems; i += gridDim.x) { // work item id
    size_t row = workItemRow[i];
    size_t rank = workItemRow[i];

    // each block is responsible for counting triangles from a contiguous set of non-zeros in the row
    // [srcStart ... srcStop)
    const Index rowStart = adj.rowPtr_[row];
    const Index rowStop = adj.rowPtr_[row + 1];
    const Index rowLen = rowStop - rowStart;
    const Index sliceStart = rowStart + BLOCK_DIM_X * rank;
    const Index sliceStop = max(sliceStart + BLOCK_DIM_X, rowStop);

    // one thread per non-zero in the slice
    for (size_t j = sliceStart + threadIdx.x; j < sliceStop; j += BLOCK_DIM_X) {
      // retrieve the nbrs of the edge dst
      Index dst = adj.colInd_[j];
      const Index dstStart = rowPtr[dst];
      const Index dstStop = rowPtr[dst + 1];
      const Index *dstNbrBegin = colInd[dstStart];
      const Index *dstNbrEnd = colInd[dstStop];

      // for each edge, need to search through the whole src row
      for (const Index *dstNbr = dstNbrBegin; dstPtr < dstNbrEnd; ++dstPtr) {
        threadCount += pangolin::serial_sorted_count_binary(&adj.colInd_[rowStart], 0, rowStop, *dstNbr);
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
class VertexBlocksBinaryTC {
private:
  int dev_;             //<! the CUDA device used by this counter
  cudaStream_t stream_; //<! a stream used by this counter
  uint64_t *count_;     //<! the triangle count
  dim3 maxGridSize_;    //<! the maximum grid size allowed by this device
  size_t rowCacheSize_; //<! the size of the kernel's shared memory row cache

public:
  VertexBlocksBinaryTC(int dev, size_t rowCacheSize) : dev_(dev), count_(nullptr), rowCacheSize_(rowCacheSize) {
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
  VertexBlocksBinaryTC() : VertexBlocksBinaryTC(0, 512) {}

  /*! count triangles in mat. May return before count is complete
   */
  template <typename CsrView>
  void count_async(const CsrView &adj,    //<! [in] a CSR adjacency matrix to count
                   const size_t numRows,  //!< [in] the number of rows this count will handle
                   const size_t rowOffset //<! [in] the first row to count
  ) {

    typedef typename CsrView::index_type Index;

    zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting

    // do the initial load-balancing search across rows
    const Index numWorkItems = adj.nnz();
    const Index numObjects = adj.num_nodes();

    std::vector<Index> rowLengths(numObjects);
    // FIXME: on GPU
    for (size_t i = 0; i < numObjects; ++i) {
      rowLengths[i] = adj.rowPtr_[i + 1] - adj.rowPtr_[i];
    }

    Index *indices = nullptr;
    const size_t indicesBytes = sizeof(Index) * numWorkItems;
    LOG(debug, "allocate {}B for indices", indicesBytes);
    CUDA_RUNTIME(cudaMalloc(&indices, sizeof(Index) * numWorkItems));
    device_load_balance(indices, numWorkItems, rowLengths, numRows, stream_);

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

    CUDA_RUNTIME(cudaFree(indices));
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