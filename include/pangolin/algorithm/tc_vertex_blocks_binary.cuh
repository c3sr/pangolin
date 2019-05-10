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
\tparam OI object index type
\tparam WI work-item index type
\tparam CsrView A CSR adjacency matrix
*/
template <size_t BLOCK_DIM_X, typename OI, typename WI, typename CsrView>
__global__ void row_block_kernel(uint64_t *count,        //<! [out] the count will be accumulated into here
                                 const CsrView adj,      //<! [in] the CSR adjacency matrix to operate on
                                 const OI *workItemRow,  //<! [in] the row associated with this work item
                                 const OI *workItemRank, //<! [in] the rank within the row for this work item
                                 const WI numWorkItems   //<! [in] the total number of work items
) {
  typedef typename CsrView::index_type Index;
  extern __shared__ Index srcShared[];
  // __shared__ Index srcShared[BLOCK_DIM_X];

  uint64_t threadCount = 0;

  // one thread-block per work-item
  for (size_t i = blockIdx.x; i < numWorkItems; i += gridDim.x) { // work item id
    OI row = workItemRow[i];
    OI rank = workItemRank[i];

    // if (threadIdx.x == 0) {
    //   printf("block %d row %d rank %d\n", blockIdx.x, row, rank);
    // }

    // each block is responsible for counting triangles from a contiguous set of non-zeros in the row
    // [srcStart ... srcStop)
    const Index rowStart = adj.rowPtr_[row];
    const Index rowStop = adj.rowPtr_[row + 1];
    const Index sliceStart = rowStart + static_cast<Index>(BLOCK_DIM_X) * rank;
    const Index sliceStop = min(sliceStart + static_cast<Index>(BLOCK_DIM_X), rowStop);
    // if (threadIdx.x == 0) {
    //   if (sizeof(Index) == 4) {
    //     printf("row %d rank %d: dsts from colInd[%d, %d)\n", row, rank, sliceStart, sliceStop);
    //   } else {
    //     printf("row %lu rank %lu: dsts from colInd[%lu, %lu)\n", row, rank, sliceStart, sliceStop);
    //   }
    // }

    // one thread per non-zero in the slice
    for (size_t j = sliceStart + threadIdx.x; j < sliceStop; j += BLOCK_DIM_X) {
      // retrieve the nbrs of the edge dst
      Index dst = adj.colInd_[j];
      const Index dstStart = adj.rowPtr_[dst];
      const Index dstStop = adj.rowPtr_[dst + 1];
      const Index *dstNbrBegin = &adj.colInd_[dstStart];
      const Index *dstNbrEnd = &adj.colInd_[dstStop];
      // printf("%d->%d  [%d %d) -> [%d %d)\n", row, dst, rowStart, rowStop, dstStart, dstStop);

      // for each edge, need to search through the whole src row
      for (const Index *dstNbr = dstNbrBegin; dstNbr < dstNbrEnd; ++dstNbr) {
        threadCount += pangolin::serial_sorted_count_binary(adj.colInd_, rowStart, rowStop, *dstNbr);
      }

      // printf("bid %d tid %d row %d rank %d: %d->%d = %lu\n", blockIdx.x, threadIdx.x, row, rank, row, dst,
      // threadCount);
    }

    // if (threadCount != 0) {
    //   printf("bid %d tid %d row %d rank %d: colInd[%d, %d): count %lu\n", blockIdx.x, threadIdx.x, row, rank,
    //          sliceStart, sliceStop, threadCount);
    // }
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

  /*! count triangles in adj for rows [rowOffset, rowOffset + numRows).
      May return before count is complete.
   */
  template <typename CsrView>
  void count_async(const CsrView &adj,    //<! [in] a CSR adjacency matrix to count
                   const size_t numRows,  //!< [in] the number of rows this count will handle
                   const size_t rowOffset //<! [in] the first row to count
  ) {

    const size_t dimBlock = 512;
    typedef typename CsrView::index_type Index;

    LOG(debug, "zero_async final count");
    zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting

    // compute the number (counts) of dimBlock-sized chunks that make up each row [rowOffset, rowOffset + numRows)
    // each workItem is dimBlock elements from a row
    // FIXME: on device
    Vector<Index> counts(numRows);
    Index numWorkItems = 0;
    for (Index i = 0; i < numRows; ++i) {
      Index row = i + rowOffset;
      const Index rowSize = adj.rowPtr_[row + 1] - adj.rowPtr_[row];
      const Index rowWorkItems = (rowSize + dimBlock - 1) / dimBlock;
      counts[i] = rowWorkItems;
      numWorkItems += rowWorkItems;
    }

    // do the initial load-balancing search across rows
    Vector<Index> indices(numWorkItems);
    Index *ranks = nullptr;
    size_t ranksBytes = sizeof(Index) * numWorkItems;
    LOG(debug, "allocate {}B for ranks", ranksBytes);
    CUDA_RUNTIME(cudaMalloc(&ranks, sizeof(Index) * numWorkItems));
    // FIXME: static_cast
    device_load_balance(indices.data(), ranks, numWorkItems, counts.data(), static_cast<Index>(numRows), stream_);

    // indices says which row is associated with each work item, so offset all entries by rowOffset
    // FIXME: on device
    for (auto &e : indices) {
      e += rowOffset;
    }

    // each slice is handled by one thread block
    const int dimGrid = std::min(numWorkItems, static_cast<typeof(numWorkItems)>(maxGridSize_.x));
    LOG(debug, "counting rows [{}, {}), adj has {} rows", rowOffset, rowOffset + numRows, adj.num_rows());
    assert(rowOffset + numRows <= adj.num_rows());
    assert(count_);
    CUDA_RUNTIME(cudaSetDevice(dev_));
    const size_t shmemBytes = rowCacheSize_ * sizeof(Index);
    LOG(debug, "device = {} row_block_kernel<<<{}, {}, {}, {}>>>", dev_, dimGrid, dimBlock, shmemBytes,
        uintptr_t(stream_));
    row_block_kernel<dimBlock>
        <<<dimGrid, dimBlock, shmemBytes, stream_>>>(count_, adj, indices.data(), ranks, numWorkItems);
    CUDA_RUNTIME(cudaGetLastError());

    CUDA_RUNTIME(cudaFree(ranks));
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