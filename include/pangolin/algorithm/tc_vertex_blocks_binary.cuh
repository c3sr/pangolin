#pragma once

#include <cub/cub.cuh>
#include <nvToolsExt.h>

#include "count.cuh"
#include "pangolin/algorithm/axpy.cuh"
#include "pangolin/algorithm/load_balance.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/vector.hu"

/*! Determine how many tileSize tiles are needed to cover each row of adj

    The caller should zero the value pointed to by counts
 */
template <size_t BLOCK_DIM_X, typename CsrView>
__global__ void __launch_bounds__(BLOCK_DIM_X) tile_rows_kernel(
    typename CsrView::index_type *counts,         //<! [out] the number of tiles each row (size = numRows)
    typename CsrView::index_type *numWorkItems,   //<! [out] the total number of tiles across all rows.  caller should 0
    const size_t tileSize,                        //<! [in] the number of non-zeros in each tile
    const CsrView adj,                            //<! [in] the adjancency matrix whos rows we will tile
    const typename CsrView::index_type rowOffset, //<! [in] the row to start tiling at
    const typename CsrView::index_type numRows    //<! [in] the number of rows to tile
) {

  typedef typename CsrView::index_type Index;
  typedef cub::BlockReduce<Index, BLOCK_DIM_X> BlockReduce;
  typename BlockReduce::TempStorage reduce;
  Index threadWorkItems = 0;

  // one thread per row
  for (size_t i = BLOCK_DIM_X * blockIdx.x + threadIdx.x; i < numRows; i += gridDim.x * BLOCK_DIM_X) {
    const Index row = i + rowOffset;
    const Index rowSize = adj.rowPtr_[row + 1] - adj.rowPtr_[row];
    const Index rowWorkItems = (rowSize + tileSize - 1) / tileSize;
    counts[i] = rowWorkItems;
    threadWorkItems += rowWorkItems;
  }

  // reduction for numWorkItems
  Index aggregate = BlockReduce(reduce).Sum(threadWorkItems);
  if (0 == threadIdx.x) {
    atomicAdd(numWorkItems, aggregate);
  }
}

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
__global__ void __launch_bounds__(BLOCK_DIM_X)
    row_block_kernel(uint64_t *count,        //<! [out] the count will be accumulated into here
                     const CsrView adj,      //<! [in] the CSR adjacency matrix to operate on
                     const OI *workItemRow,  //<! [in] the row associated with this work item
                     const OI *workItemRank, //<! [in] the rank within the row for this work item
                     const WI numWorkItems   //<! [in] the total number of work items
    ) {
  typedef typename CsrView::index_type Index;
  typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;

  // reuse shared memory for src row and block reduction temporary storage
  __shared__ union {
    Index src[BLOCK_DIM_X];
    typename BlockReduce::TempStorage reduce;
  } shared;
  // __shared__ Index srcShared[BLOCK_DIM_X];
  // __shared__ typename BlockReduce::TempStorage temp_storage;

  uint64_t threadCount = 0;

  // one thread-block per work-item
  for (size_t i = blockIdx.x; i < numWorkItems; i += gridDim.x) { // work item id
    OI row = workItemRow[i];
    OI rank = workItemRank[i];

    // each block is responsible for counting triangles from a contiguous set of non-zeros in the row
    // [srcStart ... srcStop)
    const Index rowStart = adj.rowPtr_[row];
    const Index rowStop = adj.rowPtr_[row + 1];
    const Index sliceStart = rowStart + static_cast<Index>(BLOCK_DIM_X) * rank;
    const Index sliceStop = min(sliceStart + static_cast<Index>(BLOCK_DIM_X), rowStop);
    // const Index srcLen = rowStop - rowStart;

    // each thread handles a non-zero
    // the beginning and end of the dst neighbor list
    // the neighbor list will be size 0 if there is not a non-zero for this thread
    const Index *dstNbrBegin = nullptr;
    const Index *dstNbrEnd = nullptr;
    Index dstIdx = sliceStart + threadIdx.x;
    if (dstIdx < sliceStop) {
      Index dst = adj.colInd_[dstIdx];
      const Index dstStart = adj.rowPtr_[dst];
      const Index dstStop = adj.rowPtr_[dst + 1];
      dstNbrBegin = &adj.colInd_[dstStart];
      dstNbrEnd = &adj.colInd_[dstStop];
    }

    // binary search each BLOCK_DIM_X-sized slice of the src row
    for (Index srcChunkStart = rowStart; srcChunkStart < rowStop; srcChunkStart += BLOCK_DIM_X) {
      Index srcChunkStop = min(srcChunkStart + static_cast<Index>(BLOCK_DIM_X), rowStop);
      Index srcChunkSize = srcChunkStop - srcChunkStart;

      // collaboratively load piece of src row into shared memory
      if (srcChunkStart + threadIdx.x < rowStop) {
        shared.src[threadIdx.x] = adj.colInd_[srcChunkStart + threadIdx.x];
      }
      __syncthreads();

      // search for each element of each thread's dst row into the shared memory
      for (const Index *dstNbr = dstNbrBegin; dstNbr < dstNbrEnd; ++dstNbr) {
        Index nbr = *dstNbr;
        // shared array should be sorted, so skip the search if we know it can't be in that chunk
        if (nbr >= shared.src[0] && nbr <= shared.src[srcChunkSize - 1]) {
          threadCount += pangolin::serial_sorted_count_binary(shared.src, 0, srcChunkSize, nbr);
        }
      }

      __syncthreads();
    }
  }

  // reduce counts within thread-block
  uint64_t aggregate = BlockReduce(shared.reduce).Sum(threadCount);
  if (0 == threadIdx.x) {
    atomicAdd(count, aggregate);
  }
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
  void count_async(const CsrView &adj,     //<! [in] a CSR adjacency matrix to count
                   const size_t rowOffset, //<! [in] the first row to count
                   const size_t numRows    //!< [in] the number of rows to count
  ) {

    CUDA_RUNTIME(cudaSetDevice(dev_));
    const size_t dimBlock = 512;
    typedef typename CsrView::index_type Index;

    LOG(debug, "zero_async final count");
    zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting

    // compute the number (counts) of dimBlock-sized chunks that make up each row [rowOffset, rowOffset + numRows)
    // each workItem is dimBlock elements from a row
    // FIXME: on device
    nvtxRangePush("enumerate work items");
    Vector<Index> counts(numRows);
    Vector<Index> numWorkItems(1, 0);

    LOG(debug, "device = {} tile_rows_kernel<<<{}, {}, {}, {}>>>", dev_, 512, 512, 0, uintptr_t(stream_));
    tile_rows_kernel<512>
        <<<512, 512, 0, stream_>>>(counts.data(), numWorkItems.data(), dimBlock, adj, rowOffset, numRows);
    CUDA_RUNTIME(cudaDeviceSynchronize());

    // for (Index i = 0; i < numRows; ++i) {
    //   Index row = i + rowOffset;
    //   const Index rowSize = adj.rowPtr_[row + 1] - adj.rowPtr_[row];
    //   const Index rowWorkItems = (rowSize + dimBlock - 1) / dimBlock;
    //   counts[i] = rowWorkItems;
    //   numWorkItems[0] += rowWorkItems;
    // }

    LOG(debug, "{} work items", numWorkItems[0]);
    nvtxRangePop();

    // do the initial load-balancing search across rows
    nvtxRangePush("device_load_balance");
    Vector<Index> indices(numWorkItems[0]);
    Index *ranks = nullptr;
    size_t ranksBytes = sizeof(Index) * numWorkItems[0];
    LOG(debug, "allocate {}B for ranks", ranksBytes);
    CUDA_RUNTIME(cudaMalloc(&ranks, sizeof(Index) * numWorkItems[0]));
    // FIXME: static_cast
    device_load_balance(indices.data(), ranks, numWorkItems[0], counts.data(), static_cast<Index>(numRows), stream_);
    nvtxRangePop();

    // indices says which row is associated with each work item, so offset all entries by rowOffset
    // indices[i] += rowOffset
    device_axpy_async(indices.data(), static_cast<Index>(1), static_cast<Index>(rowOffset), indices.size(), stream_);

    // each slice is handled by one thread block
    const int dimGrid = std::min(numWorkItems[0], static_cast<typeof(numWorkItems[0])>(maxGridSize_.x));
    LOG(debug, "counting rows [{}, {}), adj has {} rows", rowOffset, rowOffset + numRows, adj.num_rows());
    assert(rowOffset + numRows <= adj.num_rows());
    assert(count_);
    CUDA_RUNTIME(cudaSetDevice(dev_));
    // const size_t shmemBytes = rowCacheSize_ * sizeof(Index);
    const size_t shmemBytes = 0;
    LOG(debug, "device = {} row_block_kernel<<<{}, {}, {}, {}>>>", dev_, dimGrid, dimBlock, shmemBytes,
        uintptr_t(stream_));
    row_block_kernel<dimBlock>
        <<<dimGrid, dimBlock, shmemBytes, stream_>>>(count_, adj, indices.data(), ranks, numWorkItems[0]);
    CUDA_RUNTIME(cudaGetLastError());

    CUDA_RUNTIME(cudaFree(ranks));
  }

  /*! Synchronous triangle count

      Counts triangles for rows [rowOffset, rowOffset + numRows)
  */
  template <typename CsrView>
  uint64_t count_sync(const CsrView &adj,     //<! [in] a CSR adjacency matrix to count
                      const size_t rowOffset, //<! [in] the first row to count
                      const size_t numRows    //<! [in] the number of rows to count
  ) {
    count_async(adj, rowOffset, numRows);
    sync();
    return count();
  }

  /*! Synchronous triangle count
   */
  template <typename CsrView> uint64_t count_sync(const CsrView &adj) { return count_sync(adj, 0, adj.num_rows()); }

  /*! make the triangle count available in count()
   */
  void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

  uint64_t count() const { return *count_; }
  int device() const { return dev_; }
};

} // namespace pangolin