#pragma once

#include <cub/cub.cuh>
#include <nvToolsExt.h>

#include "axpy.cuh"
#include "count.cuh"
#include "load_balance.cuh"
#include "pangolin/dense/buffer.cuh"
#include "pangolin/dense/vector.cuh"
#include "zero.cuh"

/*! Determine how many tileSize tiles are needed to cover each row of adj

    The caller should zero the value pointed to by counts
 */
template <size_t BLOCK_DIM_X, typename CsrView>
__global__ void __launch_bounds__(BLOCK_DIM_X) vbcbb_tile_rows_kernel(
    typename CsrView::index_type *__restrict__ counts, //!< [out] the number of tiles each row (size = numRows)
    typename CsrView::index_type
        *__restrict__ numWorkItems,               //!< [out] the total number of tiles across all rows.  caller should 0
    const size_t tileSize,                        //!< [in] the number of non-zeros in each tile
    const CsrView adj,                            //!< [in] the adjancency matrix whos rows we will tile
    const typename CsrView::index_type rowOffset, //!< [in] the row to start tiling at
    const typename CsrView::index_type numRows    //!< [in] the number of rows to tile
) {

  typedef typename CsrView::index_type Index;
  typedef cub::BlockReduce<Index, BLOCK_DIM_X> BlockReduce;
  __shared__ typename BlockReduce::TempStorage reduce;
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
    vbcbb_row_block_kernel(uint64_t *count,        //!< [out] the count will be accumulated into here
                           const CsrView adj,      //!< [in] the CSR adjacency matrix to operate on
                           const OI *workItemRow,  //!< [in] the row associated with this work item
                           const OI *workItemRank, //!< [in] the rank within the row for this work item
                           const WI numWorkItems   //!< [in] the total number of work items
    ) {
  typedef typename CsrView::index_type Index;
  typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;

  // reuse shared memory for src row and block reduction temporary storage
  __shared__ union {
    Index srcdst[BLOCK_DIM_X + BLOCK_DIM_X]; // cache for src & dst
    typename BlockReduce::TempStorage reduce;
  } shared;

  Index *sharedSrc = shared.srcdst;
  Index *sharedDst = &shared.srcdst[BLOCK_DIM_X];

  uint64_t threadCount = 0;

  // one thread-block per work-item
  for (size_t i = blockIdx.x; i < numWorkItems; i += gridDim.x) { // work item id
    OI row = workItemRow[i];
    OI rank = workItemRank[i];

    // each block is responsible for counting triangles for a contiguous slice of non-zeros in the row
    // [sliceStart ... sliceStop)
    const Index rowStart = adj.rowPtr_[row];
    const Index rowStop = adj.rowPtr_[row + 1];
    const Index sliceStart = rowStart + static_cast<Index>(BLOCK_DIM_X) * rank;
    const Index sliceStop = min(sliceStart + static_cast<Index>(BLOCK_DIM_X), rowStop);
    const Index sliceSize = sliceStop - sliceStart;

    // each thread loads a non-zero (dst) from the slice
    // each dst is repeatedly accessed if the src row is longer than shared memory
    Index dstIdx = sliceStart + threadIdx.x;
    if (dstIdx < sliceStop) {
      Index dst = adj.colInd_[dstIdx];
      sharedDst[threadIdx.x] = dst;
    }
    __syncthreads();

    // search into each BLOCK_DIM_X-sized slice of the src row
    for (Index srcChunkStart = rowStart; srcChunkStart < rowStop; srcChunkStart += BLOCK_DIM_X) {
      Index srcChunkStop = min(srcChunkStart + static_cast<Index>(BLOCK_DIM_X), rowStop);
      Index srcChunkSize = srcChunkStop - srcChunkStart;

      // collaboratively load piece of src row into shared memory
      if (srcChunkStart + threadIdx.x < rowStop) {
        sharedSrc[threadIdx.x] = adj.colInd_[srcChunkStart + threadIdx.x];
      }
      __syncthreads();

      // collaboratively work on each dst this block is responsible for
      // broadcast each thread's dst to the whole block
      for (Index dstIdx = 0; dstIdx < sliceSize; ++dstIdx) {
        Index collabDst = sharedDst[dstIdx];

        const Index dstStart = adj.rowPtr_[collabDst];
        const Index dstStop = adj.rowPtr_[collabDst + 1];
        const Index *dstNbrBegin = &adj.colInd_[dstStart];
        const Index *dstNbrEnd = &adj.colInd_[dstStop];

        // dstStop - dstStart > srcChunkSize
        if (false) { // dst longer, we still get to reuse shmem for srcs for each dst
          threadCount += pangolin::block_sorted_count_binary<1, BLOCK_DIM_X>(sharedSrc, srcChunkSize, dstNbrBegin,
                                                                             dstNbrEnd - dstNbrBegin);
        } else { // src longer, we are searching into shmem
          threadCount += pangolin::block_sorted_count_binary<1, BLOCK_DIM_X>(dstNbrBegin, dstNbrEnd - dstNbrBegin,
                                                                             sharedSrc, srcChunkSize);
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

  [Vertex] Vertex-oriented
  [Blocks] tile each vertex with multiple blocks
  [Cache] cache the src neighbor list in shared memory
  [Block] each block collaboratively works on an edge
  [Binary] using parallel binary search

  The search is done from the shorter neighbor list into the longer one

  The src row may be longer than shared memory can hold.
  if so, the src row is divided up into chunks, loaded into shared, and each dst row is compared against each chunk.
  The search goes from the shorter row into the longer row

 */
class VertexBlocksCacheBlockBinary {
private:
  int dev_;             //!< the CUDA device used by this counter
  cudaStream_t stream_; //!< a stream used by this counter
  uint64_t *count_;     //!< the triangle count
  dim3 maxGridSize_;    //!< the maximum grid size allowed by this device
  size_t rowCacheSize_; //!< the size of the kernel's shared memory row cache

public:
  VertexBlocksCacheBlockBinary(int dev, size_t rowCacheSize) : dev_(dev), count_(nullptr), rowCacheSize_(rowCacheSize) {
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
  VertexBlocksCacheBlockBinary() : VertexBlocksCacheBlockBinary(0, 512) {}

  /*! count triangles in adj for rows [rowOffset, rowOffset + numRows).
      May return before count is complete.
   */
  template <typename CsrView>
  void count_async(const CsrView &adj,     //!< [in] a CSR adjacency matrix to count
                   const size_t rowOffset, //!< [in] the first row to count
                   const size_t numRows    //!< [in] the number of rows to count
  ) {

    CUDA_RUNTIME(cudaSetDevice(dev_));
    const size_t dimBlock = 64;
    typedef typename CsrView::index_type Index;

    LOG(debug, "zero_async final count");
    zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting

    // compute the number (counts) of dimBlock-sized chunks that make up each row [rowOffset, rowOffset + numRows)
    // each workItem is dimBlock elements from a row
    nvtxRangePush("enumerate work items");
    Buffer<Index> counts(numRows);    // scratch
    Vector<Index> numWorkItems(1, 0); // scratch

    constexpr size_t trkBlockDim = 512;
    size_t trkGridDim = (numRows + trkBlockDim - 1) / trkBlockDim;
    LOG(debug, "vbcbb_tile_rows_kernel<<<{}, {}, {}, {}>>> device = {} ", trkGridDim, trkBlockDim, 0,
        uintptr_t(stream_), dev_);
    vbcbb_tile_rows_kernel<trkBlockDim><<<trkGridDim, trkBlockDim, 0, stream_>>>(counts.data(), numWorkItems.data(),
                                                                                 dimBlock, adj, rowOffset, numRows);
    CUDA_RUNTIME(cudaDeviceSynchronize());
    const Index hostNumWorkItems = numWorkItems[0];
    nvtxRangePop();

    // do the initial load-balancing search across rows

    nvtxRangePush("device_load_balance");
    Buffer<Index> indices(hostNumWorkItems); // scratch
    Buffer<Index> ranks(hostNumWorkItems);   // scratch
    // FIXME: static_cast
    device_load_balance(indices.data(), ranks.data(), hostNumWorkItems, counts.data(), static_cast<Index>(numRows),
                        stream_);
    nvtxRangePop();

    // indices says which row is associated with each work item, so offset all entries by rowOffset
    // indices[i] += rowOffset
    device_axpy_async(indices.data(), static_cast<Index>(1), static_cast<Index>(rowOffset), indices.size(), stream_);

    // each slice is handled by one thread block
    const int dimGrid = std::min(hostNumWorkItems, static_cast<typeof(hostNumWorkItems)>(maxGridSize_.x));
    LOG(debug, "counting rows [{}, {}), adj has {} rows", rowOffset, rowOffset + numRows, adj.num_rows());
    assert(rowOffset + numRows <= adj.num_rows());
    assert(count_);
    CUDA_RUNTIME(cudaSetDevice(dev_));
    // const size_t shmemBytes = rowCacheSize_ * sizeof(Index);
    const size_t shmemBytes = 0;
    LOG(debug, "device = {} vbcbb_row_block_kernel<<<{}, {}, {}, {}>>>", dev_, dimGrid, dimBlock, shmemBytes,
        uintptr_t(stream_));
    vbcbb_row_block_kernel<dimBlock>
        <<<dimGrid, dimBlock, shmemBytes, stream_>>>(count_, adj, indices.data(), ranks.data(), hostNumWorkItems);
    CUDA_RUNTIME(cudaGetLastError());
  }

  /*! Synchronous triangle count

      Counts triangles for rows [rowOffset, rowOffset + numRows)
  */
  template <typename CsrView>
  uint64_t count_sync(const CsrView &adj,     //!< [in] a CSR adjacency matrix to count
                      const size_t rowOffset, //!< [in] the first row to count
                      const size_t numRows    //!< [in] the number of rows to count
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