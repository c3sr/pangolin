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
__global__ void __launch_bounds__(BLOCK_DIM_X) anjur_iyer_tile_rows_kernel(
    typename CsrView::index_type *counts,         //<! [out] the number of tiles each row (size = numRows)
    typename CsrView::index_type *numWorkItems,   //<! [out] the total number of tiles across all rows.  caller should 0
    const size_t tileSize,                        //<! [in] the number of non-zeros in each tile
    const CsrView adj,                            //<! [in] the adjancency matrix whos rows we will tile
    const typename CsrView::index_type rowOffset, //<! [in] the row to start tiling at
    const typename CsrView::index_type numRows    //<! [in] the number of rows to tile
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
    anjur_iyer_kernel(uint64_t *count,        //<! [out] the count will be accumulated into here
                      const CsrView adj,      //<! [in] the CSR adjacency matrix to operate on
                      const OI *workItemRow,  //<! [in] the row associated with this work item
                      const OI *workItemRank, //<! [in] the rank within the row for this work item
                      const WI numWorkItems   //<! [in] the total number of work items
    ) {
  typedef typename CsrView::index_type Index;
  typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;

  // reuse shared memory for src row and block reduction temporary storage
  __shared__ union {
    Index rowBuffer[BLOCK_DIM_X]; // cache for src
    Index sbBase;                 // the first edge in the portion
    Index srStart;                // starting node for the row
    Index srNext;                 // starting node for the next row
    typename BlockReduce::TempStorage reduce;
  } shared;

  uint64_t threadCount = 0;

  // one thread-block per work-item
  for (size_t i = blockIdx.x; i < numWorkItems; i += gridDim.x) { // work item id

    // identify the portion of the row that we are responsible for
    OI row = workItemRow[i];
    OI rank = workItemRank[i];

    // each block is responsible for counting triangles for a contiguous slice of non-zeros in the row
    // [sliceStart ... sliceStop)
    // same for all threads
    const Index rowStart = adj.rowPtr_[row];
    const Index rowStop = adj.rowPtr_[row + 1];
    const Index sliceStart = rowStart + static_cast<Index>(BLOCK_DIM_X) * rank;
    const Index sliceStop = min(sliceStart + static_cast<Index>(BLOCK_DIM_X), rowStop);

    // each thread has its own destination node and neighbor list
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

    // compare the dst list with each chunk of the src list
    // each thread is responsible for searching a different dst nbr list for common elements with the src list
    const Index *dstNbrPtr = dstNbrBegin;
    for (Index srcChunkStart = rowStart; srcChunkStart < rowStop; srcChunkStart += BLOCK_DIM_X) {

      Index srcChunkStop = min(srcChunkStart + static_cast<Index>(BLOCK_DIM_X), rowStop);
      Index srcChunkSize = srcChunkStop - srcChunkStart;

      if (row == 1 && rank == 0 && threadIdx.x == 1) {
        printf("working on src chunk colInd[%d-%d)\n", srcChunkStart, srcChunkStop);
      }

      if (threadIdx.x < srcChunkSize) {
        // collaboratively load piece of src row into shared memory
        shared.rowBuffer[threadIdx.x] = adj.colInd_[srcChunkStart + threadIdx.x];
      }
      __syncthreads();

      for (Index srcIdx = 0; srcIdx < srcChunkSize; ++srcIdx) {
        const Index src = shared.rowBuffer[srcIdx];

        // continue looking through dst nbr list for the src element
        // FIXME: can this be replaced with pangolin::serial_sorted_search_linear? need to check that the return value
        // is in registers not mem
        for (; dstNbrPtr < dstNbrEnd;) {
          if (*dstNbrPtr < src) {
            ++dstNbrPtr;
          } else if (*dstNbrPtr == src) {
            threadCount += 1;
            ++dstNbrPtr;
            break;
          } else { // >
            break; // went past where search value should be in dst nbr list
          }
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

 */
class AnjurIyer {
private:
  int dev_;             //<! the CUDA device used by this counter
  cudaStream_t stream_; //<! a stream used by this counter
  uint64_t *count_;     //<! the triangle count
  dim3 maxGridSize_;    //<! the maximum grid size allowed by this device
  size_t rowCacheSize_; //<! the size of the kernel's shared memory row cache

public:
  AnjurIyer(int dev, size_t rowCacheSize) : dev_(dev), count_(nullptr), rowCacheSize_(rowCacheSize) {
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
  AnjurIyer() : AnjurIyer(0, 512) {}

  /*! count triangles in adj for rows [rowOffset, rowOffset + numRows).
      May return before count is complete.
   */
  template <typename CsrView>
  void count_async(const CsrView &adj,     //<! [in] a CSR adjacency matrix to count
                   const size_t rowOffset, //<! [in] the first row to count
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
    Vector<Index> counts(numRows);
    Vector<Index> numWorkItems(1, 0);

    LOG(debug, "anjur_iyer_tile_rows_kernel<<<{}, {}, {}, {}>>> device = {} ", 512, 512, 0, uintptr_t(stream_), dev_);
    anjur_iyer_tile_rows_kernel<512>
        <<<512, 512, 0, stream_>>>(counts.data(), numWorkItems.data(), dimBlock, adj, rowOffset, numRows);
    CUDA_RUNTIME(cudaDeviceSynchronize());
    const Index hostNumWorkItems = numWorkItems[0];
    nvtxRangePop();

    // do the initial load-balancing search across rows

    nvtxRangePush("device_load_balance");
    Buffer<Index> indices(hostNumWorkItems);
    Buffer<Index> ranks(hostNumWorkItems);
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
    LOG(debug, "device = {} anjur_iyer_kernel<<<{}, {}, {}, {}>>>", dev_, dimGrid, dimBlock, shmemBytes,
        uintptr_t(stream_));
    anjur_iyer_kernel<dimBlock>
        <<<dimGrid, dimBlock, shmemBytes, stream_>>>(count_, adj, indices.data(), ranks.data(), hostNumWorkItems);
    CUDA_RUNTIME(cudaGetLastError());
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

// original code from ECE 508 project
#if 0
// Vikram Anjur (vanjur2) and Mihir Iyer (mviyer2)
// ECE 508
// Final Project - efficient sparse matrix-matrix approach for triangle counting

#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include "template.hu"

// Max shared memory per block is 49,152 B => 49152 B /(8B/entry * 1024 entries) = 6 elements/thread
#define BLOCK_SIZE 64
#define TILE_SIZE BLOCK_SIZE
#define PRE_BLOCK_SIZE 1024
// #define LOAD_VEC_SIZE 4
// #define NUM_BINS 1024

// Device info 
// Info:: There is 1 device supporting CUDA
// Info:: Device 0 name TITAN V
// Info::  Computational Capabilities: 7.0
// Info::  Maximum global memory size: 12618760192
// Info::  Maximum constant memory size: 65536
// Info::  Maximum shared memory size per block: 49152
// Info::  Maximum block dimensions: 1024x1024x64
// Info::  Maximum grid dimensions: 2147483647x65535x65535
// Info::  Warp size: 32


__global__ static void kernel_tc_opt(uint64_t *__restrict__ tc_dev, //!< global triangle counts
  const uint32_t *const edgeSrc,         //!< node ids for edge srcs
  const uint32_t *const edgeDst,         //!< node ids for edge dsts
  const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
  const size_t numEdges,                  //!< how many edges to count triangles for
  const uint32_t *const rowBlocks,         // Starting block index for each row 
  const uint32_t numRows
) {

  /*** Declare shared memory ***/
  __shared__ uint32_t sbBase; // the first edge in the portion 
  __shared__ uint32_t srStart; // starting node for the row
  __shared__ uint32_t srNext; // starting node for the next row
  __shared__ uint64_t blockCounts[BLOCK_SIZE]; // Each thread can write it's local triangle count
  __shared__ uint32_t rowBuffer[TILE_SIZE]; // Threads collaborate to load in row elements 


  /*** Identify the portion of the row that we are responsible for ***/
  int bid = blockIdx.x; 
  int tx = threadIdx.x;

  // Only thread 0 needs to do this work
  if(tx==0) {
    uint32_t low=0, high = numRows, mid, value, row, bPortion;
    
    while(low < high) {
      mid = (low + high)/2;
      value=rowBlocks[mid];

      if(value <= bid) {
        low = mid + 1;
      }
      else {
        high = mid; 
      }
    }
    row = low; 
    bPortion = bid;
    if(row>0) {
      bPortion = bid - rowBlocks[row-1];
    }

    // Write to shared memory to broadcast values
    srStart = rowPtr[row];
    srNext = rowPtr[row+1];
    sbBase = srStart  + bPortion*BLOCK_SIZE;
  }

  // Perform syncthreads so everyone is caught up
  __syncthreads();

  /*** Each thread figures out which edge it's working with ***/
  // Load useful information from shared memory 
  uint32_t rStart = srStart;
  uint32_t rNext = srNext;

  // Initialize the thread's initial tc to 0
  uint32_t tc = 0;

  // Local vars at block level: rStart, rNext, tc

  // Calculate the edge number 
  uint32_t edgeNum = sbBase + tx;
  uint32_t destNode = 0;
  if (edgeNum < rNext)
    destNode = edgeDst[edgeNum];

  // Maintain information about other row
  uint32_t otherRowStart = rowPtr[destNode];
  uint32_t otherRowNext = rowPtr[destNode+1];
  uint32_t otherRowIdx = 0;
  uint32_t otherDest;

  uint32_t jDest;
  int slice, j;
  
  // uint32_t otherRowBuffer[LOAD_VEC_SIZE];
  // uint32_t orbIdx = LOAD_VEC_SIZE;

  /*** Loop across the row ***/
  for (slice = 0; slice < ceil((float)(rNext-rStart)/TILE_SIZE); slice++) {
    // Load data into shared memory
    //for (int i = 0; i < TILE_SIZE/BLOCK_SIZE; i++) {
      if (rStart + TILE_SIZE*slice/* + i*BLOCK_SIZE*/ + tx < rNext)
        rowBuffer[tx/* + i*BLOCK_SIZE*/] = edgeDst[rStart + TILE_SIZE*slice/* + i*BLOCK_SIZE*/ + tx];
    //}
    __syncthreads();  // Row values for this slice are loaded into shared memory
    
    if (edgeNum < rNext) {
      // Actually iterate through elements and compare for matching columns
      for (j = 0; j < TILE_SIZE; j++) {
        // Make sure we are doing legitimate steps (not running off end of row)
        if (rStart + TILE_SIZE*slice + j >= rNext)
          break;
        // Read destination for current step from our row out of shared mem
        jDest = rowBuffer[j];
        // Check for jDest in other row
        while(otherRowStart+otherRowIdx<otherRowNext) { // Make sure we are not out of bounds for other row
          // Read value
          otherDest = edgeDst[otherRowStart+otherRowIdx];
          /*if (orbIdx == LOAD_VEC_SIZE) {
            reinterpret_cast<uint4 *>(otherRowBuffer)[0] = reinterpret_cast<const uint4 *>(edgeDst)[(otherRowStart+otherRowIdx)/LOAD_VEC_SIZE];
            orbIdx = (otherRowStart+otherRowIdx)%LOAD_VEC_SIZE;
          }

          otherDest = otherRowBuffer[orbIdx];*/
          // There is a match
          if(jDest == otherDest) {
            tc += 1;  // Triangle found, increment tc
            otherRowIdx += 1; // This other row dest node is too small for future jDests
            // orbIdx += 1;
            break;  // We have found the match for this jDest so break
          }

          // No match yet, but still possible
          else if (jDest > otherDest) {
            otherRowIdx += 1; // This other row dest node is too small for current and future jDests
            // orbIdx += 1;
          }

          // No match possible
          else {
            break; // This jDest is hopeless move on to next
          }    
        }
      }
    }
    __syncthreads();
  }
  blockCounts[tx] = tc; // Have each thread write it's triangle count to shared memory

  /*** Do a parallel reduction on block-level triangle counts ***/
  int stride, sumIdx; // local variables 
  for(stride =1; stride < BLOCK_SIZE; stride *= 2) {
    // Make sure all threads in block are synchronized up to this point
    __syncthreads();

    // Increment our local sum 
    sumIdx = (tx+1)*(2*stride) -1; // Make sure that we use contiguous threads to reduce divergence
    if((sumIdx < BLOCK_SIZE) && (sumIdx>=stride)) {
      blockCounts[sumIdx] += blockCounts[sumIdx-stride];
    }
  }

  // Perform a final sync threads so the block's computations are done 
  __syncthreads();

  /*** Each block will write it's output to a single location in a global memory array ***/
  if(tx==0) {
    tc_dev[bid] = blockCounts[BLOCK_SIZE-1];
    // if(blockCounts[BLOCK_SIZE-1] != 0) {
    //   atomicAdd((unsigned long long int *) tc_dev, (unsigned long long int) blockCounts[BLOCK_SIZE-1]);
    // }
  }
}

__global__ static void kernel_tc(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) {

  // Figure out the ID of the current edge 
  int edgeID = blockIdx.x*blockDim.x + threadIdx.x; 

  // Make sure we are in bounds 
  if(edgeID < numEdges) {
    // Determine the source and destination node for this edge 
    uint32_t source = edgeSrc[edgeID];
    uint32_t dest = edgeDst[edgeID];

    // Determine the start and end pointers for the source and destination 
    uint32_t source_ptr = rowPtr[source];
    uint32_t source_end = rowPtr[source+1];
    uint32_t dest_ptr = rowPtr[dest];
    uint32_t dest_end = rowPtr[dest+1];

    // Make local variables for comparison nodes 
    uint32_t sn = edgeDst[source_ptr];
    uint32_t dn = edgeDst[dest_ptr]; 

    // Initialize the count of common neighbors to zero 
    uint64_t count=0; 

    // Actually identify how many neighbors are common between source and dest nodes 
    while ((source_ptr < source_end) && (dest_ptr < dest_end)) {
      // Identify if the source neighbor is smaller
      if(sn < dn) {
        sn = edgeDst[++source_ptr];
      }

      // Or the destination neighbor is smaller 
      else if(sn > dn) {
        dn = edgeDst[++dest_ptr];
      }

      // Otherwise, we have a match
      else {
        // Increment our count value 
        count += 1; 
        // Increment both pointers 
        sn = edgeDst[++source_ptr];
        dn = edgeDst[++dest_ptr];
      }
    }

    // Write our result back
    triangleCounts[edgeID] = count;
  }
}

// Kernel to generate the number of blocks per row in current graph 
__global__ void numBlocks_kernel(const size_t numRows, const uint32_t *const rowPtr, uint32_t *__restrict__ rowBlocks) {
  // Commented out chunk is shared implementation
  /*__shared__ uint32_t srowPtr[PRE_BLOCK_SIZE];
  
  // Figure out what row the current thread is responsible for 
  int row = blockIdx.x * blockDim.x + threadIdx.x; 

  size_t curRowIdx=0;
  
  // Make sure we are in bounds 
  if (row < numRows) {
    curRowIdx = srowPtr[threadIdx.x] = rowPtr[row];
  }
  __syncthreads();
  
  if (row < numRows) {
    size_t nextRowIdx =0;
    // Get starting point of next row
    if (row == numRows-1 || threadIdx.x == blockDim.x-1) {
      nextRowIdx = rowPtr[row+1];
    } else {
      nextRowIdx = srowPtr[threadIdx.x+1];
    }

    // Figure out the number of blocks needed for this row and write result
    rowBlocks[row] = ceil((float)(nextRowIdx-curRowIdx) / BLOCK_SIZE);
  }*/
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  if (row<numRows) {
    rowBlocks[row] = ceil((float)(rowPtr[row+1]-rowPtr[row])/BLOCK_SIZE);
  }
}

uint64_t count_triangles(const pangolin::CSRCOOView<uint32_t> view, const int mode) {
  if (mode == 1) {

    // Create the pangolin vector to hold the per-edge triangle counts 
    pangolin::Vector<uint64_t> counts = pangolin::Vector<uint64_t>::Vector(view.nnz(), 0);

    // Get a pointer to this vector for the kernel to use 
    uint64_t* counts_ptr = counts.data();

    // Specify the grid and block dimensions 
    dim3 dimBlock(512);
    dim3 dimGrid(ceil((float)view.nnz()/dimBlock.x));
    
    // Launch the GPU kernel for linear search 
    kernel_tc<<<dimGrid, dimBlock>>>(counts_ptr, view.row_ind(), view.col_ind(), view.row_ptr(), view.nnz());

    // Make sure the kernel is complete 
    cudaDeviceSynchronize();

    // Perform global reduction on GPU using a reduction kernel with Thrust
    uint64_t total = thrust::reduce(thrust::device, counts_ptr, counts_ptr + view.nnz());
    
    // Return the calculated triangle count
    return total;

  } else if (mode == 2) {

    /****** Generate an array with the number of blocks per row ********/ 

    // First, allocate an array of size numRows on the GPU
    size_t numRows = view.num_rows();
    uint32_t *rowBlocks;
    cudaMalloc((void **)&rowBlocks, numRows*sizeof(uint32_t));

    /******* Identify the starting block IDs for each row *********/
    // Call a kernel to generate the number of blocks per row 
    dim3 dimBlock1(PRE_BLOCK_SIZE);
    dim3 dimGrid1(ceil((float)numRows/dimBlock1.x));
    numBlocks_kernel<<< dimGrid1, dimBlock1 >>> (numRows, view.row_ptr(), rowBlocks);

    cudaDeviceSynchronize(); // wait for numBlocks kernel to finish
    
    // Call a kernel to perform an inclusive prefix sum on the number of blocks per row 
    thrust::inclusive_scan(thrust::device, rowBlocks, rowBlocks + numRows, rowBlocks);

    // Copy the results of the prefix sum back to cpu 
    uint64_t numBlocks = 0;
    cudaMemcpy(&numBlocks, &rowBlocks[numRows-1], sizeof(uint32_t), cudaMemcpyDeviceToHost);

    /***** Start the triangle counting kernel *****/
    dim3 dimBlock2(BLOCK_SIZE);
    dim3 dimGrid2(numBlocks);
    
    uint64_t *tc_dev;
    cudaMalloc((void **)&tc_dev, numBlocks*sizeof(uint64_t));
    
    // cudaFuncSetCacheConfig(kernel_tc_opt, cudaFuncCachePreferShared);

    kernel_tc_opt<<<dimGrid2, dimBlock2>>>(tc_dev, view.row_ind(), view.col_ind(), view.row_ptr(), view.nnz(), rowBlocks, numRows);

    cudaDeviceSynchronize(); // Wait for triangle counting kernel to finish

    //uint64_t triangle_count;
    //cudaMemcpy(&triangle_count, tc_dev, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    /*** Perform parallel reduction using thrust library ***/
    uint64_t tc = thrust::reduce(thrust::device, tc_dev, tc_dev + numBlocks);
        
    // Free device memory 
    cudaFree(rowBlocks);
    cudaFree(tc_dev);

    //return triangle_count;
    return tc;
  } else {
    assert("Unexpected mode");
    return uint64_t(-1);
  }
}
#endif