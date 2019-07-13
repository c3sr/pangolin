/*! A dynamic-selection algorithm for triangle counting

*/

#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/broadcast.cuh"
#include "pangolin/algorithm/fill.cuh"
#include "pangolin/algorithm/reduction.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/cuda_cxx/rc_stream.hpp"
#include "pangolin/dense/device_bit_vector.cuh"
#include "pangolin/dense/device_buffer.cuh"
#include "pangolin/dense/vector.cuh"
#include "search.cuh"

template <size_t BLOCK_DIM_X, typename CsrView>
__global__ void __launch_bounds__(BLOCK_DIM_X)
    dyn_thread_kernel(uint64_t *__restrict__ count, //!< [inout] the count, caller should zero
                      const CsrView adj,            //!< [in] the matrix
                      const size_t rowStart,        //!< [in] the starting row this kernel will count
                      const size_t numRows          //!< [in] the number of rows this kernel will count
    ) {
  typedef typename CsrView::edge_index_type EdgeIndex;
  typedef typename CsrView::node_index_type NodeIndex;
  constexpr size_t REG_CACHE_SIZE = 10;

  static_assert(BLOCK_DIM_X % 32 == 0, "block size should be multiple of 32");

  // each thread can cache short rows
  NodeIndex srcPartCache[REG_CACHE_SIZE];

  // per-thread triangle count
  uint64_t threadCount = 0;

  // each thread handles a row
  for (NodeIndex rowIdx = rowStart + BLOCK_DIM_X * blockIdx.x + threadIdx.x; rowIdx < rowStart + numRows;
       rowIdx += BLOCK_DIM_X * gridDim.x) {

      // the part of the row we are responsible for comparing for triangle counts
      const EdgeIndex srcPartStart = adj.partitionStart_[rowIdx];
      const EdgeIndex srcPartStop = adj.partitionStop_[rowIdx];
      const NodeIndex *srcPartBegin = &adj.colInd_[srcPartStart];
      const NodeIndex *srcPartEnd = &adj.colInd_[srcPartStop];
      const EdgeIndex srcPartSz = srcPartStop - srcPartStart;

      // the whole row
      const EdgeIndex srcStart = adj.rowStart_[rowIdx];
      const EdgeIndex srcStop = adj.rowStop_[rowIdx];
      const NodeIndex *srcBegin = &adj.colInd_[srcStart];
      const NodeIndex *srcEnd = &adj.colInd_[srcStop];
      // const EdgeIndex srcSz = srcStop - srcStart;

    if (srcPartSz == 0) {
      continue; // no triangles from empty row
    } else if (srcPartSz <= REG_CACHE_SIZE) {
      // cache the source row in the registers
#pragma unroll(REG_CACHE_SIZE)
      for (NodeIndex i = 0; i < REG_CACHE_SIZE; ++i) {
        if (i < srcPartSz) {
          srcPartCache[i] = adj.colInd_[srcPartStart + i];
        }
      }

      // compare each neighbor row
      for (const NodeIndex *srcPtr = srcBegin; srcPtr < srcEnd; ++srcPtr) {

        // using rowCache in here is a non-constant access , which causes an access from global memory anyway
        NodeIndex dst = *srcPtr;
        const EdgeIndex dstPartStart = adj.partitionStart_[dst];
        const EdgeIndex dstPartStop = adj.partitionStop_[dst];
        const NodeIndex *dstPartBegin = &adj.colInd_[dstPartStart];
        const NodeIndex *dstPartEnd = &adj.colInd_[dstPartStop];
        /*!
        unroll this loop so all accesses to rowCache[] are known statically
        and can be replaced by a register access
       */
        const NodeIndex *dstPtr = dstPartBegin;
#pragma unroll(REG_CACHE_SIZE)
        for (size_t regIdx = 0; regIdx < REG_CACHE_SIZE; ++regIdx) {
          // early exit if we have run out of values in either array
          if (dstPtr == dstPartEnd || regIdx == srcPartSz) {
            break;
          }

          // load the current non-zero from the row
          EdgeIndex srcVal = srcPartCache[regIdx];

          // catch dstPtr up to srcVal or the end of the list
          while (true) {

            // done if we are at the end of the list
            if (dstPtr == dstPartEnd) {
              break;
            }
            NodeIndex dstVal = *dstPtr;
            if (dstVal == srcVal) { // done if we have caught up
              threadCount++;
              dstPtr++;
              break;
            } else if (dstVal > srcVal) { // done if we have gone too far
              break;
            } else { // dstVal < srcVal
              dstPtr++;
            }
          }
        }
      }
    } else { // row is too large. read src from global memory
      for (const NodeIndex *srcPtr = srcBegin; srcPtr < srcEnd; ++srcPtr) {
        NodeIndex dst = *srcPtr;
        const EdgeIndex dstPartStart = adj.partitionStart_[dst];
        const EdgeIndex dstPartStop = adj.partitionStop_[dst];
        const NodeIndex *dstPartBegin = &adj.colInd_[dstPartStart];
        const NodeIndex *dstPartEnd = &adj.colInd_[dstPartStop];
        threadCount += pangolin::serial_sorted_count_linear(srcPartBegin, srcPartEnd, dstPartBegin, dstPartEnd);
      }
    }
  }

  // Block-wide reduction of threadCount
  typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);

  // Add to total count
  if (0 == threadIdx.x) {
    atomicAdd(count, aggregate);
  }
}

namespace pangolin {

class VertexDynTC {
private:
  int dev_;
  RcStream stream_;
  uint64_t *count_; //<! the triangle count

  // events for measuring time
  cudaEvent_t kernelStart_;
  cudaEvent_t kernelStop_;

public:
  explicit VertexDynTC(int dev) : dev_(dev), stream_(dev), count_(nullptr) {
    SPDLOG_TRACE(logger::console(), "set dev {}", dev_);
    CUDA_RUNTIME(cudaSetDevice(dev_));
    SPDLOG_TRACE(logger::console(), "mallocManaged");
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    zero_async<1>(count_, dev_, cudaStream_t(stream_)); // zero on the device that will do the counting

    CUDA_RUNTIME(cudaEventCreate(&kernelStart_));
    CUDA_RUNTIME(cudaEventCreate(&kernelStop_));
  }

  VertexDynTC(int dev, cudaStream_t stream) : dev_(dev), stream_(dev, stream), count_(nullptr) {
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    zero_async<1>(count_, dev_, cudaStream_t(stream_)); // zero on the device that will do the counting

    CUDA_RUNTIME(cudaEventCreate(&kernelStart_));
    CUDA_RUNTIME(cudaEventCreate(&kernelStop_));
  }

  VertexDynTC(VertexDynTC &&other)
      : dev_(other.dev_), stream_(other.stream_), count_(other.count_), kernelStart_(other.kernelStart_),
        kernelStop_(other.kernelStop_) {
    other.count_ = nullptr;
    other.kernelStart_ = nullptr;
    other.kernelStop_ = nullptr;
  }

  VertexDynTC() : VertexDynTC(0) {}
  ~VertexDynTC() {
    CUDA_RUNTIME(cudaFree(count_));
    if (kernelStart_) {
      CUDA_RUNTIME(cudaEventDestroy(kernelStart_));
    }
    if (kernelStop_) {
      CUDA_RUNTIME(cudaEventDestroy(kernelStop_));
    }
  }

  template <typename Csr>
  void count_async(const Csr &adj, const size_t rowOffset, const size_t numRows, const size_t dimBlock = 256) {
    assert(count_);
    assert(rowOffset + numRows <= adj.num_rows());
    CUDA_RUNTIME(cudaSetDevice(dev_));
    zero_async<1>(count_, dev_, cudaStream_t(stream_));
    CUDA_RUNTIME(cudaGetLastError());

#define CASE(const_dimBlock)                                                                                           \
  case const_dimBlock: {                                                                                               \
    int maxActiveBlocks;                                                                                               \
    CUDA_RUNTIME(cudaOccupancyMaxActiveBlocksPerMultiprocessor(                                                        \
        &maxActiveBlocks, dyn_thread_kernel<const_dimBlock, Csr>, const_dimBlock, 0));                                 \
    cudaDeviceProp props;                                                                                              \
    CUDA_RUNTIME(cudaGetDeviceProperties(&props, dev_));                                                               \
    const int dimGrid = maxActiveBlocks * props.multiProcessorCount;                                                   \
    LOG(debug, "dyn_thread_kernel({})<<<{}, {}, 0, {}>>>", dev_, dimGrid, const_dimBlock, stream_);                    \
    CUDA_RUNTIME(cudaEventRecord(kernelStart_, cudaStream_t(stream_)));                                                \
    dyn_thread_kernel<const_dimBlock>                                                                                  \
        <<<dimGrid, const_dimBlock, 0, cudaStream_t(stream_)>>>(count_, adj, rowOffset, numRows);                      \
    CUDA_RUNTIME(cudaEventRecord(kernelStop_, cudaStream_t(stream_)));                                                 \
    break;                                                                                                             \
  }

    switch (dimBlock) {
      CASE(32)
      CASE(64)
      CASE(128)
      CASE(256)
      CASE(512)
    default:
      LOG(critical, "unsupported block dimension {}", dimBlock);
      exit(1);
    }

#undef CASE
    CUDA_RUNTIME(cudaGetLastError());
  }

  template <typename Csr> void count_async(const Csr &adj, const size_t dimBlock = 256) {
    count_async(adj, 0, adj.num_rows(), dimBlock);
  }

  template <typename Csr>
  uint64_t count_sync(const Csr &adj, const size_t rowOffset, const size_t n, const size_t dimBlock = 256) {
    count_async(adj, rowOffset, n, dimBlock);
    sync();
    return count();
  }

  template <typename Csr> uint64_t count_sync(const Csr &adj, const size_t dimBlock = 256) {
    count_async(adj, dimBlock);
    sync();
    return count();
  }

  void sync() { stream_.sync(); }

  uint64_t count() const { return *count_; }
  int device() const { return dev_; }

  double kernel_time() const {
    float millis;
    CUDA_RUNTIME(cudaEventSynchronize(kernelStop_));
    CUDA_RUNTIME(cudaEventElapsedTime(&millis, kernelStart_, kernelStop_));
    return millis / 1e3;
  }
};

} // namespace pangolin