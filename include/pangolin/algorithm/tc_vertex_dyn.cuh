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

  constexpr size_t REG_CACHE_SIZE = 10;

  typedef typename CsrView::index_type Index;
  static_assert(BLOCK_DIM_X % 32 == 0, "block size should be multiple of 32");

  // each thread can cache short rows
  Index rowCache[REG_CACHE_SIZE];

  // per-thread triangle count
  uint64_t threadCount = 0;

  // each thread handles a row
  for (Index rowIdx = rowStart + blockDim.x * blockIdx.x + threadIdx.x; rowIdx < rowStart + numRows;
       rowIdx += blockDim.x * gridDim.x) {

    const Index rowStart = adj.rowPtr_[rowIdx];
    const Index rowStop = adj.rowPtr_[rowIdx + 1];
    const Index rowSz = rowStop - rowStart;

    if (rowSz == 0) {
      continue; // no triangles from empty row
    } else if (rowSz <= REG_CACHE_SIZE) {
      // cache the source row in the registers
#pragma unroll(REG_CACHE_SIZE)
      for (Index i = 0; i < REG_CACHE_SIZE; ++i) {
        if (i < rowSz) {
          rowCache[i] = adj.colInd_[rowStart + i];
        }
      }

      // compare each neighbor row
      for (size_t srcIdx = rowStart; srcIdx < rowStop; ++srcIdx) {

        // using rowCache in here is a non-constant access , which causes an access from global memory anyway
        Index nbr = adj.colInd_[srcIdx];
        Index nbrStart = adj.rowPtr_[nbr];
        Index nbrStop = adj.rowPtr_[nbr + 1];
        const Index *nbrBegin = &adj.colInd_[nbrStart];
        const Index *nbrEnd = &adj.colInd_[nbrStop];
        /*!
        unroll this loop so all accesses to rowCache[] are known statically
        and can be replaced by a register access
       */
        const Index *nbrPtr = nbrBegin;
#pragma unroll(REG_CACHE_SIZE)
        for (size_t regIdx = 0; regIdx < REG_CACHE_SIZE; ++regIdx) {
          // early exit if we have run out of values in either array
          if (nbrPtr == nbrEnd || regIdx == rowSz) {
            break;
          }

          // load the current non-zero from the row
          Index rowVal = rowCache[regIdx];

          // catch nbrPtr up to rowVal or the end of the list
          while (true) {

            // done if we are at the end of the list
            if (nbrPtr == nbrEnd) {
              break;
            }
            Index nbrVal = *nbrPtr;
            if (nbrVal == rowVal) { // done if we have caught up
              threadCount++;
              nbrPtr++;
              break;
            } else if (nbrVal > rowVal) { // done if we have gone too far
              break;
            } else { // nbrVal < rowVal
              nbrPtr++;
            }
          }
        }
      }
    } else { // row is too large. read src from global memory
      const Index *srcBegin = &adj.colInd_[rowStart];
      const Index *srcEnd = &adj.colInd_[rowStop];
      for (const Index *srcPtr = srcBegin; srcPtr < srcEnd; ++srcPtr) {
        Index src = *srcPtr;
        const Index *dstBegin = &adj.colInd_[adj.rowPtr_[src]];
        const Index *dstEnd = &adj.colInd_[adj.rowPtr_[src + 1]];
        threadCount += pangolin::serial_sorted_count_linear(srcBegin, srcEnd, dstBegin, dstEnd);
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
  VertexDynTC(int dev) : dev_(dev), stream_(dev), count_(nullptr) {
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

  template <typename Csr> uint64_t count_sync(const Csr &adj, const size_t rowOffset, const size_t n) {
    count_async(adj, rowOffset, n);
    sync();
    return count();
  }

  template <typename Csr> uint64_t count_sync(const Csr &adj) {
    count_async(adj, 0, adj.num_rows());
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