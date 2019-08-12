/*! A dynamic-selection algorithm for triangle counting

*/

#pragma once

#include <memory>

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/broadcast.cuh"
#include "pangolin/algorithm/fill.cuh"
#include "pangolin/algorithm/reduction.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/cuda_cxx/rc_stream.hpp"
#include "pangolin/dense/device_buffer.cuh"
#include "pangolin/dense/vector.cuh"
#include "search.cuh"


template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void __launch_bounds__(BLOCK_DIM_X) tc_edge_dysel_kernel(
    uint64_t *count,                    //!< [inout] the count, caller should zero
    const CsrCooView adj,               //!< [in] the matrix
    const size_t numEdges,              //!< [in] the number of edges this kernel will count
    const size_t edgeStart,             //!< [in] the starting edge this kernel will count
    size_t *edgeIdx,                    //!< [inout] a gpu memory area for work-stealing. caller should set to edgeStart
    volatile uint64_t throughputData[4] //!< [inout] an area for recording empirical throughput data
) {

  typedef typename CsrCooView::index_type Index;
  enum class Mode {
    LINEAR,
    BINARY,
  };

  static_assert(BLOCK_DIM_X % 32 == 0, "block size should be multiple of 32");
  constexpr size_t WARPS_PER_BLOCK = BLOCK_DIM_X / 32;

  // assign each thread a lane within a warp (lx) and a global warp id (gwx), and a warp id within the threadblock (wx)
  const size_t lx = threadIdx.x % 32;
  uint64_t threadCount = 0;
  Mode mode = Mode::BINARY;

  // average rates for a warp completing edges using the linear and binary methods
  volatile uint64_t *const linearEdges = &throughputData[0];
  volatile uint64_t *const linearClocks = &throughputData[1];
  volatile uint64_t *const binaryEdges = &throughputData[2];
  volatile uint64_t *const binaryClocks = &throughputData[3];


  while (true) {

    // have each warp try to claim 32 edges at a time
    size_t warpEdgeIdx;
    if (0 == lx) {
      warpEdgeIdx = atomicAdd(edgeIdx, 32);
    }
    warpEdgeIdx = pangolin::warp_broadcast2(warpEdgeIdx, 0 /*root*/);

    // bail out of loop if all lanes don't have a real edge
    if (warpEdgeIdx >= edgeStart + numEdges) {
      // if (0 == lx) {
      //   printf("warp %lu done at %lu\n", wx, warpEdgeIdx);
      // }
      break;
    }

    // block 0 is always linear, block 1 is always binary other blocks choose
    if (blockIdx.x == 0) {
      mode = Mode::LINEAR;
    } else if (blockIdx.x == 1) {
      mode = Mode::BINARY;
    } else {
      // this is a race condition 
      const double linearTpt = double(*linearEdges) / double(*linearClocks);
      const double binaryTpt = double(*binaryEdges) / double(*binaryClocks);
      if (linearTpt == binaryTpt) {
        mode = blockIdx.x % 2 ? Mode::LINEAR : Mode::BINARY;
      } else {
        mode = linearTpt > binaryTpt ? Mode::LINEAR : Mode::BINARY;
      }
    }



    // each lane computes the cost of a different edge
    const size_t i = warpEdgeIdx + lx;
    // if (lx == 19) {
    //   printf("warp %lu lane %lu: edge %lu\n", wx, lx, i);
    // }

    // look up neighbor lists for 32 edges in parallel
    const Index *srcBegin = nullptr;
    const Index *srcEnd = nullptr;
    const Index *dstBegin = nullptr;
    const Index *dstEnd = nullptr;
    Index srcSz = 0;
    Index dstSz = 0;
    if (i < edgeStart + numEdges) {
      const Index src = adj.rowInd_[i];
      const Index dst = adj.colInd_[i];

      srcBegin = &adj.colInd_[adj.rowPtr_[src]];
      srcEnd = &adj.colInd_[adj.rowPtr_[src + 1]];
      srcSz = srcEnd - srcBegin;
      dstBegin = &adj.colInd_[adj.rowPtr_[dst]];
      dstEnd = &adj.colInd_[adj.rowPtr_[dst + 1]];
      dstSz = dstEnd - dstBegin;
    }

    clock_t start = clock();

    // based on estimated costs, choose which approach all threads will take
    if (mode == Mode::LINEAR) {
      // use one thread per edge to do the linear search
      // lanes without an edge have nullptrs for begin and end, so they won't count;

      threadCount += pangolin::serial_sorted_count_linear(srcBegin, srcEnd, dstBegin, dstEnd);
    } else {
      for (size_t j = warpEdgeIdx; j < warpEdgeIdx + 32 && j < edgeStart + numEdges; ++j) {
        const Index *edgeSrcBegin = pangolin::warp_broadcast2(srcBegin, j - warpEdgeIdx);
        const Index edgeSrcSz = pangolin::warp_broadcast2(srcSz, j - warpEdgeIdx);
        const Index *edgeDstBegin = pangolin::warp_broadcast2(dstBegin, j - warpEdgeIdx);
        const Index edgeDstSz = pangolin::warp_broadcast2(dstSz, j - warpEdgeIdx);

        if (edgeSrcSz > edgeDstSz) {
          threadCount += pangolin::warp_sorted_count_binary<1, WARPS_PER_BLOCK, Index, false /*no reduction*/>(
              edgeDstBegin, edgeDstSz, edgeSrcBegin, edgeSrcSz);
        } else {
          threadCount += pangolin::warp_sorted_count_binary<1, WARPS_PER_BLOCK, Index, false /*no reduction*/>(
              edgeSrcBegin, edgeSrcSz, edgeDstBegin, edgeDstSz);
        }
      }
    }

    if (blockIdx.x == 0) {
      if (0 == lx) {
        atomicAdd(linearClocks, uint64_t(clock() - start));
        atomicAdd(linearEdges, uint64_t(32));
      }
    } else if (blockIdx.x == 1) {
      if (0 == lx) {
        atomicAdd(binaryClocks, uint64_t(clock() - start));
        atomicAdd(binaryEdges, uint64_t(32));
      }
    }
  }

  // if (threadCount != 0) {
  //   printf("warp %lu lane %lu tris total %lu \n", wx, lx, threadCount);
  // }

  // Block-wide reduction of threadCount
  typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);

  // if (0 == threadIdx.x) {
  //   printf("block %d addregate %lu \n", blockIdx.x, aggregate);
  // }

  // Add to total count
  if (0 == threadIdx.x) {
    atomicAdd(count, aggregate);
  }
}

namespace pangolin {

class EdgeWarpDyselTC {
private:
  int dev_;
  RcStream stream_;                       //!< the stream that this triangle counter will use
  uint64_t *count_;                       //!< the triangle count
  DeviceBuffer<size_t> edgeIdx_;          //!< index of the next available edge for counting
  DeviceBuffer<uint64_t> throughputInfo_; //!< storage space for empirical information about throughput

  // events for measuring time
  cudaEvent_t kernelStart_;
  cudaEvent_t kernelStop_;

public:
  EdgeWarpDyselTC(int dev)
      : dev_(dev), stream_(std::move(RcStream(dev))), count_(nullptr), edgeIdx_(1, dev), throughputInfo_(4, dev) {
    SPDLOG_TRACE(logger::console(), "set dev {}", dev_);
    CUDA_RUNTIME(cudaSetDevice(dev_));
    SPDLOG_TRACE(logger::console(), "mallocManaged");
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));

    CUDA_RUNTIME(cudaEventCreate(&kernelStart_));
    CUDA_RUNTIME(cudaEventCreate(&kernelStop_));

    zero_async<1>(count_, dev_, cudaStream_t(stream_)); // zero on the device that will do the counting

    // CUDA_RUNTIME(cudaHostAlloc(&count_, sizeof(*count_), cudaHostAllocPortable | cudaHostAllocMapped));
    // *count_ = 0;
  }

  EdgeWarpDyselTC(int dev, cudaStream_t stream)
      : dev_(dev), stream_(std::move(RcStream(dev, stream))), count_(nullptr), edgeIdx_(1, dev),
        throughputInfo_(4, dev) {
    if (stream_.device() != dev) {
      LOG(critical, "device and stream device do not match");
      exit(1);
    }
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    zero_async<1>(count_, dev_, cudaStream_t(stream_)); // zero on the device that will do the counting
    // error may be deferred to a cudaHostGetDevicePointer
    // CUDA_RUNTIME(cudaHostAlloc(&count_, sizeof(*count_), cudaHostAllocPortable | cudaHostAllocMapped));
    // *count_ = 0;

    CUDA_RUNTIME(cudaEventCreate(&kernelStart_));
    CUDA_RUNTIME(cudaEventCreate(&kernelStop_));
  }

  EdgeWarpDyselTC(EdgeWarpDyselTC &&other)
      : dev_(other.dev_), stream_(std::move(other.stream_)), count_(other.count_), edgeIdx_(std::move(other.edgeIdx_)),
        throughputInfo_(std::move(other.throughputInfo_)), kernelStart_(other.kernelStart_),
        kernelStop_(other.kernelStop_) {
    other.count_ = nullptr;
    other.kernelStart_ = nullptr;
    other.kernelStop_ = nullptr;
  }

  EdgeWarpDyselTC() : EdgeWarpDyselTC(0) {}
  ~EdgeWarpDyselTC() {
    CUDA_RUNTIME(cudaFree(count_));

    if (kernelStart_) {
      CUDA_RUNTIME(cudaEventDestroy(kernelStart_));
    }
    if (kernelStop_) {
      CUDA_RUNTIME(cudaEventDestroy(kernelStop_));
    }
  }

  template <typename CsrCoo>
  void count_async(const CsrCoo &adj, const size_t edgeOffset, const size_t numEdges, const size_t dimBlock = 256) {
    assert(count_);
    assert(edgeOffset + numEdges <= adj.nnz());
    assert(edgeIdx_.data());
    assert(throughputInfo_.data());
    // CUDA_RUNTIME(cudaSetDevice(dev_)); // FIXME: needed?
    // stream_.sync();                    // FIXME: needed?
    zero_async<1>(count_, dev_, cudaStream_t(stream_));
    CUDA_RUNTIME(cudaGetLastError());
    device_fill(edgeIdx_.data(), 1, edgeOffset);
    device_fill(throughputInfo_.data(), 4, uint64_t(0));
    CUDA_RUNTIME(cudaGetLastError());

#define CASE(const_dimBlock)                                                                                           \
  case const_dimBlock: {                                                                                               \
    int maxActiveBlocks;                                                                                               \
    CUDA_RUNTIME(cudaOccupancyMaxActiveBlocksPerMultiprocessor(                                                        \
        &maxActiveBlocks, tc_edge_dysel_kernel<const_dimBlock, CsrCoo>, const_dimBlock, 0));                           \
    cudaDeviceProp props;                                                                                              \
    CUDA_RUNTIME(cudaGetDeviceProperties(&props, dev_));                                                               \
    const int dimGrid = maxActiveBlocks * props.multiProcessorCount;                                                   \
    LOG(debug, "device = {}, tc_edge_dysel_kernel<<<{}, {}, 0, {}>>>", dev_, dimGrid, const_dimBlock, stream_);        \
    CUDA_RUNTIME(cudaEventRecord(kernelStart_, cudaStream_t(stream_)));                                                \
    tc_edge_dysel_kernel<const_dimBlock><<<dimGrid, const_dimBlock, 0, cudaStream_t(stream_)>>>(                       \
        count_, adj, numEdges, edgeOffset, edgeIdx_.data(), throughputInfo_.data());                                   \
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

  template <typename CsrCoo> uint64_t count_sync(const CsrCoo &adj, const size_t edgeOffset, const size_t n) {
    count_async(adj, edgeOffset, n);
    sync();
    return count();
  }

  template <typename CsrCoo> uint64_t count_sync(const CsrCoo &adj) {
    count_async(adj, 0, adj.nnz());
    sync();
    return count();
  }

  void sync() { stream_.sync(); }

  uint64_t count() const { return *count_; }
  int device() const { return dev_; }

  /*! Get the time spent by the last kernel in seconds
   */
  double kernel_time() const {
    float millis;
    CUDA_RUNTIME(cudaEventSynchronize(kernelStop_));
    CUDA_RUNTIME(cudaEventElapsedTime(&millis, kernelStart_, kernelStop_));
    return millis / 1e3;
  }
};

} // namespace pangolin