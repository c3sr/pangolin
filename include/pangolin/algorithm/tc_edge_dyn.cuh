/*! A dynamic-selection algorithm for triangle counting

*/

#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/broadcast.cuh"
#include "pangolin/algorithm/fill.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/device_buffer.cuh"
#include "pangolin/dense/vector.hu"
#include "search.cuh"

template <size_t BLOCK_DIM_X, typename CsrCooView>
__global__ void __launch_bounds__(BLOCK_DIM_X)
    tc_edge_dyn_kernel(uint64_t *count,        //!< [inout] the count, caller should zero
                       const CsrCooView adj,   //<! [in] the matrix
                       const size_t numEdges,  //<! [in] the number of edges this kernel will count
                       const size_t edgeStart, // <! [in] the starting edge this kernel will count
                       size_t *edgeIdx //<! [inout] a gpu memory area for work-stealing. caller should set to edgeStart
    ) {

  typedef typename CsrCooView::index_type Index;

  static_assert(BLOCK_DIM_X % 32 == 0, "block size should be multiple of 32");
  constexpr size_t warpsPerBlock = BLOCK_DIM_X / 32;

  // per-warp estimated costs of linear and binary search method
  __shared__ size_t linearCost[warpsPerBlock];
  __shared__ size_t binaryCost[warpsPerBlock];

  // assign each thread a lane within a warp (lx) and a global warp id (gwx), and a warp id within the threadblock (wx)
  const size_t lx = threadIdx.x % 32;
  const size_t wx = threadIdx.x / 32;
  uint64_t threadCount = 0;

  // size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
  // const size_t gwx = (BLOCK_DIM_X * blockIdx.x + threadIdx.x) / 32;

  // have each warp try to claim 32 edges at a time
  while (true) {
    size_t warpEdgeIdx;

    if (0 == lx) {
      warpEdgeIdx = atomicAdd(edgeIdx, 32);
    }

    // if (lx == 0) {
    //   printf("warp %lu starting at edge %lu [%lu, %lu)\n", wx, warpEdgeIdx, edgeStart,
    //          edgeStart + numEdges);
    // }

    // broadcast the starting edge of the warp to all lanes
    warpEdgeIdx = pangolin::warp_broadcast<warpsPerBlock>(warpEdgeIdx, 0 /*root*/);

    // bail out of loop if all lanes don't have a real edge
    if (warpEdgeIdx >= edgeStart + numEdges) {
      // if (0 == lx) {
      //   printf("warp %lu done at %lu\n", wx, warpEdgeIdx);
      // }
      break;
    }

    // zero the cost of these 32 edges
    if (0 == lx) {
      linearCost[wx] = 0;
    }
    if (1 == lx) {
      binaryCost[wx] = 0;
    }
    __syncwarp();

    // each lane computes the cost of a different edge
    const size_t i = warpEdgeIdx + lx;
    // if (lx == 19) {
    //   printf("warp %lu lane %lu: edge %lu\n", wx, lx, i);
    // }

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

      // FIXME: better warp reduction?
      // contribute the linear cost of my edge to the total
      size_t edgeLinearCost = srcSz + dstSz;
      atomicAdd(&linearCost[wx], edgeLinearCost);

      // contribute the binary cost of my edge to the total
      // FIXME: better warp reduction?
      size_t edgeBinaryCost;
      if (srcSz > dstSz) {
        edgeBinaryCost = dstSz * (sizeof(srcSz) * CHAR_BIT - __clz(srcSz));
      } else {
        edgeBinaryCost = srcSz * (sizeof(dstSz) * CHAR_BIT - __clz(dstSz));
      }
      atomicAdd(&binaryCost[wx], edgeBinaryCost);
    }
    __syncwarp(); // wait for all threads in the warp to have contributed to the cost
    // if (lx == 0) {
    //   printf("warp %lu @ edge %lu: linear %lu binary %lu\n", wx, warpEdgeIdx, linearCost[wx], binaryCost[wx]);
    // }

    // based on estimated costs, choose which approach all threads will take
    if (linearCost < binaryCost) {
      // use one thread per edge to do the linear search
      // lanes without an edge have nullptrs for begin and end, so they won't count;
      threadCount += pangolin::serial_sorted_count_linear(srcBegin, srcEnd, dstBegin, dstEnd);
    } else {
      // use all threads in the warp to do a parallel binary search for each edge
      for (size_t j = warpEdgeIdx; j < warpEdgeIdx + 32 && j < edgeStart + numEdges; ++j) {

        // if (lx == 0) {
        //   printf("warp %lu working on binary edge %lu\n", wx, j);
        // }

        // FIXME: some lane already has these values, no need to reload
        const Index src = adj.rowInd_[j];
        const Index dst = adj.colInd_[j];
        const Index *srcBegin = &adj.colInd_[adj.rowPtr_[src]];
        const Index *srcEnd = &adj.colInd_[adj.rowPtr_[src + 1]];
        const Index srcSz = srcEnd - srcBegin;
        const Index *dstBegin = &adj.colInd_[adj.rowPtr_[dst]];
        const Index *dstEnd = &adj.colInd_[adj.rowPtr_[dst + 1]];
        const Index dstSz = dstEnd - dstBegin;

        if (srcSz > dstSz) {
          threadCount += pangolin::warp_sorted_count_binary<1, warpsPerBlock, Index, false /*no reduction*/>(
              dstBegin, dstSz, srcBegin, srcSz);
        } else {
          threadCount += pangolin::warp_sorted_count_binary<1, warpsPerBlock, Index, false /*no reduction*/>(
              srcBegin, srcSz, dstBegin, dstSz);
        }
        // if (threadCount != 0) {
        //   printf("warp %lu lane %lu tris so far %lu \n", wx, lx, threadCount);
        // }
      }
    }

    // wait for everyone in the warp to be done before potentially modifying the cost again
    __syncwarp();
  }

  // if (threadCount != 0) {
  //   printf("warp %lu lane %lu tris total %lu \n", wx, lx, threadCount);
  // }

  __syncthreads();
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

class EdgeWarpDynTC {
private:
  int dev_;
  cudaStream_t stream_;
  uint64_t *count_;              //<! the triangle count
  DeviceBuffer<size_t> edgeIdx_; //<! index of the next available edge for counting
  bool destroyStream_;

public:
  EdgeWarpDynTC(int dev)
      : dev_(dev), stream_(nullptr), count_(nullptr), edgeIdx_(DeviceBuffer<size_t>(1, dev)), destroyStream_(true) {
    SPDLOG_TRACE(logger::console(), "set dev {}", dev_);
    CUDA_RUNTIME(cudaSetDevice(dev_));
    SPDLOG_TRACE(logger::console(), "create stream");
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
    SPDLOG_TRACE(logger::console(), "mallocManaged");
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting

    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    // CUDA_RUNTIME(cudaHostAlloc(&count_, sizeof(*count_), cudaHostAllocPortable | cudaHostAllocMapped));
    // *count_ = 0;
  }

  EdgeWarpDynTC(int dev, cudaStream_t stream) : dev_(dev), stream_(stream), count_(nullptr), destroyStream_(false) {
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    // error may be deferred to a cudaHostGetDevicePointer
    // CUDA_RUNTIME(cudaHostAlloc(&count_, sizeof(*count_), cudaHostAllocPortable | cudaHostAllocMapped));
    // *count_ = 0;
  }

  EdgeWarpDynTC(EdgeWarpDynTC &&other)
      : dev_(other.dev_), stream_(other.stream_), count_(other.count_), destroyStream_(other.destroyStream_) {
    other.count_ = nullptr;
    other.destroyStream_ = false;
    other.stream_ = nullptr;
  }

  EdgeWarpDynTC() : EdgeWarpDynTC(0) {}
  ~EdgeWarpDynTC() {
    if (destroyStream_ && stream_) {
      SPDLOG_TRACE(logger::console(), "destroy stream {}", uintptr_t(stream_));
      CUDA_RUNTIME(cudaStreamDestroy(stream_));
    }
    CUDA_RUNTIME(cudaFree(count_));
  }

  template <typename CsrCoo>
  void count_async(const CsrCoo &mat, const size_t edgeOffset, const size_t numEdges, const size_t dimBlock = 256) {
    assert(count_);
    assert(edgeOffset + numEdges <= mat.nnz());
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    zero_async<1>(count_, dev_, stream_);
    CUDA_RUNTIME(cudaGetLastError());
    device_fill(edgeIdx_.data(), 1, edgeOffset);
    // zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting
    // uint64_t *devCount = nullptr;
    // CUDA_RUNTIME(cudaHostGetDevicePointer(&devCount, count_, 0));
    // LOG(debug, "zero {}", uintptr_t(count_));
    // *count_ = 0;
    // LOG(debug, "did zero");

#define CASE(const_dimBlock)                                                                                           \
  case const_dimBlock: {                                                                                               \
    const int dimGrid = (numEdges + dimBlock - 1) / const_dimBlock;                                                    \
    LOG(debug, "device = {}, tc_edge_dyn_kernel<<<{}, {}, 0, {}>>>", dev_, dimGrid, const_dimBlock, uintptr_t(stream_));                 \
    tc_edge_dyn_kernel<const_dimBlock>                                                                                 \
        <<<dimGrid, const_dimBlock, 0, stream_>>>(count_, mat, numEdges, edgeOffset, edgeIdx_.data());                 \
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

  template <typename CsrCoo> uint64_t count_sync(const CsrCoo &mat, const size_t edgeOffset, const size_t n) {
    count_async(mat, edgeOffset, n);
    sync();
    return count();
  }

  template <typename CsrCoo> uint64_t count_sync(const CsrCoo &adj) {
    count_async(adj, 0, adj.nnz());
    sync();
    return count();
  }

  void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

  uint64_t count() const { return *count_; }
  int device() const { return dev_; }
};

} // namespace pangolin