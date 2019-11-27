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
#include "pangolin/dense/buffer.cuh"
#include "pangolin/dense/device_bit_vector.cuh"
#include "pangolin/dense/device_buffer.cuh"
#include "pangolin/dense/vector.cuh"
#include "search.cuh"

template <typename T, typename Index> __device__ __forceinline__ void bitmap_set_atomic(T *bitmap, Index i) {
  Index field = i / (sizeof(T) * CHAR_BIT);
  Index bit = i % (sizeof(T) * CHAR_BIT);
  T bits = T(1) << bit;
  atomicOr(&bitmap[field], bits);
}

template <typename T, typename Index> __device__ __forceinline__ bool bitmap_get(T *bitmap, Index i) {
  Index fieldIdx = i / Index(sizeof(T) * CHAR_BIT);
  Index bitIdx = i % Index(sizeof(T) * CHAR_BIT);
  T bits = bitmap[fieldIdx];
  return (bits >> bitIdx) & T(1);
}

// clear all bits between [first, second]
// may reset more bits than that
template <typename T, typename Index> __device__ void block_bitmap_clear(T *bitmap, Index first, Index second) {
  const Index firstIdx = first / Index(sizeof(T) * CHAR_BIT);
  const Index secondIdx = second / Index(sizeof(T) * CHAR_BIT);
  for (Index i = firstIdx + threadIdx.x; i <= secondIdx; i += blockDim.x) {
    bitmap[i] = 0;
  }
}

// clear bit i
// may clear more bits than that
template <typename T, typename Index> __device__ __forceinline__ void bitmap_clear(T *bitmap, Index i) {
  Index fieldIdx = i / Index(sizeof(T) * CHAR_BIT);
  bitmap[fieldIdx] = 0;
}

template <size_t BLOCK_DIM_X, typename CsrView, typename bitmap_type>
__global__ void __launch_bounds__(BLOCK_DIM_X) dyn_blocks_low_tri_kernel(
    uint64_t *__restrict__ count, //!< [inout] the count, caller should zero
    const CsrView adj,            //!< [in] the matrix
    bitmap_type *globalBitmaps,   //!< [in] start of global memory bitmap space (caller should zero)
    const size_t globalBitmapsSz) //!< [in] the total size of the global memory bitmaps

{
  typedef typename CsrView::index_type Index;

  // align global memory bitmaps to 512 bytes
  constexpr size_t ALIGN_BYTES = 512;
  constexpr size_t ALIGN = ALIGN_BYTES / sizeof(bitmap_type);
  const size_t blockBitmapSz = (globalBitmapsSz / gridDim.x) / ALIGN * ALIGN;

  // set up shared bitmap
  constexpr size_t BITMAP_KB = 8;
  constexpr size_t BITMAP_MAX_FIELD = BITMAP_KB * 1024 / sizeof(bitmap_type);
  constexpr size_t BITMAP_MAX_BIT = BITMAP_MAX_FIELD * sizeof(bitmap_type) * CHAR_BIT;
  __shared__ bitmap_type sharedBitmap[BITMAP_MAX_FIELD];
  for (size_t i = 0; i < BITMAP_MAX_FIELD; i += blockDim.x) {
    sharedBitmap[i] = 0;
  }
  __syncthreads();

  uint64_t threadCount = 0;

  for (Index bi = blockIdx.x; bi < adj.num_rows(); bi += gridDim.x) {
    const Index io_s = adj.rowPtr_[bi];
    const Index io_e = adj.rowPtr_[bi + 1];

    // this row is empty, skip
    if (io_s == io_e) {
      continue;
    }

    // use shared memory if the largest column index is short enough
    bitmap_type *bitmap = nullptr;
    if (adj.colInd_[io_e - 1] < BITMAP_MAX_BIT) {
      // if (threadIdx.x == 0) {
      //   printf("row %llu: use shared bitmap\n", bi);
      // }
      bitmap = sharedBitmap;
    } else {
      // if (threadIdx.x == 0) {
      //   printf("row %llu: use global bitmap\n", bi);
      // }
      bitmap = &globalBitmaps[blockIdx.x * blockBitmapSz];
    }

    // want all threads active for clearing bitmaps later
    // ceil (io_e / BLOCK_DIM_X)
    Index blk_bound = (io_e + BLOCK_DIM_X - 1) / BLOCK_DIM_X * BLOCK_DIM_X;

    // whole block collaboartively sets bits for parts of a row
    for (Index io = io_s; io < blk_bound; io += blockDim.x) {
      const int64_t c = (io + threadIdx.x < io_e) ? adj.colInd_[io + threadIdx.x] : -1;
      if (c > -1) {
        bitmap_set_atomic(bitmap, c);
      }
      __syncthreads();
      for (short t = 0; t < blockDim.x; t++) {
        const int64_t j = pangolin::block_broadcast(c, t);
        if (j == -1) {
          break;
        }
        const Index jo_s = adj.rowPtr_[j];
        const Index jo_e = adj.rowPtr_[j + 1];
        for (Index jo = jo_s + threadIdx.x; jo < jo_e; jo += blockDim.x) {
          const int64_t k = adj.colInd_[jo];
          threadCount += bitmap_get(bitmap, k);
        }
      }
    }
    __syncthreads();
    if (io_s != io_e) {
      const Index first = adj.colInd_[io_s];
      const Index second = adj.colInd_[io_e - 1];
      const size_t numEntries = io_e - io_s;                           // the number of set bits
      const size_t numFields = (second - first) / sizeof(bitmap_type); // the range of fields that could be set
      if (numFields < numEntries) {
        block_bitmap_clear(bitmap, first, second);
      } else {
        for (Index i = io_s + threadIdx.x; i < io_e; i += blockDim.x) {
          bitmap_clear(bitmap, adj.colInd_[i]);
        }
      }
    }

    __syncthreads();
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

class BissonFaticaTC {
private:
  int dev_;
  cudaStream_t stream_;
  uint64_t *count_; //<! the triangle count

  // the global memory bitmaps
  DeviceBuffer<uint32_t> bitmaps_;

  // events for measuring time
  cudaEvent_t kernelStart_;
  cudaEvent_t kernelStop_;
  float time;

public:
  BissonFaticaTC(int dev, cudaStream_t stream = 0) : dev_(dev), stream_(stream), count_(nullptr) {
    SPDLOG_TRACE(logger::console(), "set dev {}", dev_);
    CUDA_RUNTIME(cudaSetDevice(dev_));
    SPDLOG_TRACE(logger::console(), "mallocManaged");
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    zero_async<1>(count_, dev_, cudaStream_t(stream_)); // zero on the device that will do the counting

    CUDA_RUNTIME(cudaEventCreate(&kernelStart_));
    CUDA_RUNTIME(cudaEventCreate(&kernelStop_));
  }

  BissonFaticaTC() : BissonFaticaTC(0) {}

  BissonFaticaTC(BissonFaticaTC &&other)
      : dev_(other.dev_), stream_(other.stream_), count_(other.count_), kernelStart_(other.kernelStart_),
        kernelStop_(other.kernelStop_) {
    other.count_ = nullptr;
    other.kernelStart_ = nullptr;
    other.kernelStop_ = nullptr;
  }

  ~BissonFaticaTC() {
    CUDA_RUNTIME(cudaFree(count_));
    if (kernelStart_) {
      CUDA_RUNTIME(cudaEventDestroy(kernelStart_));
    }
    if (kernelStop_) {
      CUDA_RUNTIME(cudaEventDestroy(kernelStop_));
    }
  }

  template <typename Csr> void count_async(const Csr &adj, const size_t dimBlock = 256) {
    assert(count_);
    CUDA_RUNTIME(cudaSetDevice(dev_));
    zero_async<1>(count_, dev_, cudaStream_t(stream_));
    CUDA_RUNTIME(cudaGetLastError());
    bitmaps_.resize(1 << 30);
    zero_async(bitmaps_.data(), bitmaps_.size(), 0, cudaStream_t(stream_));

#define CASE(const_dimBlock)                                                                                           \
  case const_dimBlock: {                                                                                               \
    int maxActiveBlocks;                                                                                               \
    CUDA_RUNTIME(cudaOccupancyMaxActiveBlocksPerMultiprocessor(                                                        \
        &maxActiveBlocks, dyn_blocks_low_tri_kernel<const_dimBlock, Csr, uint32_t>, const_dimBlock, 0));               \
    cudaDeviceProp props;                                                                                              \
    CUDA_RUNTIME(cudaGetDeviceProperties(&props, dev_));                                                               \
    const int dimGrid = maxActiveBlocks * props.multiProcessorCount;                                                   \
    LOG(debug, "dyn_blocks_low_tri_kernel({})<<<{}, {}, 0, {}>>>", dev_, dimGrid, const_dimBlock, uintptr_t(stream_)); \
    CUDA_RUNTIME(cudaEventRecord(kernelStart_, cudaStream_t(stream_)));                                                \
    dyn_blocks_low_tri_kernel<const_dimBlock>                                                                          \
        <<<dimGrid, const_dimBlock, 0, cudaStream_t(stream_)>>>(count_, adj, bitmaps_.data(), bitmaps_.size());        \
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

  template <typename Csr> uint64_t count_sync(const Csr &adj) {
    count_async(adj);
    sync();
    return count();
  }

  void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

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
