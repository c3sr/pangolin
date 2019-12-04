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
#include "pangolin/logger.hpp"
#include "pangolin/sparse/bitmap.cuh"
#include "search.cuh"

/*
Max registers
        th   bl   div
3.5+:  255   64k  256
7.0:   255   64k  256
7.5:   255   64k  256
*/

template <size_t BLOCK_DIM_X, typename CsrView>
__global__ void __launch_bounds__(BLOCK_DIM_X)
    thread_kernel(uint64_t *__restrict__ count, //!< [inout] the count, caller should zero
                  const CsrView adj             //!< [in] the matrix
    )

{
  typedef typename CsrView::index_type Index;

  // per-thread row
  constexpr size_t REG_SZ = 32;
  Index regRow[REG_SZ];

  uint64_t threadCount = 0;

  // one thread per row
  for (Index ti = blockDim.x * blockIdx.x + threadIdx.x; ti < adj.num_rows(); ti += gridDim.x * blockDim.x) {
    const Index io_s = adj.rowPtr_[ti];
    const Index io_e = adj.rowPtr_[ti + 1];

    // this row is empty, skip
    if (io_s == io_e) {
      continue;
    }

    // if the row is short enough, copy to register file
    if (io_e - io_s < REG_SZ) {

      // copy row to local memory
      asm("/*load regs*/");
#pragma unroll(REG_SZ)
      for (size_t i = 0; i < REG_SZ; ++i) {
        if (i < io_e - io_s) {

          regRow[i] = adj.colInd_[i + io_s];
        } else {
          break;
        }
      }

      // search through row
      // could unroll this and c = regRow[i] instead
      for (size_t i = io_s; i < io_e; ++i) {
        // if (io_s + i >= io_e) {
        //   break;
        // }
        const Index c = adj.colInd_[i];
        const Index jo_s = adj.rowPtr_[c];
        const Index jo_e = adj.rowPtr_[c + 1];

        // count number of intersections between row j and regRow
        // expect regRow[i] to be a register
        Index jo = jo_s;
        int64_t k;
        if (jo < jo_e) {
          k = adj.colInd_[jo];
        }
#pragma unroll(REG_SZ)
        for (size_t i = 0; i < REG_SZ; ++i) {
          if (jo >= jo_e) { // outside of j
            break;
          }
          if (i >= io_e - io_s) { // outside of i
            break;
          }

          int64_t c = regRow[i];

          // go to next c
          if (c < k) {
            continue; // next c
          }

          // go through k until it reaches c
          while (c > k) {
            ++jo;
            if (jo < jo_e) {
              k = adj.colInd_[jo];
            } else {
              goto done; // triangle counting done
            }
          }

          if (c == k) {
            ++threadCount;
            k = adj.colInd_[++jo];
            continue; // next c
          }
        }
      done:
        asm("/*done*/");
      }

    } else { // otherwise, don't load row into registers

      for (size_t i = io_s; i < io_e; ++i) {
        const Index c = adj.colInd_[i];
        const Index jo_s = adj.rowPtr_[c];
        const Index jo_e = adj.rowPtr_[c + 1];
        threadCount += pangolin::serial_sorted_count_linear(&adj.colInd_[io_s], &adj.colInd_[io_e], &adj.colInd_[jo_s],
                                                            &adj.colInd_[jo_e]);
      }
    }
  }

  // Block-wide reduction of threadCount
  asm("/*cub_reduction*/");
  typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);

  // Add to total count
  asm("/*atomic_add*/");
  if (0 == threadIdx.x) {
    atomicAdd(count, aggregate);
  }
}

#define PANGOLIN_SYNC_WARP() __syncthreads()

template <size_t BLOCK_DIM_X, typename CsrView, typename bitmap_type>
__global__ void __launch_bounds__(BLOCK_DIM_X) warp_kernel(
    uint64_t *__restrict__ count,            //!< [inout] the count, caller should zero
    const CsrView adj,                       //!< [in] the matrix
    bitmap_type *__restrict__ globalBitmaps, //!< [in] start of global memory bitmap space (caller should zero)
    const size_t globalBitmapsSz)            //!< [in] the total size of the global memory bitmaps

{
  typedef typename CsrView::index_type Index;

  // compute warp index
  constexpr size_t WARPS_PER_BLOCK = BLOCK_DIM_X / 32;
  static_assert(BLOCK_DIM_X % 32 == 0, "expect block size multiple of 32");
  const size_t lx = threadIdx.x % 32; // lane idx
  const size_t wx = threadIdx.x / 32; // warp idx
  const size_t warpsPerGrid = WARPS_PER_BLOCK * gridDim.x;

  // align global memory bitmaps to 512 bytes for each warp
  // round each warp bitmap size down to multiple of 512 bytes
  constexpr size_t ALIGN_BYTES = 512;
  constexpr size_t ALIGN = ALIGN_BYTES / sizeof(bitmap_type);
  const size_t warpBitmapSz = (globalBitmapsSz / warpsPerGrid) / ALIGN * ALIGN;
  bitmap_type *bitmap = &globalBitmaps[(blockIdx.x * WARPS_PER_BLOCK + wx) * warpBitmapSz];

  // set up shared memory
  constexpr size_t SHMEM_KB = 8;
  constexpr size_t SHMEM_SZ = SHMEM_KB * 1024 / sizeof(Index);
  constexpr size_t WARP_SHMEM_SZ = SHMEM_SZ / WARPS_PER_BLOCK;
  __shared__ Index sharedRow[WARPS_PER_BLOCK][WARP_SHMEM_SZ];

  // zero shared memory
  for (size_t i = lx; i < WARP_SHMEM_SZ; i += 32) {
    sharedRow[wx][i] = 0;
  }
  PANGOLIN_SYNC_WARP();

  uint64_t threadCount = 0;

  // one warp per row
  for (Index wi = wx + blockIdx.x * WARPS_PER_BLOCK; wi < adj.num_rows(); wi += WARPS_PER_BLOCK * gridDim.x) {
    const Index io_s = adj.rowPtr_[wi];
    const Index io_e = adj.rowPtr_[wi + 1];

    // this row is empty, skip
    if (io_s == io_e) {
      continue;
    }

    // if the row is short enough, copy to shared buffer
    if (0 && io_e - io_s < WARP_SHMEM_SZ) {

      for (size_t i = lx; i < io_e - io_s; i += 32) {
        sharedRow[wx][i] = adj.colInd_[i + io_s];
      }
      PANGOLIN_SYNC_WARP();

      // warp collaboratively searches each neighbord adj list in sharedRow
      for (size_t i = 0; i < io_e - io_s; i += 32) {
        const Index c = sharedRow[wx][i];
        const Index jo_s = adj.rowPtr_[c];
        const Index jo_e = adj.rowPtr_[c + 1];
        for (Index jo = jo_s + lx; jo < jo_e; jo += 32) {
          const int64_t k = adj.colInd_[jo];
          // scan sharedRow from i (the index of c) backward to the first element smaller than k (the searched value)
          for (int64_t si = i; si >= 0; --si) {
            if (sharedRow[wx][si] == k) {
              ++threadCount;
              break;
            } else if (sharedRow[wx][si] < k) {
              break;
            } else {
              assert(0 && "how did we get here");
            }
          }
        }
      }
      PANGOLIN_SYNC_WARP();

    } else { // otherwise, use bitmap

      if (lx == 0 && wx + blockIdx.x * WARPS_PER_BLOCK == 0) {
        printf("row %u %u %u\n", wi, io_s, io_e);
      }

      // want all threads active for clearing bitmaps later
      // ceil (io_e / 32)
      Index warpBound = (io_e + 32 - 1) / 32 * 32;

      // warp collaboratively sets bits for row
      for (Index io = io_s; io < warpBound; io += 32) {
        const int64_t c = (io + lx < io_e) ? adj.colInd_[io + lx] : -1;
        if (c > -1) {
          bitmap_set_atomic(bitmap, c);
          if (wx + blockIdx.x * WARPS_PER_BLOCK == 0) {
            printf("  set %lu\n", c);
          }
        }
        PANGOLIN_SYNC_WARP();
        for (short t = 0; t < 32; t++) {
          const int64_t j = pangolin::warp_broadcast(c, t);
          if (j == -1) {
            break;
          }
          const Index jo_s = adj.rowPtr_[j];
          const Index jo_e = adj.rowPtr_[j + 1];
          for (Index jo = jo_s + lx; jo < jo_e; jo += 32) {
            const int64_t k = adj.colInd_[jo];
            bool get = bitmap_get(bitmap, k);
            if (wx + blockIdx.x * WARPS_PER_BLOCK == 0 && get) {
              printf("  found %lu\n", k);
            }
            threadCount += get;
          }
        }
      }
      PANGOLIN_SYNC_WARP();
      const Index first = adj.colInd_[io_s];
      const Index second = adj.colInd_[io_e - 1];
      const size_t numEntries = io_e - io_s; // the number of set bits
      const size_t numFields =
          (second - first + 1 + sizeof(bitmap_type) - 1) / sizeof(bitmap_type); // the range of fields that could be set
      // if (lx == 0) {
      //   printf("first = %u second = %u numFields = %lu numEntries = %lu\n", first, second, numFields, numEntries);
      // }
      if (numFields < numEntries) {
        warp_bitmap_clear(bitmap, first, second, lx);
      } else {
        for (Index i = io_s + lx; i < io_e; i += 32) {
          bitmap_clear(bitmap, adj.colInd_[i]);
        }
      }

      PANGOLIN_SYNC_WARP();
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

template <size_t BLOCK_DIM_X, typename CsrView>
__global__ void __launch_bounds__(BLOCK_DIM_X)
    block_shared_kernel(uint64_t *__restrict__ count, //!< [inout] the count, caller should zero
                        const CsrView adj             //!< [in] the matrix
    )

{
  typedef typename CsrView::index_type Index;
  typedef uint32_t bitmap_type;

  // set up shared bitmap
  constexpr size_t BITMAP_KB = 8;
  constexpr size_t BITMAP_MAX_FIELD = BITMAP_KB * 1024 / sizeof(bitmap_type);
  __shared__ bitmap_type bitmap[BITMAP_MAX_FIELD];
  for (size_t i = threadIdx.x; i < BITMAP_MAX_FIELD; i += blockDim.x) {
    bitmap[i] = 0;
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

    // want all threads active for clearing bitmaps later
    // ceil (io_e / BLOCK_DIM_X)
    Index blkBound = (io_e + BLOCK_DIM_X - 1) / BLOCK_DIM_X * BLOCK_DIM_X;

    // whole block collaboartively sets bits for parts of a row
    for (Index io = io_s; io < blkBound; io += blockDim.x) {
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

template <size_t BLOCK_DIM_X, typename CsrView, typename bitmap_type>
__global__ void __launch_bounds__(BLOCK_DIM_X) block_global_kernel(
    uint64_t *__restrict__ count,            //!< [inout] the count, caller should zero
    const CsrView adj,                       //!< [in] the matrix
    bitmap_type *__restrict__ globalBitmaps, //!< [in] start of global memory bitmap space (caller should zero)
    const size_t globalBitmapsSz)            //!< [in] the total size of the global memory bitmaps

{
  typedef typename CsrView::index_type Index;

  // align global memory bitmaps to 512 bytes
  constexpr size_t ALIGN_BYTES = 512;
  constexpr size_t ALIGN = ALIGN_BYTES / sizeof(bitmap_type);
  const size_t blockBitmapSz = (globalBitmapsSz / gridDim.x) / ALIGN * ALIGN;
  bitmap_type *bitmap = &globalBitmaps[blockIdx.x * blockBitmapSz];

  uint64_t threadCount = 0;

  for (Index bi = blockIdx.x; bi < adj.num_rows(); bi += gridDim.x) {
    const Index io_s = adj.rowPtr_[bi];
    const Index io_e = adj.rowPtr_[bi + 1];

    // this row is empty, skip
    if (io_s == io_e) {
      continue;
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
  typedef uint32_t bitmap_type;
  DeviceBuffer<bitmap_type> bitmaps_;

  // events for measuring time
  cudaEvent_t kernelStart_;
  cudaEvent_t kernelStop_;
  float time;

  // device properties for computing kernel dimensions
  int multiProcessorCount_;

public:
  BissonFaticaTC(int dev, cudaStream_t stream = 0) : dev_(dev), stream_(stream), count_(nullptr) {
    SPDLOG_TRACE(logger::console(), "set dev {}", dev_);
    CUDA_RUNTIME(cudaSetDevice(dev_));
    SPDLOG_TRACE(logger::console(), "mallocManaged");
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting

    CUDA_RUNTIME(cudaEventCreate(&kernelStart_));
    CUDA_RUNTIME(cudaEventCreate(&kernelStop_));

    cudaDeviceProp props;
    CUDA_RUNTIME(cudaGetDeviceProperties(&props, dev_));
    multiProcessorCount_ = props.multiProcessorCount;
    LOG(debug, "dev {} sm count {}", dev_, multiProcessorCount_);
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
    zero_async<1>(count_, dev_, stream_);
    CUDA_RUNTIME(cudaGetLastError());

    // compute the average nnz per row
    double nnzPerRow = double(adj.nnz()) / double(adj.num_rows());
    LOG(debug, "{} nnz per row", nnzPerRow);

    if (nnzPerRow < 3.5) { // thread_kernel
      constexpr size_t const_dimBlock = 256;
      int maxActiveBlocks;
      CUDA_RUNTIME(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, thread_kernel<const_dimBlock, Csr>,
                                                                 const_dimBlock, 0));
      const int dimGrid = maxActiveBlocks * multiProcessorCount_;
      LOG(debug, "thread_kernel: max blocks = {} grid = {}", maxActiveBlocks, dimGrid);
      CUDA_RUNTIME(cudaEventRecord(kernelStart_, stream_));
      thread_kernel<const_dimBlock><<<dimGrid, const_dimBlock, 0, stream_>>>(count_, adj);
      CUDA_RUNTIME(cudaEventRecord(kernelStop_, stream_));

    } else if (nnzPerRow < 38) { // warp_kernel
      LOG(debug, "selected warp approach", nnzPerRow);
      // determine the bitmap size
      const size_t dimGrid = 10;
      constexpr size_t const_dimBlock = 256;
      const size_t numWarps = dimGrid * const_dimBlock / 32;
      const size_t bitmapSzPerWarp = (adj.num_rows() + sizeof(bitmap_type) - 1) / sizeof(bitmap_type);
      const size_t bitmapSz = bitmapSzPerWarp * numWarps;
      bitmaps_.resize(bitmapSz);
      zero_async(bitmaps_.data(), bitmaps_.size(), 0, cudaStream_t(stream_));
      CUDA_RUNTIME(cudaEventRecord(kernelStart_, cudaStream_t(stream_)));
      warp_kernel<const_dimBlock>
          <<<dimGrid, const_dimBlock, 0, cudaStream_t(stream_)>>>(count_, adj, bitmaps_.data(), bitmaps_.size());
      CUDA_RUNTIME(cudaEventRecord(kernelStop_, cudaStream_t(stream_)));

    } else if (adj.num_rows() < 65536) { // block_shared_kernel
      LOG(debug, "selected block-shared approach", nnzPerRow);
      // determine the bitmap size

      const int dimGrid = 10;
      constexpr size_t const_dimBlock = 256;
      CUDA_RUNTIME(cudaEventRecord(kernelStart_, cudaStream_t(stream_)));
      block_shared_kernel<const_dimBlock><<<dimGrid, const_dimBlock, 0, cudaStream_t(stream_)>>>(count_, adj);
      CUDA_RUNTIME(cudaEventRecord(kernelStop_, cudaStream_t(stream_)));

    } else { // block_global_kernel
      LOG(debug, "selected block-global approach", nnzPerRow);
      // determine the bitmap size
      const size_t dimGrid = 10;
      constexpr size_t const_dimBlock = 256;
      const size_t bitmapSzPerBlock = (adj.num_rows() + sizeof(bitmap_type) - 1) / sizeof(bitmap_type);
      const size_t bitmapSz = bitmapSzPerBlock * dimGrid;
      bitmaps_.resize(bitmapSz);
      zero_async(bitmaps_.data(), bitmaps_.size(), 0, cudaStream_t(stream_));
      CUDA_RUNTIME(cudaEventRecord(kernelStart_, cudaStream_t(stream_)));
      block_global_kernel<const_dimBlock>
          <<<dimGrid, const_dimBlock, 0, cudaStream_t(stream_)>>>(count_, adj, bitmaps_.data(), bitmaps_.size());
      CUDA_RUNTIME(cudaEventRecord(kernelStop_, cudaStream_t(stream_)));
    }

    /*
#define CASE(const_dimBlock)                                                                                           \
  case const_dimBlock: {                                                                                               \
    int maxActiveBlocks;                                                                                               \
    CUDA_RUNTIME(cudaOccupancyMaxActiveBlocksPerMultiprocessor(                                                        \
        &maxActiveBlocks, block_shared_kernel<const_dimBlock, Csr, uint32_t>, const_dimBlock, 0));                     \
    cudaDeviceProp props;                                                                                              \
    CUDA_RUNTIME(cudaGetDeviceProperties(&props, dev_));                                                               \
    const int dimGrid = maxActiveBlocks * props.multiProcessorCount;                                                   \
    LOG(debug, "block_shared_kernel({})<<<{}, {}, 0, {}>>>", dev_, dimGrid, const_dimBlock, uintptr_t(stream_));       \
    CUDA_RUNTIME(cudaEventRecord(kernelStart_, cudaStream_t(stream_)));                                                \
    block_shared_kernel<const_dimBlock>                                                                                \
        <<<dimGrid, const_dimBlock, 0, cudaStream_t(stream_)>>>(count_, adj, bitmaps_.data(), bitmaps_.size());        \
    CUDA_RUNTIME(cudaEventRecord(kernelStop_, cudaStream_t(stream_)));                                                 \
    break;                                                                                                             \
  }
  */

    /*
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
      */
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

  double bitmap_time() const {}
};

} // namespace pangolin

#undef PANGOLIN_SYNC_WARP