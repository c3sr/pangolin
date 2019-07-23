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
#include "pangolin/dense/buffer.cuh"
#include "pangolin/dense/vector.cuh"
#include "search.cuh"

template <typename T>
__device__ void atomic_set(T *bitmap, size_t c) {
    size_t field = c / (sizeof(T)*CHAR_BIT);
    size_t bit = c % (sizeof(T)*CHAR_BIT);
    T bits = T(1) << bit;
    atomicOr(&bitmap[field], bits);
}

//bitmap get function: return whether if bit is 1 or  0 return a number
template <typename T>
__device__ size_t get_bitmap(T *bitmap, size_t k) {
    size_t fieldIdx = k / (sizeof(T)*CHAR_BIT);    
    size_t bitIdx = k % (sizeof(T)*CHAR_BIT);
    T bits = bitmap[fieldIdx];
    return (bits >> bitIdx) & T(1);
}

//bitmap reset function --cleanup
template <typename T>
__device__ void reset_bitmap(T *bitmap, size_t c) {
    size_t fieldIdx = c / (sizeof(T)*CHAR_BIT);
    bitmap[fieldIdx] = 0;
}

//zero bitmap before launching it on host
//hollywood 2009 data set
template <size_t BLOCK_DIM_X, typename CsrView>
__global__ void __launch_bounds__(BLOCK_DIM_X)
    dyn_blocks_kernel(uint64_t *__restrict__ count, //!< [inout] the count(the number of triangles found by each block), caller should zero 
                      const CsrView adj,            //!< [in] the matrix
                      const size_t rowStart,        //!< [in] the starting row this kernel will count a.k.a roff[]
                      const size_t numRows,         //!< [in] the number of rows this kernel will count a.k.a rows[]
		      uint32_t *bitmaps,            //!< [in] array of bitmaps, one per thread block
                      const size_t bitmapSz           
    ) {

  constexpr size_t REG_CACHE_SIZE = 10;

  typedef typename CsrView::index_type Index;
  static_assert(BLOCK_DIM_X % 32 == 0, "block size should be multiple of 32");

  // each thread can cache short rows
  Index rowCache[REG_CACHE_SIZE];

  // per-thread triangle count (is this the tricnt?) (nope)
  uint64_t threadCount = 0;
  
  for(int bi = blockIdx.x; bi < adj.num_rows(); bi += gridDim.x) { //have to do sizeof the array which needs to be added
    const Index io_s = adj.rowPtr_[bi];
    const Index io_e = adj.rowPtr_[bi + 1];
    if(bi == 539 && threadIdx.x == 0) {
      printf("io_s: %d io_e:  %d\n" , io_s, io_e);
      
    } 
    if (threadIdx.x == 0) {
//       printf(" blockID: %d  bi: %d\n", blockIdx.x, bi); 
    }

    Index blk_bound = (io_e + BLOCK_DIM_X - 1) / BLOCK_DIM_X * BLOCK_DIM_X;
 
    for(Index io = io_s; io < io_e; io += blockDim.x) { //find a multiple of blockDim.x that is larger than io_e and use that so io < xyz
      const int64_t c = (io + threadIdx.x < io_e) ? adj.colInd_[io + threadIdx.x]: -1;
      if (c > -1) {
        if (bi == 539 ) {
        printf("tid %d c = %ld\n", threadIdx.x, c);
        }
        atomic_set(&bitmaps[blockIdx.x * bitmapSz], c);
      }
    }
    __syncthreads();

    for(Index io = io_s; io < blk_bound; io += blockDim.x) { 
       const int64_t c = (io + threadIdx.x < io_e) ? adj.colInd_[io + threadIdx.x]: -1;
      
       for (Index t = 0; t < blockDim.x; t++) {
         const int64_t j = pangolin::block_broadcast(c,t);
         if (j == -1) {
            break;
         }
         const Index jo_s = adj.rowPtr_[j];
         const Index jo_e = adj.rowPtr_[j+1];
         if (bi == 539 && threadIdx.x == 0) {
            printf(" j %lld jo_s: %d jo_e: %d \n", j, jo_s , jo_e); 
         }
         for (Index jo = jo_s + threadIdx.x; jo < jo_e; jo += blockDim.x) {
	    const int64_t k = adj.colInd_[jo];
            if (get_bitmap(&bitmaps[blockIdx.x * bitmapSz], k) == 1) {
              threadCount++;
 	    } else {
              if (bi == 539) {
              printf(" k: %d  j: %lld  \n" , k , j);
            }
         }

       }
    }
    } //need to figure out the braket situation
     __syncthreads();

     for(Index io = io_s; io < blk_bound; io += blockDim.x) {
       const int64_t c = (io + threadIdx.x < io_e) ? adj.colInd_[io + threadIdx.x]: -1;
       if (c != -1) {    
          reset_bitmap(&bitmaps[blockIdx.x * bitmapSz], c); //multiplied by bitmapSz here
       }
     }
     __syncthreads();
  } //acconted for
 
  // Block-wide reduction of threadCount
  typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);

  // Add to total count
  
  if (0 == threadIdx.x) {
   // printf("%p", count);
    atomicAdd(count, aggregate);
  }
}

namespace pangolin {

class BissonFaticaTC {
private:
  int dev_;
  RcStream stream_;
  uint64_t *count_; //<! the triangle count (if this is the triangle count where can I use it up there)

  // events for measuring time
  cudaEvent_t kernelStart_;
  cudaEvent_t kernelStop_;
  float time;
  DeviceBuffer<uint32_t> bitmaps_;

public:
  BissonFaticaTC(int dev) : dev_(dev), stream_(dev), count_(nullptr) {
    SPDLOG_TRACE(logger::console(), "set dev {}", dev_);
    CUDA_RUNTIME(cudaSetDevice(dev_));
    SPDLOG_TRACE(logger::console(), "mallocManaged");
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    zero_async<1>(count_, dev_, cudaStream_t(stream_)); // zero on the device that will do the counting

    CUDA_RUNTIME(cudaEventCreate(&kernelStart_));
    CUDA_RUNTIME(cudaEventCreate(&kernelStop_));
  }

  BissonFaticaTC(int dev, cudaStream_t stream) : dev_(dev), stream_(dev, stream), count_(nullptr) {
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    zero_async<1>(count_, dev_, cudaStream_t(stream_)); // zero on the device that will do the counting

    CUDA_RUNTIME(cudaEventCreate(&kernelStart_));
    CUDA_RUNTIME(cudaEventCreate(&kernelStop_));
  }

  BissonFaticaTC(BissonFaticaTC &&other)
      : dev_(other.dev_), stream_(other.stream_), count_(other.count_), kernelStart_(other.kernelStart_),
        kernelStop_(other.kernelStop_) {
    other.count_ = nullptr;
    other.kernelStart_ = nullptr;
    other.kernelStop_ = nullptr;
  }

  BissonFaticaTC() : BissonFaticaTC(0) {}
  ~BissonFaticaTC() {
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
    bitmaps_.resize(1 << 30);
    zero_async(bitmaps_.data(), bitmaps_.size(), 0, cudaStream_t(stream_));

#define CASE(const_dimBlock)                                                                                           \
  case const_dimBlock: {                                                                                               \
    int maxActiveBlocks;                                                                                               \
    CUDA_RUNTIME(cudaOccupancyMaxActiveBlocksPerMultiprocessor(                                                        \
        &maxActiveBlocks, dyn_blocks_kernel<const_dimBlock, Csr>, const_dimBlock, 0));                                 \
    cudaDeviceProp props;                                                                                              \
    CUDA_RUNTIME(cudaGetDeviceProperties(&props, dev_));                                                               \
    const int dimGrid = maxActiveBlocks * props.multiProcessorCount;                                                   \
    LOG(debug, "dyn_blocks_kernel({})<<<{}, {}, 0, {}>>>", dev_, dimGrid, const_dimBlock, stream_);                    \
    CUDA_RUNTIME(cudaEventRecord(kernelStart_, cudaStream_t(stream_)));                                                \
    dyn_blocks_kernel<const_dimBlock>                                                                                  \
        <<<dimGrid, const_dimBlock, 0, cudaStream_t(stream_)>>>(count_, adj, rowOffset, numRows, bitmaps_.data(), bitmaps_.size()/dimGrid );  \
    CUDA_RUNTIME(cudaEventRecord(kernelStop_, cudaStream_t(stream_)));                                                 \
   CUDA_RUNTIME(cudaEventSynchronize(kernelStop_));                                                                    \
   CUDA_RUNTIME(cudaEventElapsedTime(&time, kernelStart_, kernelStop_));                                               \
   printf("Generation Time: %3.1f ms \n" , time);                                                                     \
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
