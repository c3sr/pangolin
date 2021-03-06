/*! A dynamic-selection algorithm for triangle counting

*/

#pragma once

#include <cub/cub.cuh>

#include "count.cuh"
#include "pangolin/algorithm/broadcast.cuh"
#include "pangolin/algorithm/fill.cuh"
#include "pangolin/algorithm/reduction.cuh"
#include "pangolin/algorithm/zero.cuh"
#include "pangolin/dense/device_bit_vector.cuh"
#include "pangolin/dense/device_buffer.cuh"
#include "pangolin/dense/vector.cuh"
#include "search.cuh"

template <size_t BLOCK_DIM_X, typename CsrView>
__global__ void __launch_bounds__(BLOCK_DIM_X) tc_vertex_bitvector_kernel(
    uint64_t *count,          //!< [inout] the count, caller should zero
    const CsrView adj,        //!< [in] the matrix
    const size_t numRows,     //!< [in] the number of rows this kernel will count
    const size_t rowStart,    //!< [in] the starting row this kernel will count
    char *globalBitVecs,      //!< [out] a scratch area for generating row bit vectors
    size_t globalBitVecBytes, //!< [in] the number of bytes in the globalBitVecs
    size_t *rowIdx            //!< [inout] a gpu memory area for work-stealing. caller should set to rowStart
) {

  typedef typename CsrView::index_type Index;
  constexpr size_t GLOBAL_ALIGNMENT = 256;
  static_assert(BLOCK_DIM_X % 32 == 0, "block size should be multiple of 32");
  constexpr size_t SHMEM_KB = 8;

  // shared memory for a fast bit vector
  __shared__ char shMem[SHMEM_KB * 1024];

  // each block gets a portion of the global memory for a bigger, slower bitvector
  size_t blockBitVecSz = (globalBitVecBytes / gridDim.x) / GLOBAL_ALIGNMENT * GLOBAL_ALIGNMENT;
  char *globalBitVecRaw = &globalBitVecs[blockIdx.x * blockBitVecSz];

  uint32_t *slowBitvecFields = reinterpret_cast<pangolin::DeviceBitVector::field_type *>(globalBitVecRaw);
  size_t slowBitvecSz = blockBitVecSz / sizeof(pangolin::DeviceBitVector::field_type);
  uint32_t *fastBitvecFields = reinterpret_cast<pangolin::DeviceBitVector::field_type *>(shMem);
  size_t fastBitvecSz = SHMEM_KB * 1024 / sizeof(pangolin::DeviceBitVector::field_type);

  // per-thread triangle count
  uint64_t threadCount = 0;

  // have each block claim 1 row at a time
  while (true) {
    Index blockRow;

    if (0 == threadIdx.x) {
      blockRow = atomicAdd(rowIdx, 1);
    }

    // broadcast the starting row of the block to all threads
    blockRow = pangolin::block_broadcast2(blockRow, 0 /*root*/);

    // bail out of loop if we are about to operate on too large of a row
    if (blockRow >= rowStart + numRows) {
      break;
    }

    const Index rowStart = adj.rowPtr_[blockRow];
    const Index rowStop = adj.rowPtr_[blockRow + 1];
    const Index rowSz = rowStop - rowStart;

    // if the row is empty, just skip
    if (rowSz == 0) {
      // no triangles
      continue;
    }

    // determine the maximum non-zero column in the row
    Index minCol = adj.colInd_[rowStart];
    Index maxCol = adj.colInd_[rowStop - 1];

    // if the row will fit in either bit vector, use a bitvector
    if (maxCol - minCol < slowBitvecSz * pangolin::DeviceBitVector::BITS_PER_FIELD) {

      uint32_t *fields = nullptr;
      size_t sz = 0;

      if (maxCol - minCol < fastBitvecSz * pangolin::DeviceBitVector::BITS_PER_FIELD) {
        // printf("%d using fast for row %d\n", blockIdx.x, blockRow);
        fields = fastBitvecFields;
        sz = fastBitvecSz;
      } else {
        fields = slowBitvecFields;
        sz = slowBitvecSz;
      }

      // clock_t start, stop;
      pangolin::DeviceBitVector bitvec(fields, sz, minCol);
      bitvec.block_clear_inclusive(minCol, maxCol);

      __syncthreads();

      // build the bit vector
      for (Index colIdx = threadIdx.x + rowStart; colIdx < rowStop; colIdx += BLOCK_DIM_X) {
        Index col = adj.colInd_[colIdx];
        bitvec.atomic_set(col);
      }

      __syncthreads();

      // the block handles each neighbor in turn
      for (Index nbrIdx = rowStart; nbrIdx < rowStop; ++nbrIdx) {
        Index nbr = adj.colInd_[nbrIdx];
        Index nbrStart = adj.rowPtr_[nbr];
        Index nbrStop = adj.rowPtr_[nbr + 1];
        const Index *nbrBegin = &adj.colInd_[nbrStart];
        const Index *nbrEnd = &adj.colInd_[nbrStop];

        for (const Index *p = nbrBegin + threadIdx.x; p < nbrEnd; p += BLOCK_DIM_X) {
          // the vector is offset by minCol and we only zeroed up to maxCol
          Index searchVal = *p;
          bool match = (searchVal >= minCol) && (searchVal <= maxCol) && bitvec.get(searchVal);
          threadCount += match;
        }
      }

    } else { // use another approach
      if (0 == threadIdx.x) {
        printf("AHH Block %d row %lu wont fit, needs %lu\n", blockIdx.x, uint64_t(blockRow), uint64_t(rowSz));
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

class VertexBitvectorTC {
private:
  int dev_;
  cudaStream_t stream_;
  uint64_t *count_;                    //!< the triangle count
  DeviceBuffer<size_t> rowIdx_;        //!< index of the next available row for counting
  DeviceBuffer<char> globalBitVector_; //!< device array for the bit vector

  // whether we own the stream we are using
  bool ownStream_;

  // events for measuring time
  float kernelMillis_;
  cudaEvent_t kernelStart_;
  cudaEvent_t kernelStop_;

public:
  VertexBitvectorTC(int dev)
      : dev_(dev), stream_(nullptr), count_(nullptr), rowIdx_(1, dev), ownStream_(true), kernelMillis_(0) {
    SPDLOG_TRACE(logger::console(), "set dev {}", dev_);
    CUDA_RUNTIME(cudaSetDevice(dev_));
    SPDLOG_TRACE(logger::console(), "create stream");
    CUDA_RUNTIME(cudaStreamCreate(&stream_));
    SPDLOG_TRACE(logger::console(), "mallocManaged");
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting

    // allocate a large space for bit vectors on the device
    size_t fr, total;
    CUDA_RUNTIME(cudaMemGetInfo(&fr, &total));
    (void)total;
    fr = fr * 0.9;
    LOG(debug, "allocating {} MB for global bit vectors", fr / 1024.0 / 1024.0);
    globalBitVector_ = std::move(DeviceBuffer<char>(fr, dev));

    CUDA_RUNTIME(cudaEventCreate(&kernelStart_));
    CUDA_RUNTIME(cudaEventCreate(&kernelStop_));

    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
  }

  VertexBitvectorTC(int dev, cudaStream_t stream)
      : dev_(dev), stream_(stream), count_(nullptr), rowIdx_(1, dev), ownStream_(false), kernelMillis_(0) {
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaMallocManaged(&count_, sizeof(*count_)));
    zero_async<1>(count_, dev_, stream_); // zero on the device that will do the counting

    // allocate a large space for bit vectors on the device
    size_t fr, total;
    CUDA_RUNTIME(cudaMemGetInfo(&fr, &total));
    (void)total;
    fr = fr * 0.9;
    LOG(debug, "allocating {} MB for global bit vectors", fr / 1024.0 / 1024.0);
    globalBitVector_ = std::move(DeviceBuffer<char>(fr, dev));

    CUDA_RUNTIME(cudaEventCreate(&kernelStart_));
    CUDA_RUNTIME(cudaEventCreate(&kernelStop_));

    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
  }

  VertexBitvectorTC(VertexBitvectorTC &&other)
      : dev_(other.dev_), stream_(other.stream_), count_(other.count_), rowIdx_(std::move(other.rowIdx_)),
        globalBitVector_(std::move(other.globalBitVector_)), ownStream_(other.ownStream_),
        kernelStart_(other.kernelStart_), kernelStop_(other.kernelStop_) {
    other.count_ = nullptr;
    other.ownStream_ = false;
    other.stream_ = nullptr;
    other.kernelStart_ = nullptr;
    other.kernelStop_ = nullptr;
  }

  VertexBitvectorTC() : VertexBitvectorTC(0) {}
  ~VertexBitvectorTC() {
    if (ownStream_ && stream_) {
      SPDLOG_TRACE(logger::console(), "destroy stream {}", uintptr_t(stream_));
      CUDA_RUNTIME(cudaStreamDestroy(stream_));
    }
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
    assert(rowIdx_.data());
    CUDA_RUNTIME(cudaSetDevice(dev_));
    CUDA_RUNTIME(cudaStreamSynchronize(stream_));
    zero_async<1>(count_, dev_, stream_);
    CUDA_RUNTIME(cudaGetLastError());
    device_fill(rowIdx_.data(), 1, rowOffset);
    CUDA_RUNTIME(cudaGetLastError());

#define CASE(const_dimBlock)                                                                                           \
  case const_dimBlock: {                                                                                               \
    int maxActiveBlocks;                                                                                               \
    CUDA_RUNTIME(cudaOccupancyMaxActiveBlocksPerMultiprocessor(                                                        \
        &maxActiveBlocks, tc_vertex_bitvector_kernel<const_dimBlock, Csr>, const_dimBlock, 0));                        \
    cudaDeviceProp props;                                                                                              \
    CUDA_RUNTIME(cudaGetDeviceProperties(&props, dev_));                                                               \
    const int dimGrid = maxActiveBlocks * props.multiProcessorCount;                                                   \
    LOG(debug, "device = {}, tc_vertex_bitvector_kernel<<<{}, {}, 0, {}>>>", dev_, dimGrid, const_dimBlock,            \
        uintptr_t(stream_));                                                                                           \
    CUDA_RUNTIME(cudaEventRecord(kernelStart_, stream_));                                                              \
    tc_vertex_bitvector_kernel<const_dimBlock><<<dimGrid, const_dimBlock, 0, stream_>>>(                               \
        count_, adj, numRows, rowOffset, globalBitVector_.data(), globalBitVector_.size(), rowIdx_.data());            \
    CUDA_RUNTIME(cudaEventRecord(kernelStop_, stream_));                                                               \
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

  void sync() { CUDA_RUNTIME(cudaStreamSynchronize(stream_)); }

  uint64_t count() const { return *count_; }
  int device() const { return dev_; }

  double kernel_time() const {
    float millis;
    CUDA_RUNTIME(cudaEventElapsedTime(&millis, kernelStart_, kernelStop_));
    return millis / 1e3;
  }
};

} // namespace pangolin