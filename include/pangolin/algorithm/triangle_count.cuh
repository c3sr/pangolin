#pragma once

#include <cub/cub.cuh>

#include "pangolin/atomic_add.cuh"
#include "pangolin/dense/vector.hu"
#include "search.cuh"

namespace pangolin {

/*! \brief count of elements in A that appear in B

    @param[inout] count pointer to the count. caller should initialize to 0.
    \param        A     the array of needles
    \param        aSz   the number of elements in A
    \param        B     the haystack
    \param        bSz   the number of elements in B
*/
template <size_t BLOCK_DIM_X, typename T>
__global__ void kernel_sorted_count_binary(uint64_t *count, const T *const A, const size_t aSz, const T *const B,
                                           const size_t bSz) {

  // Specialize BlockReduce for a 1D block of 128 threads on type int
  typedef cub::BlockReduce<uint64_t, BLOCK_DIM_X> BlockReduce;
  // Allocate shared memory for BlockReduce
  __shared__ typename BlockReduce::TempStorage tempStorage;

  int gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;

  uint64_t threadCount = 0;

  for (size_t i = gx; i < aSz; i += gridDim.x * BLOCK_DIM_X) {
    // printf("looking for %d from 0 to %lu\n", A[i], bSz);
    ulonglong2 t = serial_sorted_search_binary_exclusive(B, 0, bSz, A[i]);
    threadCount += t.x;
  }

  // aggregate all counts found by this block
  uint64_t aggregate = BlockReduce(tempStorage).Sum(threadCount);

  // Add to total count
  if (0 == threadIdx.x) {
    // printf("found %lu\n", aggregate);
    atomicAdd(count, aggregate);
  }
}

template <typename CsrCooMatrix> __global__ void launcher(uint64_t *tempStorage, CsrCooMatrix csr) {
  using Index = typename CsrCooMatrix::index_type;

  const int gx = blockDim.x * blockIdx.x + threadIdx.x;

  const uint64_t numEdges = csr.nnz();
  for (uint64_t i = gx; i < numEdges; i += blockDim.x * gridDim.x) {
    // printf("working on edge %lu\n", i);
    Index src = csr.device_row_ind()[i];
    Index dst = csr.device_col_ind()[i];
    Index srcStart = csr.device_row_ptr()[src];
    Index srcStop = csr.device_row_ptr()[src + 1];
    Index dstStart = csr.device_row_ptr()[dst];
    Index dstStop = csr.device_row_ptr()[dst + 1];
    // printf("%d [%d-%d] -> %d [%d-%d]\n", src, srcStart, srcStop, dst, dstStart, dstStop);

    const Index *srcBegin = &csr.device_col_ind()[srcStart];
    const size_t srcSz = srcStop - srcStart;
    const Index *dstBegin = &csr.device_col_ind()[dstStart];
    const size_t dstSz = dstStop - dstStart;

    constexpr size_t dimBlock = 256;
    uint64_t *edgeCount = &tempStorage[i];
    if (srcSz > dstSz) { // src has more neighbors than dst, search for srcs in parallel
      const size_t dimGrid = (srcSz + dimBlock - 1) / dimBlock;
      kernel_sorted_count_binary<dimBlock><<<dimGrid, dimBlock>>>(edgeCount, srcBegin, srcSz, dstBegin, dstSz);
    } else {
      const size_t dimGrid = (dstSz + dimBlock - 1) / dimBlock;
      kernel_sorted_count_binary<dimBlock><<<dimGrid, dimBlock>>>(edgeCount, dstBegin, dstSz, srcBegin, srcSz);
    }
  }
}

template <typename CsrCooMatrix> uint64_t triangle_count(const CsrCooMatrix &csr) {

  // one thread per non-zero (edge)
  constexpr size_t dimBlock = 512;
  const dim3 dimGrid = (csr.nnz() + dimBlock - 1) / dimBlock;
  // constexpr size_t dimBlock = 1;
  // const size_t dimGrid = 1;

  // allocate space for the total count
  uint64_t *count = nullptr;
  CUDA_RUNTIME(cudaMallocManaged(&count, sizeof(*count)));
  *count = 0;

  // allocate temporary storage of one count per edge
  Vector<uint64_t> tempStorage(csr.nnz());

  // count triangles on the device
  launcher<<<dimGrid, dimBlock>>>(tempStorage.data(), csr.view());
  CUDA_RUNTIME(cudaDeviceSynchronize());

  // reduction
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, tempStorage.data(), count, csr.nnz());
  CUDA_RUNTIME(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, tempStorage.data(), count, csr.nnz());
  CUDA_RUNTIME(cudaDeviceSynchronize());

  CUDA_RUNTIME(cudaFree(d_temp_storage));
  uint64_t ret = *count;
  CUDA_RUNTIME(cudaFree(count));

  return ret;
}

} // namespace pangolin