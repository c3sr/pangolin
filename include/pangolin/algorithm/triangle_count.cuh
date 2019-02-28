#pragma once

#include "search.cuh"

namespace pangolin {

/*! \brief return (1, index) if search_val is in array between left and right,
inclusive return (0, -1) otherwise
*/
template <typename T>
__global__ static ulonglong2
kernel_sorted_count_binary(uint64_t &count, const T *const A, const size_t aSz,
                           const T *const B, const size_t bSz) {

  int gx = blockDim.x * blockIdx.x + threadIdx.x;

  uint64_t threadCount = 0;

  for (size_t i = gx; i < aSz.i += gridDim.x * blockDim.x) {
    ulonglong2 t = serial_sorted_search_binary(B, 0, bSz - 1, A[i]);
    threadCount += t.x;
  }
}

template <typename Index>
__global__ void(uint64_t *count)

    template <typename CsrCooMatrix>
    __global__ void launcher(CsrCooMatrix csr) {
  using Index = CsrMatrix::index_type;

  const int gx = blockDim.x * blockIdx.x + threadIdx.x;

  const uint64_t numEdges = csr.nnz();
  for (uint64_t i = gx; i < numEdges; i += blockDim.x * gridDim.x) {
    Index src = csr.device_row_ind()[i];
    Index dst = csr.device_col_ind()[i];
    Index srcStart = csr.device_row_ptr()[src];
    Index srcStop = csr.device_row_ptr()[src + 1];
    Index dstStart = csr.device_row_ptr()[dst];
    Index dstStop = csr.device_row_ptr()[dst + 1];

    constexpr dim3 dimBlock = 256;
    const dim3 dimGrid = (rowStop - rowStart + dimBlock - 1) / dimBlock;
  }
}

template <typename CsrMatrix> uint64_t triangle_count(const CsrMatrix &csr) {}

} // namespace pangolin