#pragma once

#include "pangolin/logger.hpp"
#include "pangolin/utilities.hpp"

namespace pangolin {

/*! a[i] = a[i] * x + y
 */
template <size_t GRID_DIM_X, size_t BLOCK_DIM_X, typename T> __device__ void grid_apy(T *a, const T y, const size_t n) {
  for (size_t i = BLOCK_DIM_X * blockIdx.x + threadIdx.x; i < n; i += GRID_DIM_X * BLOCK_DIM_X) {
    a[i] += y;
  }
}

/*! a[i] = a[i] * x + y
 */
template <size_t GRID_DIM_X, size_t BLOCK_DIM_X, typename T>
__global__ void apy_async_kernel(T *a, const T y, const size_t n) {
  grid_apy<GRID_DIM_X, BLOCK_DIM_X>(a, y, n);
}

/*! a[i] = a[i] * x + y
 */
template <typename T> void device_axpy_async(T *a, const T x, const T y, const size_t n, cudaStream_t stream) {
  if (1 == x) {
    constexpr size_t dimGrid = 150;
    constexpr size_t dimBlock = 512;
    LOG(debug, "launch apy_async_kernel<<<{}, {}, 0, {}>>>", dimGrid, dimBlock, uintptr_t(stream));
    apy_async_kernel<dimGrid, dimBlock><<<dimGrid, dimBlock, 0, stream>>>(a, y, n);
    CUDA_RUNTIME(cudaGetLastError());
  } else {
    LOG(error, "unimplemented!");
    exit(-1);
  }

} // namespace pangolin

} // namespace pangolin