#pragma once

#include "pangolin/logger.hpp"
#include "pangolin/utilities.hpp"

namespace pangolin {

template <typename T> __global__ void zero(T *ptr, const size_t n) {
  const size_t gx = blockDim.x * blockIdx.x + threadIdx.x;
  for (size_t i = gx; i < n; i += gridDim.x * blockDim.x) {
    ptr[i] = static_cast<T>(0);
  }
}

template <size_t GRID_DIM_X, size_t BLOCK_DIM_X, typename T> __global__ void zero(T *ptr, const size_t n) {
  const size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
  for (size_t i = gx; i < n; i += GRID_DIM_X * BLOCK_DIM_X) {
    ptr[i] = static_cast<T>(0);
  }
}

template <size_t N, typename T> __global__ void zero(T *ptr) {
  const size_t gx = blockDim.x * blockIdx.x + threadIdx.x;
  for (size_t i = gx; i < N; i += gridDim.x * blockDim.x) {
    ptr[i] = static_cast<T>(0);
  }
}

/*! zero array ptr of size N

    \tparam N           the size of the array
    \tparam GRID_DIM_X  the CUDA grid dimension
    \tparam BLOCK_DIM_X the CUDA threadblock dimension
    \tparam T           the type of array element (inferred)

 */
template <size_t N, size_t GRID_DIM_X, size_t BLOCK_DIM_X, typename T>
__global__ void zero(T *ptr //!< [in] pointer to array
) {
  const size_t gx = BLOCK_DIM_X * blockIdx.x + threadIdx.x;
  for (size_t i = gx; i < N; i += GRID_DIM_X * BLOCK_DIM_X) {
    ptr[i] = static_cast<T>(0);
  }
}

/*! zero array ptr of size N

    \tparam T the type of array element
 */
template <typename T> void zero_async(T *ptr, const size_t N, const int dev, cudaStream_t stream) {
  CUDA_RUNTIME(cudaSetDevice(dev));
  constexpr size_t dimGrid = 150;
  constexpr size_t dimBlock = 512;
  LOG(debug, "launch zero: device = {}, blocks = {}, threads = {}", dev, dimGrid, dimBlock);
  zero<dimGrid, dimBlock><<<dimGrid, dimBlock, 0, stream>>>(ptr, N);
  CUDA_RUNTIME(cudaGetLastError());
}

/*! zero array ptr of size N

    \tparam N the size of the array
    \tparam T the type of array element
 */
template <size_t N, typename T> void zero_async(T *ptr, const int dev, cudaStream_t stream) {
  CUDA_RUNTIME(cudaSetDevice(dev));
  constexpr size_t dimBlock = 512;
  constexpr size_t dimGrid = (dimBlock + N - 1) / dimBlock;
  LOG(debug, "launch zero: device = {}, blocks = {}, threads = {} stream = {}", dev, dimGrid, dimBlock,
      uintptr_t(stream));
  zero<N, dimGrid, dimBlock><<<dimGrid, dimBlock, 0, stream>>>(ptr);
  CUDA_RUNTIME(cudaGetLastError());
}

} // namespace pangolin