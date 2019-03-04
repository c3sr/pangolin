#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/pangolin.cuh"
#include "pangolin/pangolin.hpp"

using namespace pangolin;

template <typename T> __global__ void kernel_sscl(uint64_t *count, T *A, size_t aSz, T *B, size_t bSz) {
  size_t gx = blockDim.x * blockIdx.x + threadIdx.x;
  if (gx == 0) {
    *count = serial_sorted_count_linear(A, &A[aSz], B, &B[bSz]);
  }
}

template <typename T> static uint64_t sscl(std::initializer_list<T> a, std::initializer_list<T> b) {

  uint64_t *count = nullptr;
  CUDA_RUNTIME(cudaMallocManaged(&count, sizeof(*count)));

  // compute the size of the intersection
  Vector<T> A(a);
  Vector<T> B(b);
  kernel_sscl<<<1, 1>>>(count, A.data(), a.size(), B.data(), B.size());
  CUDA_RUNTIME(cudaDeviceSynchronize());

  uint64_t ret = *count;
  CUDA_RUNTIME(cudaFree(count));
  return ret;
}

TEST_CASE("sscl") {
  logger::set_level(logger::Level::TRACE);
  REQUIRE(0 == sscl<int>({}, {}));
  REQUIRE(0 == sscl<int>({}, {1}));
  REQUIRE(1 == sscl<int>({0}, {0}));
  REQUIRE(1 == sscl<int>({0}, {0, 1}));
  REQUIRE(0 == sscl<int>({0}, {1}));
  REQUIRE(0 == sscl<int>({0}, {1, 2}));
  REQUIRE(0 == sscl<int>({0, 2, 4}, {1, 3, 5}));
  REQUIRE(1 == sscl<int>({0, 1, 4}, {1}));
  REQUIRE(2 == sscl<int>({0, 1, 4}, {1, 4}));
  REQUIRE(4 == sscl<int>({0, 1, 2, 3, 4}, {1, 2, 3, 4, 5}));
}

template <size_t BLOCK_DIM_X, typename T>
__global__ void kernel_gscb(uint64_t *count, T *A, size_t aSz, T *B, size_t bSz) {
  grid_sorted_count_binary<BLOCK_DIM_X>(count, A, aSz, B, bSz);
}

template <typename T> static uint64_t gscb(std::initializer_list<T> a, std::initializer_list<T> b) {

  uint64_t *count = nullptr;
  CUDA_RUNTIME(cudaMallocManaged(&count, sizeof(*count)));

  // compute the size of the intersection
  Vector<T> A(a);
  Vector<T> B(b);
  constexpr size_t dimBlock = 32;
  kernel_gscb<dimBlock><<<2, dimBlock>>>(count, A.data(), a.size(), B.data(), B.size());
  CUDA_RUNTIME(cudaDeviceSynchronize());

  uint64_t ret = *count;
  CUDA_RUNTIME(cudaFree(count));
  return ret;
}

TEST_CASE("gscb") {
  logger::set_level(logger::Level::TRACE);
  REQUIRE(0 == gscb<int>({}, {}));
  REQUIRE(0 == gscb<int>({}, {1}));
  REQUIRE(1 == gscb<int>({0}, {0}));
  REQUIRE(1 == gscb<int>({0}, {0, 1}));
  REQUIRE(0 == gscb<int>({0}, {1}));
  REQUIRE(0 == gscb<int>({0}, {1, 2}));
  REQUIRE(0 == gscb<int>({0, 2, 4}, {1, 3, 5}));
  REQUIRE(1 == gscb<int>({0, 1, 4}, {1}));
  REQUIRE(2 == gscb<int>({0, 1, 4}, {1, 4}));
  REQUIRE(4 == gscb<int>({0, 1, 2, 3, 4}, {1, 2, 3, 4, 5}));
}

template <typename T> __global__ void kernel_wscb(uint64_t *count, T *A, size_t aSz, T *B, size_t bSz) {
  if (threadIdx.x < 32 && blockIdx.x == 0) {
    warp_sorted_count_binary<1>(count, A, aSz, B, bSz);
  }
}

template <typename T> static uint64_t wscb(std::initializer_list<T> a, std::initializer_list<T> b) {

  uint64_t *count = nullptr;
  CUDA_RUNTIME(cudaMallocManaged(&count, sizeof(*count)));

  // compute the size of the intersection
  Vector<T> A(a);
  Vector<T> B(b);
  kernel_wscb<<<1, 32>>>(count, A.data(), a.size(), B.data(), B.size());
  CUDA_RUNTIME(cudaDeviceSynchronize());

  uint64_t ret = *count;
  CUDA_RUNTIME(cudaFree(count));
  return ret;
}

TEST_CASE("wscb") {
  logger::set_level(logger::Level::TRACE);
  REQUIRE(0 == wscb<int>({}, {}));
  REQUIRE(0 == wscb<int>({}, {1}));
  REQUIRE(1 == wscb<int>({0}, {0}));
  REQUIRE(1 == wscb<int>({0}, {0, 1}));
  REQUIRE(0 == wscb<int>({0}, {1}));
  REQUIRE(0 == wscb<int>({0}, {1, 2}));
  REQUIRE(0 == wscb<int>({0, 2, 4}, {1, 3, 5}));
  REQUIRE(1 == wscb<int>({0, 1, 4}, {1}));
  REQUIRE(2 == wscb<int>({0, 1, 4}, {1, 4}));
  REQUIRE(4 == wscb<int>({0, 1, 2, 3, 4}, {1, 2, 3, 4, 5}));
}

template <size_t C, size_t BLOCK_DIM_X, typename T>
__global__ void kernel_bscb(uint64_t *count, T *A, size_t aSz, T *B, size_t bSz) {
  if (blockIdx.x == 0) {
    block_sorted_count_binary<C, BLOCK_DIM_X>(count, A, aSz, B, bSz);
  }
}

template <size_t C, typename T> static uint64_t bscb(std::initializer_list<T> a, std::initializer_list<T> b) {

  uint64_t *count = nullptr;
  CUDA_RUNTIME(cudaMallocManaged(&count, sizeof(*count)));

  // compute the size of the intersection
  Vector<T> A(a);
  Vector<T> B(b);
  constexpr size_t dimBlock = 32;
  kernel_bscb<C, dimBlock><<<1, dimBlock>>>(count, A.data(), a.size(), B.data(), B.size());
  CUDA_RUNTIME(cudaDeviceSynchronize());

  uint64_t ret = *count;
  CUDA_RUNTIME(cudaFree(count));
  return ret;
}

TEST_CASE("bscb") {
  logger::set_level(logger::Level::TRACE);
  REQUIRE(0 == bscb<2, int>({}, {}));
  REQUIRE(0 == bscb<2, int>({}, {1}));
  REQUIRE(1 == bscb<2, int>({0}, {0}));
  REQUIRE(1 == bscb<2, int>({0}, {0, 1}));
  REQUIRE(0 == bscb<2, int>({0}, {1}));
  REQUIRE(0 == bscb<2, int>({0}, {1, 2}));
  REQUIRE(0 == bscb<2, int>({0, 2, 4}, {1, 3, 5}));
  REQUIRE(1 == bscb<2, int>({0, 1, 4}, {1}));
  REQUIRE(2 == bscb<2, int>({0, 1, 4}, {1, 4}));
  REQUIRE(4 == bscb<2, int>({0, 1, 2, 3, 4}, {1, 2, 3, 4, 5}));
  REQUIRE(4 == bscb<1, int>({0, 1, 2, 3, 4}, {1, 2, 3, 4, 5}));
  REQUIRE(4 == bscb<3, int>({0, 1, 2, 3, 4}, {1, 2, 3, 4, 5}));
}