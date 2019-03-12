#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/algorithm/zero.cuh"
#include "pangolin/logger.hpp"
#include "pangolin/utilities.hpp"

using namespace pangolin;

TEST_CASE("dynamic 10", "[gpu]") {
  int *a = nullptr;
  size_t n = 10;
  const int dev = 0;
  cudaStream_t stream;
  CUDA_RUNTIME(cudaMallocManaged(&a, sizeof(*a) * n));
  for (int i = 0; i < n; ++i) {
    a[i] = i + 1;
  }

  CUDA_RUNTIME(cudaStreamCreate(&stream));

  zero_async(a, n, dev, stream);
  CUDA_RUNTIME(cudaStreamSynchronize(stream));

  for (int i = 0; i < n; ++i) {
    REQUIRE(a[i] == 0);
  }

  CUDA_RUNTIME(cudaFree(a));
  CUDA_RUNTIME(cudaStreamDestroy(stream));
}

TEST_CASE("dconstexpr 10", "[gpu]") {
  int *a = nullptr;
  constexpr size_t n = 10;
  int dev = 0;
  cudaStream_t stream;
  CUDA_RUNTIME(cudaMallocManaged(&a, sizeof(*a) * n));
  for (int i = 0; i < n; ++i) {
    a[i] = i + 1;
  }

  CUDA_RUNTIME(cudaStreamCreate(&stream));

  zero_async<10>(a, dev, stream);
  CUDA_RUNTIME(cudaStreamSynchronize(stream));

  for (int i = 0; i < n; ++i) {
    REQUIRE(a[i] == 0);
  }

  CUDA_RUNTIME(cudaFree(a));
  CUDA_RUNTIME(cudaStreamDestroy(stream));
}
