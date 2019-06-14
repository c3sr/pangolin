#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/dense/vector.cuh"
#include "pangolin/init.hpp"
#include "pangolin/logger.hpp"
#include "pangolin/map/map.cuh"

using namespace pangolin;

__global__ void kernel_for(int *a, const size_t n) {
  WarpRange r(0, n);
  for (auto i = r.begin(); i != r.end(); ++i) {
    if (0 == i->lane_idx()) {
      a[i->idx()] = 0;
    }
  }
}

__global__ void kernel_range_for(int *a, const size_t n) {
  for (auto i : WarpRange(0, n)) {
    if (0 == i.lane_idx()) {
      a[i.idx()] = 0;
    }
  }
}

TEST_CASE("zero (1 warp)", "[gpu]") {
  pangolin::init();
  logger::set_level(logger::Level::TRACE);
  constexpr int dimGrid = 1;
  constexpr int dimBlock = 1;
  constexpr size_t n = 100;

  Vector<int> a(n);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = i;
  }

  kernel_range_for<<<dimGrid, dimBlock>>>(a.data(), a.size());
  CUDA_RUNTIME(cudaGetLastError());
  CUDA_RUNTIME(cudaDeviceSynchronize());

  for (size_t i = 0; i < a.size(); ++i) {
    REQUIRE(a[i] == 0);
  }
}

TEST_CASE("zero (2 warp)", "[gpu]") {
  pangolin::init();
  logger::set_level(logger::Level::TRACE);
  constexpr int dimGrid = 1;
  constexpr int dimBlock = 64;
  constexpr size_t n = 100;

  Vector<int> a(n);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = i;
  }

  kernel_range_for<<<dimGrid, dimBlock>>>(a.data(), a.size());
  CUDA_RUNTIME(cudaGetLastError());
  CUDA_RUNTIME(cudaDeviceSynchronize());

  for (size_t i = 0; i < a.size(); ++i) {
    REQUIRE(a[i] == 0);
  }
}

TEST_CASE("range_for") {
  pangolin::init();
  pangolin::init();
  logger::set_level(logger::Level::TRACE);
  constexpr int dimGrid = 1;
  constexpr int dimBlock = 1;
  constexpr size_t n = 100;

  Vector<int> a(n);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = i;
  }

  kernel_for<<<dimGrid, dimBlock>>>(a.data(), a.size());
  CUDA_RUNTIME(cudaGetLastError());
  CUDA_RUNTIME(cudaDeviceSynchronize());

  for (size_t i = 0; i < a.size(); ++i) {
    REQUIRE(a[i] == 0);
  }
}