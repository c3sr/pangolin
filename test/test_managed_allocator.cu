#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/allocator/cuda_managed.hpp"
#include "pangolin/init.hpp"
#include "pangolin/algorithm/zero.cuh"


using namespace pangolin;

TEST_CASE("vectors can be resized", "[gpu]") {
  pangolin::init();

  std::vector<int, pangolin::allocator::CUDAManaged<int> > v;

  SECTION("vectors can be resized") {
    v.resize(100);
    REQUIRE(v.size() == 100);

    v.resize(10);
    REQUIRE(v.size() == 10);
  }

  SECTION("vectors can be resized and written by the CPU") {
    v.resize(100);
    for (auto &e : v) {
      e = 1;
    }
    REQUIRE(v[50] == 1);
  }

  SECTION("vectors can be resized and zeroed by the GPU") {
    v.resize(100);

    int dev = 0;
    cudaStream_t stream = 0;
    pangolin::zero_async<100>(v.data(), dev, stream);
    CUDA_RUNTIME(cudaDeviceSynchronize());

    for (size_t i = 0; i < 100; ++i) {
      REQUIRE(v[i] == 0);
    }
  }
}

