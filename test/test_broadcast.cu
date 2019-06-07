#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/algorithm/broadcast.cuh"
#include "pangolin/dense/vector.hu"
#include "pangolin/init.hpp"

using namespace pangolin;

/*!
kernel for broadcast with one block
*/
template <typename T> __global__ void test_broadcast_kernel(T *buf, size_t n) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    buf[i] = pangolin::warp_broadcast<1>(buf[i], 0);
  }
}

TEMPLATE_TEST_CASE("init", "[gpu]", int, uint64_t) {
  pangolin::init();

  Vector<TestType> v;
  TestType VAL1 = 0x8BADF00D; // a 32 or 64-bit value
  TestType VAL2 = 0xDEADBEEF; // a 32 or 64-bit value
  TestType VAL3 = 3;          // a 32 or 64-bit value

  SECTION("full warp", "") {
    v = Vector<TestType>(32, VAL3);
    v[0] = VAL1;

    test_broadcast_kernel<<<1, v.size()>>>(v.data(), v.size());
    CUDA_RUNTIME(cudaDeviceSynchronize());
    for (auto e : v) {
      REQUIRE(e == VAL1);
    }
  }

  SECTION("empty warp", "") {
    v.resize(0);
    test_broadcast_kernel<<<1, v.size()>>>(v.data(), v.size());
    CUDA_RUNTIME(cudaDeviceSynchronize());
    // expect no crash here
  }

  SECTION("half warp", "") {
    v = Vector<TestType>(16, VAL3);
    v[0] = VAL1;
    test_broadcast_kernel<<<1, v.size()>>>(v.data(), v.size());
    CUDA_RUNTIME(cudaDeviceSynchronize());
    for (auto e : v) {
      REQUIRE(e == VAL1);
    }
  }

  SECTION("two warps", "") {
    v = Vector<TestType>(64, VAL3);
    v[0] = VAL1;
    v[32] = VAL2;
    test_broadcast_kernel<<<1, v.size()>>>(v.data(), v.size());
    CUDA_RUNTIME(cudaDeviceSynchronize());
    for (size_t i = 0; i < 32; ++i) {
      REQUIRE(v[i] == VAL1);
    }
    for (size_t i = 32; i < v.size(); ++i) {
      REQUIRE(v[i] == VAL2);
    }
  }

  SECTION("1.5 warps", "") {
    v = Vector<TestType>(48, VAL3);
    v[0] = VAL1;
    v[32] = VAL2;
    test_broadcast_kernel<<<1, v.size()>>>(v.data(), v.size());
    CUDA_RUNTIME(cudaDeviceSynchronize());
    for (size_t i = 0; i < 32; ++i) {
      REQUIRE(v[i] == VAL1);
    }
    for (size_t i = 32; i < v.size(); ++i) {
      REQUIRE(v[i] == VAL2);
    }
  }
}
