#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/dense/device_buffer.cuh"
#include "pangolin/init.hpp"

using namespace pangolin;

TEMPLATE_TEST_CASE("device buffer", "[gpu]", int, size_t) {
  pangolin::init();
  DeviceBuffer<TestType> b;
  REQUIRE(b.size() == 0);
  REQUIRE(b.data() == nullptr);

  SECTION("buffers can be constructed with a size", "[gpu]") {
    DeviceBuffer<TestType> c(1024, 0);
    REQUIRE(c.size() == 1024);
  }

  SECTION("buffers can grow and shrink", "[gpu]") {
    b.resize(10);
    REQUIRE(b.size() == 10);
    REQUIRE(b.data() != nullptr);

    b.resize(1);
    REQUIRE(b.size() == 1);
    REQUIRE(b.data() != nullptr);

    b.resize(0);
    REQUIRE(b.data() == nullptr);
  }

  SECTION("buffers can be moved", "[gpu]") {
    DeviceBuffer<TestType> dst;
    b.resize(10);
    dst = std::move(b);
    REQUIRE(dst.size() == 10);
    REQUIRE(b.size() == 0);
  }

  SECTION("buffers can be move-constructed", "[gpu]") {
    b.resize(10);
    DeviceBuffer<TestType> dst(std::move(b));
    REQUIRE(dst.size() == 10);
    REQUIRE(b.size() == 0);
  }

  SECTION("buffers can be copy-constructed", "[gpu]") {
    b.resize(10);
    DeviceBuffer<TestType> dst(b);
    REQUIRE(dst.size() == 10);
    REQUIRE(b.size() == 10);
  }
}
