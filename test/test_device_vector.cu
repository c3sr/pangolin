#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/dense/device_vector.cuh"
#include "pangolin/init.hpp"

using namespace pangolin;

TEMPLATE_TEST_CASE("ctor from std::vector", "[gpu]", int, size_t) {
  pangolin::init();
  std::vector<TestType> hostVec(10, 17);
  DeviceVector<TestType> deviceVec = hostVec;
  REQUIRE(deviceVec.size() == 10);
  deviceVec.sync();

  SECTION("convert to std::vector") {
    std::vector<TestType> a(deviceVec);
    REQUIRE(a.size() == deviceVec.size());
    deviceVec.sync();
    REQUIRE(a[9] == 17);
  }
}

