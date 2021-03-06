
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/algorithm/tc_edge_binary.cuh"
#include "pangolin/init.hpp"

using namespace pangolin;

TEST_CASE("ctor") {
  pangolin::init();
  BinaryTC c;
  REQUIRE(c.count() == 0);
}

TEST_CASE("vector") {
  pangolin::init();
  std::vector<BinaryTC> v;
}