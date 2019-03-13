
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/algorithm/tc_edge_binary.cuh"

using namespace pangolin;

TEST_CASE("ctor") {
  BinaryTC c;
  REQUIRE(c.count() == 0);
}

TEST_CASE("vector") { std::vector<BinaryTC> v; }