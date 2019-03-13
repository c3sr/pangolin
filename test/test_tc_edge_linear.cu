
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/algorithm/tc_edge_linear.cuh"

using namespace pangolin;

TEST_CASE("ctor") {
  LinearTC c;
  REQUIRE(c.count() == 0);
}

TEST_CASE("vector") { std::vector<LinearTC> v; }