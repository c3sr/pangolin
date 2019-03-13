
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/algorithm/tc_edge_merge.cuh"

using namespace pangolin;

TEST_CASE("ctor") {
  MergeTC c;
  REQUIRE(c.count() == 0);
}

TEST_CASE("vector") { std::vector<MergeTC> v; }