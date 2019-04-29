#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/topology/topology.hpp"

using namespace pangolin;

TEST_CASE("get_cpus") {
  auto cpus = pangolin::topology::get_cpus();
  REQUIRE(cpus.size() > 0);
}