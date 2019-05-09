#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/init.hpp"
#include "pangolin/topology/topology.hpp"

using namespace pangolin;

TEST_CASE("get_cpus") {
  pangolin::init();
  logger::set_level(logger::Level::TRACE);

  auto &topology = topology::topology();

  SECTION("") { REQUIRE(topology.cpus_.size() > 0); }
}