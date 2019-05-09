#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/init.hpp"
#include "pangolin/topology/topology.hpp"

using namespace pangolin;

TEST_CASE("get_cpus") {
  pangolin::init();
  logger::set_level(logger::Level::TRACE);

  auto &topology = topology::topology();

  SECTION("at least 1 cpu") { REQUIRE(topology.cpus_.size() >= 1); }

  SECTION("numa region of address") {
    auto numa = topology.page_numa(nullptr);
  }
}