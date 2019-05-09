
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/algorithm/tc_edge_linear.cuh"
#include "pangolin/generator/hubspoke.hpp"
#include "pangolin/init.hpp"
#include "pangolin/logger.hpp"
#include "pangolin/sparse/coo.hpp"

using namespace pangolin;

TEST_CASE("ctor") {
  pangolin::init();
  LinearTC c;
  REQUIRE(c.count() == 0);
}

TEST_CASE("vector") {
  pangolin::init();
  std::vector<LinearTC> v;
}

TEST_CASE("single counter", "[gpu]") {
  pangolin::init();
  logger::set_level(logger::Level::DEBUG);
  LinearTC c;
  REQUIRE(c.count() == 0);

  SECTION("hub-spoke 3", "[gpu]") {
    using NodeTy = int;

    generator::HubSpoke<NodeTy> g(3);

    // highest index node is the hub, so keep those for high out-degree
    auto keep = [](EdgeTy<NodeTy> e) { return e.first > e.second; };
    auto csrcoo = COO<NodeTy>::from_edges(g.begin(), g.end(), keep);

    REQUIRE(c.count() == 0);
    REQUIRE(2 == c.count_sync(csrcoo.view()));
  }

  SECTION("hub-spoke 539", "[gpu]") {
    using NodeTy = int;

    generator::HubSpoke<NodeTy> g(539);

    // highest index node is the hub, so keep those for high out-degree
    auto keep = [](EdgeTy<NodeTy> e) { return e.first > e.second; };
    auto csrcoo = COO<NodeTy>::from_edges(g.begin(), g.end(), keep);

    REQUIRE(c.count() == 0);
    REQUIRE(538 == c.count_sync(csrcoo.view()));
  }
}