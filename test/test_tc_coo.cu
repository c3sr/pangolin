#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/algorithm/tc_edge_dynpar.cuh"
#include "pangolin/generator/complete.hpp"
#include "pangolin/init.hpp"
#include "pangolin/logger.hpp"
#include "pangolin/sparse/coo.hpp"

using namespace pangolin;

TEST_CASE("count complete graph 3") {
  pangolin::init();
  logger::set_level(logger::Level::TRACE);
  using NodeTy = int;

  generator::Complete<NodeTy> c(3);
  auto keep = [](EdgeTy<NodeTy> e) { return e.first < e.second; };
  COO<NodeTy> coo = COO<NodeTy>::from_edges(c.begin(), c.end(), keep);

  REQUIRE(coo.nnz() == 3);
  REQUIRE(tc_edge_dynpar(coo) == 1);
}

TEST_CASE("count complete graph 4") {
  pangolin::init();
  INFO("ctor");
  logger::set_level(logger::Level::TRACE);
  using NodeTy = int;

  generator::Complete<NodeTy> c(4);
  auto keep = [](EdgeTy<NodeTy> e) { return e.first < e.second; };
  COO<NodeTy> coo = COO<NodeTy>::from_edges(c.begin(), c.end(), keep);

  REQUIRE(coo.nnz() == 6);
  REQUIRE(tc_edge_dynpar(coo) == 4);
}