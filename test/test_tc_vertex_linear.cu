
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/algorithm/tc_vertex_linear.cuh"
#include "pangolin/generator/complete.hpp"
#include "pangolin/logger.hpp"
#include "pangolin/sparse/csr.hpp"

using namespace pangolin;

TEST_CASE("ctor", "[gpu]") {
  VertexLinearTC c;
  REQUIRE(c.count() == 0);
}

TEST_CASE("complete(3)", "[gpu]") {
  logger::set_level(logger::Level::TRACE);
  using NodeTy = int;

  // complete graph with 3 nodes
  generator::Complete<NodeTy> g(3);

  auto keep = [](EdgeTy<NodeTy> e) { return e.first < e.second; };
  auto csr = CSR<NodeTy>::from_edges(g.begin(), g.end(), keep);

  REQUIRE(csr.nnz() == 3);

  VertexLinearTC tc;
  REQUIRE(tc.count() == 0);
  REQUIRE(1 == tc.count_sync(csr.view()));
}

TEST_CASE("complete(4)", "[gpu]") {
  logger::set_level(logger::Level::TRACE);
  using NodeTy = int;

  // complete graph with 4 nodes
  generator::Complete<NodeTy> g(4);

  auto keep = [](EdgeTy<NodeTy> e) { return e.first < e.second; };
  auto csr = CSR<NodeTy>::from_edges(g.begin(), g.end(), keep);

  REQUIRE(csr.nnz() == 6);

  VertexLinearTC tc;
  REQUIRE(tc.count() == 0);
  REQUIRE(4 == tc.count_sync(csr.view()));
}