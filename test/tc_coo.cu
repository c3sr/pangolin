#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/pangolin.hpp"

using namespace pangolin;

TEST_CASE("count complete graph 3") {
  logger::set_level(logger::Level::TRACE);

  using NodeTy = int;

  INFO("create generator");
  generator::Complete<NodeTy> c(3);
  auto keep = [](EdgeTy<NodeTy> e) { return e.first < e.second; };
  INFO("build coo");
  COO<NodeTy> coo = COO<NodeTy>::from_edges(c.begin(), c.end(), keep);

  REQUIRE(coo.nnz() == 3);
  INFO("count triangles");

  GraphDescription descr;
  descr.format_ = GraphFormat::CsrCoo;
  descr.indexSize_ = sizeof(NodeTy);

  uint64_t count = 0;
  triangleCount(&count, &coo, descr);
  REQUIRE(count == 1);
}

// TEST_CASE("count complete graph 4") {
//   INFO("ctor");
//   logger::set_level(logger::Level::TRACE);

//   using NodeTy = int;

//   generator::Complete<NodeTy> c(4);
//   auto keep = [](EdgeTy<NodeTy> e) { return e.first < e.second; };
//   COO<NodeTy> coo = COO<NodeTy>::from_edges(c.begin(), c.end(), keep);

//   REQUIRE(coo.nnz() == 6);
//   REQUIRE(triangleCount(coo) == 4);
// }