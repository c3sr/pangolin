
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/algorithm/tc_vertex_blocks_binary.cuh"
#include "pangolin/generator/complete.hpp"
#include "pangolin/init.hpp"
#include "pangolin/logger.hpp"
#include "pangolin/sparse/csr.hpp"

using namespace pangolin;

TEST_CASE("ctor", "[gpu]") {
  pangolin::init();
  logger::set_level(logger::Level::DEBUG);
  VertexBlocksBinaryTC c;
  REQUIRE(c.count() == 0);

  SECTION("complete(3)", "[gpu]") {
    using NodeTy = int;

    // complete graph with 3 nodes
    generator::Complete<NodeTy> g(3);

    auto keep = [](EdgeTy<NodeTy> e) { return e.first < e.second; };
    auto csr = CSR<NodeTy>::from_edges(g.begin(), g.end(), keep);

    REQUIRE(csr.nnz() == 3);
    REQUIRE(c.count() == 0);
    REQUIRE(1 == c.count_sync(csr.view()));
  }

  SECTION("complete(4)", "[gpu]") {
    using NodeTy = int;

    // complete graph with 4 nodes
    generator::Complete<NodeTy> g(4);

    auto keep = [](EdgeTy<NodeTy> e) { return e.first < e.second; };
    auto csr = CSR<NodeTy>::from_edges(g.begin(), g.end(), keep);

    REQUIRE(csr.nnz() == 6);
    REQUIRE(c.count() == 0);
    REQUIRE(4 == c.count_sync(csr.view()));
  }

  SECTION("complete(4) row partition", "[gpu]") {
    using NodeTy = int;

    VertexBlocksBinaryTC cs[2];
    REQUIRE(cs[0].count() == 0);
    REQUIRE(cs[1].count() == 0);

    // complete graph with 4 nodes
    generator::Complete<NodeTy> g(4);

    auto keep = [](EdgeTy<NodeTy> e) { return e.first < e.second; };
    auto csr = CSR<NodeTy>::from_edges(g.begin(), g.end(), keep);
    REQUIRE(csr.nnz() == 6);

    uint64_t a = cs[0].count_sync(csr.view(), 0, 2); // first 2 rows
    uint64_t b = cs[1].count_sync(csr.view(), 2, 2); // next 2 rows
    REQUIRE(4 == a + b);
  }
}
