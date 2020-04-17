
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/algorithm/tc_anjur_iyer.cuh"
#include "pangolin/file/edge_list_file.hpp"
#include "pangolin/filesystem/filesystem.hpp"
#include "pangolin/generator/complete.hpp"
#include "pangolin/generator/hubspoke.hpp"
#include "pangolin/init.hpp"
#include "pangolin/logger.hpp"
#include "pangolin/sparse/csr.hpp"

using namespace pangolin;

typedef AnjurIyer Counter;

template <typename NodeTy> void count(uint64_t expected, const std::string &graphFile, Counter &c) {
  char *graphDir = std::getenv("PANGOLIN_GRAPH_DIR");
  if (nullptr != graphDir) {
    std::string graphDirPath(graphDir);
    graphDirPath += "/" + graphFile;
    if (filesystem::is_file(graphDirPath)) {
      EdgeListFile file(graphDirPath);
      std::vector<DiEdge<NodeTy>> edges;
      std::vector<DiEdge<NodeTy>> fileEdges;
      while (file.get_edges(fileEdges, 10)) {
        edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
      }
      auto upperTriangularFilter = [](DiEdge<NodeTy> e) { return e.src < e.dst; };
      auto csr = CSR<NodeTy>::from_edges(edges.begin(), edges.end(), upperTriangularFilter);

      REQUIRE(expected == c.count_sync(csr.view()));
    }
  }
}

TEST_CASE("ctor", "[gpu]") {
  pangolin::init();
  logger::set_level(logger::Level::DEBUG);
  Counter c;
  REQUIRE(c.count() == 0);

  SECTION("complete(3)", "[gpu]") {
    using NodeTy = int;

    // complete graph with 3 nodes
    generator::Complete<NodeTy> g(3);

    auto keep = [](DiEdge<NodeTy> e) { return e.src < e.dst; };
    auto csr = CSR<NodeTy>::from_edges(g.begin(), g.end(), keep);

    REQUIRE(csr.nnz() == 3);
    REQUIRE(c.count() == 0);
    REQUIRE(1 == c.count_sync(csr.view()));
  }

  SECTION("complete(4)", "[gpu]") {
    using NodeTy = int;

    // complete graph with 4 nodes
    generator::Complete<NodeTy> g(4);

    auto keep = [](DiEdge<NodeTy> e) { return e.src < e.dst; };
    auto csr = CSR<NodeTy>::from_edges(g.begin(), g.end(), keep);

    REQUIRE(csr.nnz() == 6);
    REQUIRE(c.count() == 0);
    REQUIRE(4 == c.count_sync(csr.view()));
  }

  SECTION("hub-spoke 3", "[gpu]") {
    LOG(debug, "hub-spoke 3");
    using NodeTy = int;

    generator::HubSpoke<NodeTy> g(3);

    // highest index node is the hub, so keep those for high out-degree
    auto keep = [](DiEdge<NodeTy> e) { return e.src > e.dst; };
    auto csr = CSR<NodeTy>::from_edges(g.begin(), g.end(), keep);

    REQUIRE(c.count() == 0);
    REQUIRE(2 == c.count_sync(csr.view()));
  }

  SECTION("complete(4) row partition", "[gpu]") {
    using NodeTy = int;

    Counter cs[2];
    REQUIRE(cs[0].count() == 0);
    REQUIRE(cs[1].count() == 0);

    generator::Complete<NodeTy> g(4);

    auto keep = [](DiEdge<NodeTy> e) { return e.src < e.dst; };
    auto csr = CSR<NodeTy>::from_edges(g.begin(), g.end(), keep);
    REQUIRE(csr.nnz() == 6);

    uint64_t a = cs[0].count_sync(csr.view(), 0, 2); // first 2 rows
    uint64_t b = cs[1].count_sync(csr.view(), 2, 2); // next 2 rows
    REQUIRE(4 == a + b);
  }

  SECTION("hub-spoke 539", "[gpu]") {
    LOG(debug, "hub-spoke 539");
    using NodeTy = int;

    generator::HubSpoke<NodeTy> g(539);

    // highest index node is the hub, so keep those for high out-degree
    auto keep = [](DiEdge<NodeTy> e) { return e.src > e.dst; };
    auto csr = CSR<NodeTy>::from_edges(g.begin(), g.end(), keep);

    REQUIRE(c.count() == 0);
    REQUIRE(538 == c.count_sync(csr.view()));
  }

  SECTION("complete(67) lt ", "[gpu]") {
    LOG(debug, "complete 67 lt");
    using NodeTy = int;

    generator::Complete<NodeTy> g(67);

    auto keep = [](DiEdge<NodeTy> e) { return e.src > e.dst; };
    auto csr = CSR<NodeTy>::from_edges(g.begin(), g.end(), keep);

    REQUIRE(c.count() == 0);
    REQUIRE(g.num_triangles() == c.count_sync(csr.view()));
  }

  SECTION("complete(67) ut ", "[gpu]") {
    LOG(debug, "complete 67 ut");
    using NodeTy = int;

    generator::Complete<NodeTy> g(67);

    auto keep = [](DiEdge<NodeTy> e) { return e.src < e.dst; };
    auto csr = CSR<NodeTy>::from_edges(g.begin(), g.end(), keep);

    REQUIRE(c.count() == 0);
    REQUIRE(g.num_triangles() == c.count_sync(csr.view()));
  }

  SECTION("complete(67) row partition ut", "[gpu]") {
    LOG(debug, "complete 67 ut row partition");
    using NodeTy = int;

    Counter cs[2];
    REQUIRE(cs[0].count() == 0);
    REQUIRE(cs[1].count() == 0);

    generator::Complete<NodeTy> g(67);

    auto keep = [](DiEdge<NodeTy> e) { return e.src < e.dst; };
    auto csr = CSR<NodeTy>::from_edges(g.begin(), g.end(), keep);

    uint64_t a = cs[0].count_sync(csr.view(), 0, 34);  // first 270 rows
    uint64_t b = cs[1].count_sync(csr.view(), 34, 33); // next 269 rows
    REQUIRE(g.num_triangles() == a + b);
  }

  SECTION("complete(539) row partition lt", "[gpu]") {
    using NodeTy = int;

    Counter cs[2];
    REQUIRE(cs[0].count() == 0);
    REQUIRE(cs[1].count() == 0);

    generator::Complete<NodeTy> g(539);

    auto keep = [](DiEdge<NodeTy> e) { return e.src > e.dst; };
    auto csr = CSR<NodeTy>::from_edges(g.begin(), g.end(), keep);

    uint64_t a = cs[0].count_sync(csr.view(), 0, 270);   // first 270 rows
    uint64_t b = cs[1].count_sync(csr.view(), 270, 269); // next 269 rows
    REQUIRE(g.num_triangles() == a + b);
  }

  SECTION("amazon0302_adj.bel", "[gpu]") {
    using NodeTy = int;
    count<NodeTy>(717719, "amazon0302_adj.bel", c);
  }

  SECTION("as20000102_adj.bel", "[gpu]") {
    using NodeTy = int;
    count<NodeTy>(6584, "as20000102_adj.bel", c);
  }
}
