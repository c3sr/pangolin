
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/algorithm/tc_edge_dysel.cuh"
#include "pangolin/file/edge_list_file.hpp"
#include "pangolin/filesystem/filesystem.hpp"
#include "pangolin/generator/complete.hpp"
#include "pangolin/generator/hubspoke.hpp"
#include "pangolin/init.hpp"
#include "pangolin/logger.hpp"
#include "pangolin/sparse/csr_coo.hpp"

using namespace pangolin;

typedef EdgeWarpDyselTC Counter;

template <typename Node> void count(uint64_t expected, const std::string &graphFile, Counter &c) {
  char *graphDir = std::getenv("PANGOLIN_GRAPH_DIR");
  if (nullptr != graphDir) {
    std::string graphDirPath(graphDir);
    graphDirPath += "/" + graphFile;
    if (filesystem::is_file(graphDirPath)) {
      EdgeListFile file(graphDirPath);
      std::vector<DiEdge<Node>> edges;
      std::vector<DiEdge<Node>> fileEdges;
      while (file.get_edges(fileEdges, 10)) {
        edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
      }
      auto upperTriangularFilter = [](DiEdge<Node> e) { return e.src < e.dst; };
      auto csrcoo = CSRCOO<Node>::from_edges(edges.begin(), edges.end(), upperTriangularFilter);

      REQUIRE(expected == c.count_sync(csrcoo.view()));
    }
  }
}

TEST_CASE("ctor") {
  pangolin::init();
  Counter c;
  REQUIRE(c.count() == 0);
}

TEST_CASE("vector") {
  pangolin::init();
  std::vector<EdgeWarpDyselTC> v;
}

TEST_CASE("single counter", "[gpu]") {
  pangolin::init();
  logger::set_level(logger::Level::DEBUG);
  EdgeWarpDyselTC c;
  REQUIRE(c.count() == 0);

  SECTION("hub-spoke 3", "[gpu]") {
    using NodeTy = int;

    generator::HubSpoke<NodeTy> g(3);

    // highest index node is the hub, so keep those for high out-degree
    auto keep = [](DiEdge<NodeTy> e) { return e.src > e.dst; };
    auto csrcoo = CSRCOO<NodeTy>::from_edges(g.begin(), g.end(), keep);

    REQUIRE(c.count() == 0);
    REQUIRE(2 == c.count_sync(csrcoo.view()));
  }
  SECTION("hub-spoke 539", "[gpu]") {
    using NodeTy = int;

    generator::HubSpoke<NodeTy> g(539);

    // highest index node is the hub, so keep those for high out-degree
    auto keep = [](DiEdge<NodeTy> e) { return e.src > e.dst; };
    auto csrcoo = CSRCOO<NodeTy>::from_edges(g.begin(), g.end(), keep);

    REQUIRE(c.count() == 0);
    REQUIRE(538 == c.count_sync(csrcoo.view()));
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

TEST_CASE("two counters", "[gpu]") {
  pangolin::init();
  logger::set_level(logger::Level::DEBUG);
  EdgeWarpDyselTC cs[2];
  REQUIRE(cs[0].count() == 0);
  REQUIRE(cs[1].count() == 0);

  SECTION("complete(539) row partition ut", "[gpu]") {
    using NodeTy = int;

    generator::Complete<NodeTy> g(539);

    auto keep = [](DiEdge<NodeTy> e) { return e.src < e.dst; };
    auto csrcoo = CSRCOO<NodeTy>::from_edges(g.begin(), g.end(), keep);

    uint64_t a = cs[0].count_sync(csrcoo.view(), 0, csrcoo.nnz()/2);   // first half of edges
    uint64_t b = cs[1].count_sync(csrcoo.view(), csrcoo.nnz()/2, csrcoo.nnz() - csrcoo.nnz()/2); // last half of edges
    REQUIRE(g.num_triangles() == a + b);
  }
}