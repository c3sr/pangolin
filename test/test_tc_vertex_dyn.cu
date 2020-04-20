#pragma GCC diagnostic push "-Wno-unused-local-typedefs"
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/algorithm/tc_vertex_dyn.cuh"
#include "pangolin/file/edge_list_file.hpp"
#include "pangolin/filesystem/filesystem.hpp"
#include "pangolin/generator/hubspoke.hpp"
#include "pangolin/init.hpp"
#include "pangolin/logger.hpp"
#include "pangolin/sparse/csr_binned.hpp"

using namespace pangolin;

#warning "test disabled"
#if 0
template <typename NodeIndex, typename EdgeIndex>
void count(uint64_t expected, const std::string &graphFile, VertexDynTC &c, const uint64_t maxExpectedNode) {
  char *graphDir = std::getenv("PANGOLIN_GRAPH_DIR");
  if (nullptr != graphDir) {
    std::string graphDirPath(graphDir);
    graphDirPath += "/" + graphFile;
    if (filesystem::is_file(graphDirPath)) {
      EdgeListFile file(graphDirPath);
      std::vector<DiEdge<NodeIndex>> edges;
      std::vector<DiEdge<NodeIndex>> fileEdges;
      while (file.get_edges(fileEdges, 100)) {
        edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
      }
      auto upperTriangularFilter = [](DiEdge<NodeIndex> e) { return e.src < e.dst; };
      auto csr = CSRBinned<NodeIndex, EdgeIndex>::from_edges(edges.begin(), edges.end(), maxExpectedNode,
                                                             upperTriangularFilter);

      REQUIRE(expected == c.count_sync(csr.view()));
    }
  } else {
    LOG(warn, "PANGOLIN_GRAPH_DIR undefined, skipping count of {}", graphFile);
  }
}
#endif

TEST_CASE("ctor") {
  pangolin::init();
  VertexDynTC c;
  REQUIRE(c.count() == 0);
}

TEST_CASE("vector") {
  pangolin::init();
  std::vector<VertexDynTC> v;
}

TEST_CASE("move ctor") {
  pangolin::init();
  VertexDynTC c1;
  VertexDynTC c2(std::move(c1));
}

TEST_CASE("single counter", "[gpu]") {
  pangolin::init();
  logger::set_level(logger::Level::DEBUG);
  VertexDynTC c;
  REQUIRE(c.count() == 0);

  SECTION("hub-spoke 3 ut", "[gpu]") {
    typedef uint32_t NodeIndex;
    typedef uint64_t EdgeIndex;
    typedef DiEdge<NodeIndex> Edge;
    typedef CSRBinned<NodeIndex, EdgeIndex> CSR;

    // hub is node with highest index
    generator::HubSpoke<NodeIndex> g(3);
    auto ut = [](Edge e) { return e.src < e.dst; };
    std::vector<Edge> edges;
    for (auto edge : g) {
      if (ut(edge)) {
        edges.push_back(edge);
      }
    }
    CSR csr(4, edges.size());
    for (auto edge : edges) {
      csr.add_next_edge(edge);
    }
    csr.finish_edges();

    REQUIRE(csr.nnz() == 5);
    REQUIRE(csr.num_rows() == 4);

    REQUIRE(c.count() == 0);
    REQUIRE(2 == c.count_sync(csr.view()));
  }

  SECTION("hub-spoke 3 lt", "[gpu]") {
    typedef uint32_t NodeIndex;
    typedef uint64_t EdgeIndex;
    typedef DiEdge<NodeIndex> Edge;
    typedef CSRBinned<NodeIndex, EdgeIndex> CSR;

    // hub is node with highest index
    generator::HubSpoke<NodeIndex> g(3);
    auto keep = [](Edge e) { return e.src > e.dst; };
    std::vector<Edge> edges;
    for (auto edge : g) {
      if (keep(edge)) {
        edges.push_back(edge);
      }
    }
    CSR csr(4, edges.size());
    for (auto edge : edges) {
      csr.add_next_edge(edge);
    }
    csr.finish_edges();

    REQUIRE(csr.nnz() == 5);
    REQUIRE(csr.num_rows() == 4);

    REQUIRE(c.count() == 0);
    REQUIRE(2 == c.count_sync(csr.view()));
  }

  SECTION("hub-spoke 539 lt", "[gpu]") {
    LOG(debug, "starting hub-spoke 539 lt");
    typedef uint32_t NodeIndex;
    typedef uint64_t EdgeIndex;
    typedef DiEdge<NodeIndex> Edge;
    typedef CSRBinned<NodeIndex, EdgeIndex> CSR;

    // highest index node is the hub
    generator::HubSpoke<NodeIndex> g(539);

    auto keep = [](Edge e) { return e.src > e.dst; };
    std::vector<Edge> edges;
    for (auto edge : g) {
      if (keep(edge)) {
        edges.push_back(edge);
      }
    }
    CSR csr(540, edges.size());
    for (auto edge : edges) {
      csr.add_next_edge(edge);
    }
    csr.finish_edges();

    REQUIRE(csr.view().num_rows() == 540);
    REQUIRE(csr.view().nnz() == 1077);
    REQUIRE(c.count() == 0);
    REQUIRE(538 == c.count_sync(csr.view()));
  }

  SECTION("hub-spoke 539 ut", "[gpu]") {
    typedef uint32_t NodeIndex;
    typedef uint64_t EdgeIndex;
    typedef DiEdge<NodeIndex> Edge;
    typedef CSRBinned<NodeIndex, EdgeIndex> CSR;
    
    generator::HubSpoke<NodeIndex> g(539);
    
    // highest index node is the hub, so keep those for high out-degree
    auto keep = [](Edge e) { return e.src < e.dst; };
    std::vector<Edge> edges;
    for (auto edge : g) {
      if (keep(edge)) {
        edges.push_back(edge);
      }
    }
    CSR csr(540, edges.size());
    for (auto edge : edges) {
      csr.add_next_edge(edge);
    }
    csr.finish_edges();
    
    REQUIRE(c.count() == 0);
    REQUIRE(538 == c.count_sync(csr.view()));
  }

  #warning "test disabled"
#if 0
  SECTION("as20000102_adj.bel", "[gpu]") {
    using NodeIndex = uint32_t;
    using EdgeIndex = uint64_t;
    count<NodeIndex, EdgeIndex>(6584, "as20000102_adj.bel", c, 6474);
  }

  SECTION("amazon0302_adj.bel", "[gpu]") {
    using NodeIndex = uint32_t;
    using EdgeIndex = uint64_t;
    count<NodeIndex, EdgeIndex>(717719, "amazon0302_adj.bel", c, 262111);
  }
#endif
}
#pragma GCC diagnostic pop