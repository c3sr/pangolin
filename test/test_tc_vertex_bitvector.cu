#pragma GCC diagnostic push "-Wno-unused-local-typedefs"
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/algorithm/tc_vertex_bitvector.cuh"
#include "pangolin/file/edge_list_file.hpp"
#include "pangolin/filesystem/filesystem.hpp"
#include "pangolin/generator/hubspoke.hpp"
#include "pangolin/init.hpp"
#include "pangolin/logger.hpp"
#include "pangolin/sparse/csr.hpp"

using namespace pangolin;

template <typename NodeTy> void count(uint64_t expected, const std::string &graphFile, VertexBitvectorTC &c) {
  char *graphDir = std::getenv("PANGOLIN_GRAPH_DIR");
  if (nullptr != graphDir) {
    std::string graphDirPath(graphDir);
    graphDirPath += "/" + graphFile;
    if (filesystem::is_file(graphDirPath)) {
      EdgeListFile file(graphDirPath);
      std::vector<DiEdge<NodeTy>> edges;
      std::vector<DiEdge<NodeTy>> fileEdges;
      while (file.get_edges(fileEdges, 100)) {
        edges.insert(edges.end(), fileEdges.begin(), fileEdges.end());
      }
      auto upperTriangularFilter = [](DiEdge<NodeTy> e) { return e.src < e.dst; };
      auto csr = CSR<NodeTy>::from_edges(edges.begin(), edges.end(), upperTriangularFilter);

      REQUIRE(expected == c.count_sync(csr.view()));
    }
  } else {
    LOG(debug, "PANGOLIN_GRAPH_DIR undefined, skipping count");
  }
}

TEST_CASE("ctor") {
  pangolin::init();
  VertexBitvectorTC c;
  REQUIRE(c.count() == 0);
}

TEST_CASE("vector") {
  pangolin::init();
  std::vector<VertexBitvectorTC> v;
}

TEST_CASE("single counter", "[gpu]") {
  pangolin::init();
  logger::set_level(logger::Level::DEBUG);
  VertexBitvectorTC c;
  REQUIRE(c.count() == 0);
  
  SECTION("hub-spoke 3", "[gpu]") {
    using NodeTy = int;
    
    // highest index node is the hub, so keep those for high out-degree
    generator::HubSpoke<NodeTy> g(3);
    auto keep = [](DiEdge<NodeTy> e) { return e.src > e.dst; };
    auto csr = CSR<NodeTy>::from_edges(g.begin(), g.end(), keep);
    
    REQUIRE(csr.nnz() == 5);
    REQUIRE(csr.num_rows() == 4);
    
    REQUIRE(c.count() == 0);
    REQUIRE(2 == c.count_sync(csr.view()));
  }
  
  SECTION("hub-spoke 539", "[gpu]") {
    using NodeTy = int;
    
    generator::HubSpoke<NodeTy> g(539);
    
    // highest index node is the hub, so keep those for high out-degree
    auto keep = [](DiEdge<NodeTy> e) { return e.src > e.dst; };
    auto csr = CSR<NodeTy>::from_edges(g.begin(), g.end(), keep);
    
    REQUIRE(c.count() == 0);
    REQUIRE(538 == c.count_sync(csr.view()));
  }
  
  SECTION("as20000102_adj.bel", "[gpu]") {
    using NodeTy = int;
    count<NodeTy>(6584, "as20000102_adj.bel", c);
  }
  
  SECTION("amazon0302_adj.bel", "[gpu]") {
    using NodeTy = int;
    count<NodeTy>(717719, "amazon0302_adj.bel", c);
  }
}
#pragma GCC diagnostic pop