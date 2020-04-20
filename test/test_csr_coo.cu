#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/init.hpp"
#include "pangolin/sparse/csr_coo.hpp"

using namespace pangolin;



TEMPLATE_TEST_CASE("ctor", "[gpu]", int, size_t) {
  pangolin::init();
  CSRCOO<TestType> adj;
}

TEST_CASE("CSRCOO<int>::from_edgelist") {
  typedef int Node;
  typedef DiEdgeList<Node> EdgeList;

  pangolin::init();
  INFO("ctor");

  CSRCOO<Node> coo;

  EdgeList el = {
      {0, 1},
  };

  INFO("from_edgelist");
  coo = CSRCOO<int>::from_edgelist(el);

  INFO("check nnz");
  REQUIRE(coo.nnz() == 1);
  REQUIRE(coo.row_ind()[0] == 0);
  REQUIRE(coo.col_ind()[0] == 1);
}

TEST_CASE("CSRCOO<int>::from_edges upper triangular") {
  typedef int Node;
  typedef DiEdge<Node> Edge;
  typedef DiEdgeList<Node> EdgeList;
  typedef CSRCOO<Node> CSR;

  pangolin::init();
  EdgeList el = {{0, 1}, {0, 2}, {1, 0}, {1, 2}, {1, 3}, {1, 4}, {2, 0}, {2, 1},
                                      {2, 3}, {2, 4}, {3, 1}, {3, 2}, {3, 4}, {4, 1}, {4, 2}, {4, 3}};

  INFO("from_edgelist");
  auto ut = [](Edge e) { return e.src < e.dst; };
  auto coo = CSR::from_edges(el.begin(), el.end(), ut);

  REQUIRE(coo.nnz() == 8);
  REQUIRE(coo.num_rows() == 5);
}

TEST_CASE("CSRCOO<int>::from_edges lower triangular") {
  typedef int Node;
  typedef DiEdge<Node> Edge;
  typedef DiEdgeList<Node> EdgeList;
  typedef CSRCOO<Node> CSR;

  pangolin::init();
  EdgeList el = {{0, 1}, {0, 2}, {1, 0}, {1, 2}, {1, 3}, {1, 4}, {2, 0}, {2, 1},
                                      {2, 3}, {2, 4}, {3, 1}, {3, 2}, {3, 4}, {4, 1}, {4, 2}, {4, 3}};

  INFO("from_edgelist");
  auto lt = [](Edge e) { return e.src > e.dst; };
  auto coo = CSR::from_edges(el.begin(), el.end(), lt);

  REQUIRE(coo.nnz() == 8);
  REQUIRE(coo.num_rows() == 5);
}

TEST_CASE("CSRCOO<int>::num_nodes") {
  typedef int Node;
  typedef DiEdge<Node> Edge;
  typedef DiEdgeList<Node> EdgeList;
  typedef CSRCOO<Node> CSR;

  pangolin::init();
  EdgeList el = {{1, 2}, {1, 3}, {2, 1}, {2, 3}, {2, 4}, {2, 5}, {3, 1}, {3, 2},
                                      {3, 4}, {3, 5}, {4, 2}, {4, 3}, {4, 5}, {5, 2}, {5, 3}, {5, 4}};

  INFO("from_edgelist");
  auto ut = [](Edge e) { return e.src < e.dst; };
  auto coo = CSR::from_edges(el.begin(), el.end(), ut);

  REQUIRE(coo.num_nodes() == 6);
  REQUIRE(coo.num_rows() == 6);
}

TEST_CASE("CSRCOO<int> 2->100 ut") {
  typedef int Node;
  typedef DiEdge<Node> Edge;
  typedef DiEdgeList<Node> EdgeList;
  typedef CSRCOO<Node> CSR;
  pangolin::init();
  EdgeList el = {{2, 100}};

  INFO("from_edgelist");
  auto ut = [](Edge e) { return e.src < e.dst; };
  auto coo = CSR::from_edges(el.begin(), el.end(), ut);

  REQUIRE(coo.num_rows() == 101);
}

// this should be an empty matrix
TEST_CASE("CSRCOO<int> 2->100 lt") {
  typedef int Node;
  typedef DiEdge<Node> Edge;
  typedef DiEdgeList<Node> EdgeList;
  typedef CSRCOO<Node> CSR;
  pangolin::init();
  EdgeList el = {{2, 100}};

  INFO("from_edgelist");
  auto lt = [](Edge e) { return e.src > e.dst; };
  auto coo = CSR::from_edges(el.begin(), el.end(), lt);

  REQUIRE(coo.nnz() == 0);
  REQUIRE(coo.num_rows() == 0);
}

// make sure that the incremental build works with empty final rows
TEMPLATE_TEST_CASE("COO 2->100 ut incremental", "[gpu", uint64_t, uint32_t) {
  pangolin::init();
  std::vector<DiEdge<TestType>> el = {{2, 100}};

  INFO("from_edgelist");
  auto ut = [](DiEdge<TestType> e) { return e.src < e.dst; };
  CSRCOO<TestType> coo;
  TestType maxNode = 0;
  for (const auto e : el) {
    if (ut(e)) {
      coo.add_next_edge(e);
    }
    maxNode = max(e.src, maxNode);
    maxNode = max(e.dst, maxNode);
  }
  coo.finish_edges(maxNode);

  REQUIRE(coo.num_rows() == 101);
}