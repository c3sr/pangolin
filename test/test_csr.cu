#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/init.hpp"
#include "pangolin/sparse/csr.hpp"
#include "pangolin/edge_list.hpp"

using namespace pangolin;

TEST_CASE("ctor") {
  pangolin::init();
  CSR<int> m;

  SECTION("reserve") {
    m.reserve(10, 100);
    REQUIRE(m.num_rows() == 0);
    REQUIRE(m.nnz() == 0);
  }
}

TEST_CASE("COO<int>::from_edgelist") {
  pangolin::init();
  DiEdgeList<int> el = {
      {0, 1},
  };

  INFO("from_edgelist");
  auto csr = CSR<int>::from_edges(el.begin(), el.end());

  INFO("check nnz");
  REQUIRE(csr.nnz() == 1);
  REQUIRE(csr.row_ptr()[1] - csr.row_ptr()[0] == 1);
  REQUIRE(csr.col_ind()[0] == 1);
}

TEST_CASE("CSR<int>::from_edges upper triangular") {
  pangolin::init();
  DiEdgeList<uint64_t> el = {{0, 1}, {0, 2}, {1, 0}, {1, 2}, {1, 3}, {1, 4}, {2, 0}, {2, 1},
                                      {2, 3}, {2, 4}, {3, 1}, {3, 2}, {3, 4}, {4, 1}, {4, 2}, {4, 3}};

  INFO("from_edgelist");
  auto ut = [](DiEdge<uint64_t> e) { return e.src < e.dst; };
  auto csr = CSR<uint64_t>::from_edges(el.begin(), el.end(), ut);

  REQUIRE(csr.nnz() == 8);
}

TEST_CASE("CSR<int>::from_edges lower triangular") {
  pangolin::init();
  std::vector<DiEdge<uint64_t>> el = {{0, 1}, {0, 2}, {1, 0}, {1, 2}, {1, 3}, {1, 4}, {2, 0}, {2, 1},
                                      {2, 3}, {2, 4}, {3, 1}, {3, 2}, {3, 4}, {4, 1}, {4, 2}, {4, 3}};

  INFO("from_edgelist");
  auto lt = [](DiEdge<uint64_t> e) { return e.src > e.dst; };
  auto csr = CSR<uint64_t>::from_edges(el.begin(), el.end(), lt);

  REQUIRE(csr.nnz() == 8);
}

TEST_CASE("CSR<uint64_t>::view lower triangular") {
  pangolin::init();
  std::vector<DiEdge<uint64_t>> el = {{0, 1}, {0, 2}, {1, 0}, {1, 2}, {1, 3}, {1, 4}, {2, 0}, {2, 1},
                                      {2, 3}, {2, 4}, {3, 1}, {3, 2}, {3, 4}, {4, 1}, {4, 2}, {4, 3}};

  auto lt = [](DiEdge<uint64_t> e) { return e.src > e.dst; };
  auto csr = CSR<uint64_t>::from_edges(el.begin(), el.end(), lt);

  SECTION("nnz") {
    REQUIRE(csr.nnz() == 8);
    REQUIRE(csr.nnz() == csr.view().nnz());
  }

  SECTION("num_nodes") {
    REQUIRE(csr.num_nodes() == 5);
    REQUIRE(csr.num_nodes() == csr.view().num_nodes());
  }

  SECTION("num_rows") { REQUIRE(csr.num_rows() == csr.view().num_rows()); }
}

TEST_CASE("edge 2->100 lt") {
  pangolin::init();
  std::vector<DiEdge<uint64_t>> el = {{2, 100}};

  auto lt = [](DiEdge<uint64_t> e) { return e.src > e.dst; };
  auto csr = CSR<uint64_t>::from_edges(el.begin(), el.end(), lt);

  // this should be an empty matrix

  SECTION("nnz") {
    REQUIRE(csr.nnz() == 0);
    REQUIRE(csr.nnz() == csr.view().nnz());
  }

  SECTION("num_rows") {
    REQUIRE(csr.num_rows() == 0); // 0 - 100
    REQUIRE(csr.num_rows() == csr.view().num_rows());
  }
}

TEST_CASE("edge 2->100 ut") {
  pangolin::init();
  std::vector<DiEdge<uint64_t>> el = {{2, 100}};

  auto lt = [](DiEdge<uint64_t> e) { return e.src < e.dst; };
  auto csr = CSR<uint64_t>::from_edges(el.begin(), el.end(), lt);

  SECTION("nnz") {
    REQUIRE(csr.nnz() == 1);
    REQUIRE(csr.nnz() == csr.view().nnz());
  }

  SECTION("num_rows") {
    REQUIRE(csr.num_rows() == 101); // 0 - 100
    REQUIRE(csr.num_rows() == csr.view().num_rows());
  }
}