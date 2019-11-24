#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/init.hpp"
#include "pangolin/sparse/csr_val.hpp"

using namespace pangolin;

TEST_CASE("ctor") {
  pangolin::init();
  CSR<int, uint64_t, float> m;

  SECTION("reserve") {
    m.reserve(10, 100);
    REQUIRE(m.num_rows() == 0);
    REQUIRE(m.nnz() == 0);
  }
}

TEST_CASE("from_edgelist") {

  typedef int NodeI;
  typedef uint64_t EdgeI;
  typedef float ValT;

  typedef WeightedDiEdge<NodeI, ValT> Edge;
  typedef CSR<NodeI, EdgeI, ValT> CSR;

  pangolin::init();
  INFO("ctor");

  SECTION("el1") {
    std::vector<Edge> el = {
        {0, 1, 1.0},
    };

    INFO("from_edgelist");
    auto m = CSR::from_edges(el.begin(), el.end());

    INFO("check nnz");
    REQUIRE(m.nnz() == 1);
    REQUIRE(m.col_ind()[0] == 1);
    REQUIRE(m.num_rows() == 2);
    REQUIRE(m.num_cols() == 2);
  }

  SECTION("el2") {
    std::vector<Edge> el = {
        {0, 100, 1.0},
    };

    INFO("from_edgelist");
    auto m = CSR::from_edges(el.begin(), el.end());

    INFO("check nnz");
    REQUIRE(m.nnz() == 1);
    REQUIRE(m.col_ind()[0] == 100);
    REQUIRE(m.num_rows() == 101);
    REQUIRE(m.num_cols() == 101);
  }
}