#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/init.hpp"
#include "pangolin/sparse/coo.hpp"

using namespace pangolin;

TEST_CASE("ctor") {
  pangolin::init();
  COO<int> coo;
}

TEST_CASE("COO<int>::from_edgelist") {
  pangolin::init();
  INFO("ctor");
  COO<int> coo;

  EdgeList el = {
      {0, 1},
  };

  INFO("from_edgelist");
  coo = COO<int>::from_edgelist(el);

  INFO("check nnz");
  REQUIRE(coo.nnz() == 1);
  REQUIRE(coo.row_ind()[0] == 0);
  REQUIRE(coo.col_ind()[0] == 1);
}

TEST_CASE("COO<int>::from_edges upper triangular") {
  pangolin::init();
  std::vector<EdgeTy<uint64_t>> el = {{0, 1}, {0, 2}, {1, 0}, {1, 2}, {1, 3}, {1, 4}, {2, 0}, {2, 1},
                                      {2, 3}, {2, 4}, {3, 1}, {3, 2}, {3, 4}, {4, 1}, {4, 2}, {4, 3}};

  INFO("from_edgelist");
  auto ut = [](EdgeTy<uint64_t> e) { return e.first < e.second; };
  auto coo = COO<uint64_t>::from_edges(el.begin(), el.end(), ut);

  REQUIRE(coo.nnz() == 8);
  REQUIRE(coo.num_rows() == 5);
}

TEST_CASE("COO<int>::from_edges lower triangular") {
  pangolin::init();
  std::vector<EdgeTy<uint64_t>> el = {{0, 1}, {0, 2}, {1, 0}, {1, 2}, {1, 3}, {1, 4}, {2, 0}, {2, 1},
                                      {2, 3}, {2, 4}, {3, 1}, {3, 2}, {3, 4}, {4, 1}, {4, 2}, {4, 3}};

  INFO("from_edgelist");
  auto lt = [](EdgeTy<uint64_t> e) { return e.first > e.second; };
  auto coo = COO<uint64_t>::from_edges(el.begin(), el.end(), lt);

  REQUIRE(coo.nnz() == 8);
  REQUIRE(coo.num_rows() == 5);
}

TEST_CASE("COO<int>::num_nodes") {
  pangolin::init();
  std::vector<EdgeTy<uint64_t>> el = {{1, 2}, {1, 3}, {2, 1}, {2, 3}, {2, 4}, {2, 5}, {3, 1}, {3, 2},
                                      {3, 4}, {3, 5}, {4, 2}, {4, 3}, {4, 5}, {5, 2}, {5, 3}, {5, 4}};

  INFO("from_edgelist");
  auto ut = [](EdgeTy<uint64_t> e) { return e.first < e.second; };
  auto coo = COO<uint64_t>::from_edges(el.begin(), el.end(), ut);

  REQUIRE(coo.num_nodes() == 6);
  REQUIRE(coo.num_rows() == 6);
}

TEST_CASE("COO<int>::num_rows") {
  pangolin::init();
  std::vector<EdgeTy<uint64_t>> el = {{2, 100}};

  INFO("from_edgelist");
  auto ut = [](EdgeTy<uint64_t> e) { return e.first < e.second; };
  auto coo = COO<uint64_t>::from_edges(el.begin(), el.end(), ut);

  REQUIRE(coo.num_rows() == 101);
}