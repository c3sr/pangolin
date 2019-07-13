#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/init.hpp"
#include "pangolin/sparse/csr_binned.hpp"

using namespace pangolin;

typedef uint32_t NodeIndex;
typedef uint64_t EdgeIndex;

TEST_CASE("ctor") {
  pangolin::init();
  typedef CSRBinned<NodeIndex, EdgeIndex> CSR;
  CSR m(0);

  SECTION("reserve") {
    m.reserve(10, 100);
    REQUIRE(m.num_rows() == 0);
    REQUIRE(m.nnz() == 0);
  }
}

TEST_CASE("from_edgelist") {
  pangolin::init();
  pangolin::logger::set_level(pangolin::logger::Level::DEBUG);
  typedef CSRBinned<NodeIndex, EdgeIndex> CSR;
  EdgeList el = {
      {0, 1},
  };

  INFO("from_edgelist");
  auto csr = CSR::from_edges(el.begin(), el.end(), 1);

  INFO("check nnz");
  REQUIRE(csr.nnz() == 1);
  INFO("check num_rows");
  REQUIRE(csr.num_rows() == 2);
  INFO("check row_start");
  REQUIRE(csr.row_start()[0] == 0);
  INFO("check row_stop");
  REQUIRE(csr.row_stop()[0] == 1);
  INFO("check colInd_");
  REQUIRE(csr.colInd_[0] == 1);
}

TEST_CASE("CSR<int>::from_edges upper triangular") {
  pangolin::init();
  pangolin::logger::set_level(pangolin::logger::Level::DEBUG);
  typedef CSRBinned<NodeIndex, EdgeIndex> CSR;
  std::vector<EdgeTy<NodeIndex>> el = {{0, 1}, {0, 2}, {1, 0}, {1, 2}, {1, 3}, {1, 4}, {2, 0}, {2, 1},
                                       {2, 3}, {2, 4}, {3, 1}, {3, 2}, {3, 4}, {4, 1}, {4, 2}, {4, 3}};

  INFO("from_edgelist");
  auto ut = [](EdgeTy<NodeIndex> e) { return e.first < e.second; };
  auto csr = CSR::from_edges(el.begin(), el.end(), 4, ut);

  REQUIRE(csr.nnz() == 8);

  INFO("check partitions are contiguous");
  for (size_t i = 0; i < csr.num_partitions() - 1; ++i) {
    auto view = csr.view(i);
    auto nextView = csr.view(i + 1);
    REQUIRE(view.rowStop_ == nextView.rowStart_);
  }
  INFO("last partition end should be row ends");
  REQUIRE(csr.view(csr.num_partitions() - 1).rowStop_ == csr.view().rowStop_);
}

TEST_CASE("CSR<int>::from_edges lower triangular") {
  pangolin::init();
  pangolin::logger::set_level(pangolin::logger::Level::DEBUG);
  typedef CSRBinned<NodeIndex, EdgeIndex> CSR;
  typedef EdgeTy<NodeIndex> Edge;
  std::vector<Edge> el = {{0, 1}, {0, 2}, {1, 0}, {1, 2}, {1, 3}, {1, 4}, {2, 0}, {2, 1},
                          {2, 3}, {2, 4}, {3, 1}, {3, 2}, {3, 4}, {4, 1}, {4, 2}, {4, 3}};

  INFO("from_edgelist");
  auto lt = [](Edge e) { return e.first > e.second; };
  auto csr = CSR::from_edges(el.begin(), el.end(), 4, lt);

  REQUIRE(csr.nnz() == 8);
}

TEST_CASE("edge 2->100 ut") {
  pangolin::init();
  pangolin::logger::set_level(pangolin::logger::Level::DEBUG);
  typedef CSRBinned<NodeIndex, EdgeIndex> CSR;
  typedef EdgeTy<NodeIndex> Edge;
  std::vector<Edge> el = {{2, 100}};

  auto keep = [](Edge e) { return e.first < e.second; };
  auto csr = CSR::from_edges(el.begin(), el.end(), 100, keep);

  SECTION("nnz") {
    REQUIRE(csr.nnz() == 1);
    REQUIRE(csr.nnz() == csr.view().nnz());
  }

  SECTION("num_rows") {
    REQUIRE(csr.num_rows() == 101); // 0 - 100
    REQUIRE(csr.num_rows() == csr.view().num_rows());
  }
}

TEST_CASE("edge 2->100 lt") {
  pangolin::init();
  pangolin::logger::set_level(pangolin::logger::Level::TRACE);
  typedef CSRBinned<NodeIndex, EdgeIndex> CSR;
  typedef EdgeTy<NodeIndex> Edge;
  std::vector<Edge> el = {{2, 100}};

  auto keep = [](Edge e) { return e.first > e.second; };
  auto csr = CSR::from_edges(el.begin(), el.end(), 100, keep);

  SECTION("nnz") {
    REQUIRE(csr.nnz() == 0);
    REQUIRE(csr.nnz() == csr.view().nnz());
  }

  SECTION("num_rows") {
    REQUIRE(csr.num_rows() == 0);
    REQUIRE(csr.num_rows() == csr.view().num_rows());
  }
}
