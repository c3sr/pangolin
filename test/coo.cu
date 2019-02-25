#include "pangolin/pangolin.hpp"

#include <catch.hpp>
using namespace pangolin;

TEST_CASE("COO<int>::from_edgelist") {
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