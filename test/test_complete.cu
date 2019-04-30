#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/generator/complete.hpp"
#include "pangolin/init.hpp"

using namespace pangolin;

TEST_CASE("c2") {
  pangolin::init();
  using NodeTy = unsigned;

  generator::Complete<NodeTy> c2(2);

  // all edges are within graph
  for (auto e : c2) {
    REQUIRE(e.first < 2);
    REQUIRE(e.second < 2);
  }

  // expected number of edges
  size_t count = 0;
  for (auto _ : c2) {
    (void)_;
    ++count;
  }
  REQUIRE(count == 2);
}

TEST_CASE("c3") {
  pangolin::init();
  using NodeTy = int;

  generator::Complete<NodeTy> c3(3);

  // all edges are within graph
  for (auto e : c3) {
    REQUIRE(e.first < 3);
    REQUIRE(e.second < 3);
  }

  // expected number of edges
  size_t count = 0;
  for (auto _ : c3) {
    (void)_;
    ++count;
  }
  REQUIRE(count == 6);
}