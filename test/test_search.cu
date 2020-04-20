#include <catch2/catch.hpp>

#include "pangolin/algorithm/search.cuh"
#include "pangolin/init.hpp"

using namespace pangolin;

template <typename T> size_t lb(T searchVal, std::initializer_list<T> a) {
  std::vector<T> A(a);
  return binary_search<Bounds::LOWER>(A.data(), A.size(), searchVal);
}

template <typename T> size_t ub(T searchVal, std::initializer_list<T> a) {
  std::vector<T> A(a);
  return binary_search<Bounds::UPPER>(A.data(), A.size(), searchVal);
}

TEST_CASE("binary_0") {
  pangolin::init();
  REQUIRE(0 == lb(0, {}));
  REQUIRE(0 == ub(0, {}));
}

TEST_CASE("binary_1") {
  pangolin::init();
  REQUIRE(0 == lb(0, {0}));
  REQUIRE(1 == ub(0, {0}));
  REQUIRE(1 == lb(1, {0}));
  REQUIRE(1 == ub(1, {0}));
  REQUIRE(0 == lb(0, {1}));
  REQUIRE(0 == ub(0, {1}));
  REQUIRE(0 == lb(1, {1}));
  REQUIRE(1 == ub(1, {1}));
  REQUIRE(1 == lb(2, {1}));
  REQUIRE(1 == ub(2, {1}));
}

TEST_CASE("binary_2") {
  pangolin::init();
  REQUIRE(0 == lb(0, {0, 0}));
  REQUIRE(2 == ub(0, {0, 0}));
  REQUIRE(0 == lb(0, {0, 1}));
  REQUIRE(1 == ub(0, {0, 1}));
  REQUIRE(1 == lb(1, {0, 1}));
  REQUIRE(2 == ub(1, {0, 1}));
  REQUIRE(2 == lb(2, {0, 1}));
  REQUIRE(2 == ub(2, {0, 1}));
}

TEST_CASE("binary_3") {
  pangolin::init();
  REQUIRE(1 == lb(1, {0, 1, 2}));
  REQUIRE(2 == ub(1, {0, 1, 2}));
}
