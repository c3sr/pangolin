#include <catch2/catch.hpp>

#include "pangolin/algorithm/merge_path.cuh"
#include "pangolin/init.hpp"

using namespace pangolin;

template <typename T> size_t mp_lb(T diag, std::vector<T> A, std::vector<T> B) {
  return merge_path<Bounds::LOWER>(A.data(), A.size(), B.data(), B.size(), diag);
}

template <typename T> size_t mp_ub(T diag, std::vector<T> A, std::vector<T> B) {
  return merge_path<Bounds::UPPER>(A.data(), A.size(), B.data(), B.size(), diag);
}

TEST_CASE("0") {
  pangolin::init();

  /*
    0 1 2 3
    _ _ _ _
  4        |
  5        |
  6        |
  7        |
  */

  std::vector<unsigned> a = {0, 1, 2, 3};
  std::vector<unsigned> b = {4, 5, 6, 7};

  REQUIRE(0 == mp_lb(0u, a, b));
  REQUIRE(1 == mp_lb(1u, a, b));
  REQUIRE(2 == mp_lb(2u, a, b));
  REQUIRE(3 == mp_lb(3u, a, b));
  REQUIRE(4 == mp_lb(4u, a, b));
  REQUIRE(4 == mp_lb(5u, a, b));
  REQUIRE(4 == mp_lb(6u, a, b));
  REQUIRE(4 == mp_lb(7u, a, b));
  REQUIRE(4 == mp_lb(8u, a, b));

  REQUIRE(0 == mp_ub(0u, a, b));
  REQUIRE(1 == mp_ub(1u, a, b));
  REQUIRE(2 == mp_ub(2u, a, b));
  REQUIRE(3 == mp_ub(3u, a, b));
  REQUIRE(4 == mp_ub(4u, a, b));
  REQUIRE(4 == mp_ub(5u, a, b));
  REQUIRE(4 == mp_ub(6u, a, b));
  REQUIRE(4 == mp_ub(7u, a, b));
  REQUIRE(4 == mp_ub(8u, a, b));
}

TEST_CASE("1") {
  pangolin::init();

  /*
    0 1 2 3
    _ _ _
  3      |_
  4        |
  5        |
  6        |
  */

  std::vector<unsigned> a = {0, 1, 2, 3};
  std::vector<unsigned> b = {3, 4, 5, 6};

  REQUIRE(0 == mp_lb(0u, a, b));
  REQUIRE(0 == mp_ub(0u, a, b));

  REQUIRE(1 == mp_lb(1u, a, b));
  REQUIRE(1 == mp_ub(1u, a, b));

  REQUIRE(2 == mp_lb(2u, a, b));
  REQUIRE(2 == mp_ub(2u, a, b));

  REQUIRE(3 == mp_lb(3u, a, b));
  REQUIRE(3 == mp_ub(3u, a, b));

  REQUIRE(4 == mp_lb(4u, a, b));
  REQUIRE(3 == mp_ub(4u, a, b));

  REQUIRE(4 == mp_lb(5u, a, b));
  REQUIRE(4 == mp_ub(5u, a, b));

  REQUIRE(4 == mp_lb(6u, a, b));
  REQUIRE(4 == mp_ub(6u, a, b));

  REQUIRE(4 == mp_lb(7u, a, b));
  REQUIRE(4 == mp_ub(7u, a, b));

  REQUIRE(4 == mp_lb(8u, a, b));
  REQUIRE(4 == mp_ub(8u, a, b));
}

TEST_CASE("2") {
  pangolin::init();

  /*
    0 1 2 3
    _ _ _
  3      |_
  4        |
  5        |
  6        |
  */

  std::vector<unsigned> a = {0, 1, 2, 3};
  std::vector<unsigned> b = {2, 3, 4, 5};

  REQUIRE(3 == mp_lb(3u, a, b));
  REQUIRE(3 == mp_lb(4u, a, b));
  REQUIRE(4 == mp_lb(5u, a, b));
  REQUIRE(4 == mp_lb(6u, a, b));

  REQUIRE(2 == mp_ub(3u, a, b));
  REQUIRE(3 == mp_ub(4u, a, b));
  REQUIRE(3 == mp_ub(5u, a, b));
  REQUIRE(4 == mp_ub(6u, a, b));
}