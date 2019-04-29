#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/dense/vector.hu"

using namespace pangolin;

TEST_CASE("Vector ctor(1)") {
  Vector<int> v(10);
  REQUIRE(v.size() == 10);
}

TEST_CASE("Vector ctor(2)") {
  Vector<int> v(10, 1);
  REQUIRE(v.size() == 10);
  for (size_t i = 0; i < v.size(); ++i) {
    REQUIRE(v[i] == 1);
  }
}

TEST_CASE("Vector copy-assignment") {
  Vector<int> v(15, 1);
  Vector<int> w(10, 2);

  w = v;
  REQUIRE(w.size() == 15);
  for (size_t i = 0; i < w.size(); ++i) {
    REQUIRE(v[i] == 1);
  }
}

TEST_CASE("Vector move-assignment") {
  Vector<int> v(15, 1);
  Vector<int> w(10, 2);

  w = std::move(v);
  REQUIRE(w.size() == 15);
  for (size_t i = 0; i < w.size(); ++i) {
    REQUIRE(w[i] == 1);
  }

  REQUIRE(v.size() == 0);
}

TEST_CASE("Vector reserve") {
  Vector<int> v;
  v.reserve(10);
  REQUIRE(v.size() == 0);
  REQUIRE(v.capacity() >= 10);
}

TEST_CASE("initializer-list 3") {
  Vector<int> v{0, 1, 2};
  REQUIRE(v.size() == 3);
  REQUIRE(v[0] == 0);
  REQUIRE(v[1] == 1);
  REQUIRE(v[2] == 2);
}

TEST_CASE("initializer-list 0") {
  Vector<int> v{};
  REQUIRE(v.size() == 0);
}