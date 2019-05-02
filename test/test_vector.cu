#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/dense/vector.hu"
#include "pangolin/init.hpp"

using namespace pangolin;

TEMPLATE_TEST_CASE("default ctor", "[gpu]", int, size_t) {
  pangolin::init();
  Vector<TestType> v(10);
  REQUIRE(v.size() == 10);
}

TEMPLATE_TEST_CASE("fill ctor", "[gpu]", int, size_t) {
  pangolin::init();
  Vector<TestType> v(1);
  v[0] = 0;
  REQUIRE(*v.data() == 0);
}

TEST_CASE("fill ctor with val") {
  pangolin::init();
  Vector<int> v(10, 1);
  REQUIRE(v.size() == 10);
  for (size_t i = 0; i < v.size(); ++i) {
    REQUIRE(v[i] == 1);
  }
}

TEMPLATE_TEST_CASE("iterators", "[gpu]", int, size_t) {
  pangolin::init();
  Vector<TestType> v(1);
  REQUIRE(v.begin() != v.end());
  REQUIRE(v.end() - v.begin() == 1);
}

TEMPLATE_TEST_CASE("const iterators", "[gpu]", int, size_t) {
  pangolin::init();
  const Vector<TestType> v(1);
  REQUIRE(v.begin() != v.end());
  REQUIRE(v.end() - v.begin() == 1);
}

TEST_CASE("Vector copy-ctor") {
  pangolin::init();
  Vector<int> v(10, 1);
  Vector<int> w(v);

  REQUIRE(v.size() == 10);
  REQUIRE(w.size() == 10);
  for (size_t i = 0; i < v.size(); ++i) {
    REQUIRE(v[i] == 1);
    REQUIRE(w[i] == 1);
  }
}

TEST_CASE("Vector copy-assignment") {
  pangolin::init();
  Vector<int> v(15, 1);
  Vector<int> w(10, 2);

  w = v;
  REQUIRE(w.size() == 15);
  for (size_t i = 0; i < w.size(); ++i) {
    REQUIRE(v[i] == 1);
  }
}

TEST_CASE("Vector move-assignment") {
  pangolin::init();
  Vector<int> v(15, 1);
  Vector<int> w(10, 2);

  w = std::move(v);
  REQUIRE(w.size() == 15);
  for (size_t i = 0; i < w.size(); ++i) {
    REQUIRE(w[i] == 1);
  }
}

TEST_CASE("Vector reserve") {
  pangolin::init();
  Vector<int> v;
  v.reserve(10);
  REQUIRE(v.size() == 0);
  REQUIRE(v.capacity() >= 10);
}

TEST_CASE("initializer-list 3") {
  pangolin::init();
  Vector<int> v{0, 1, 2};
  REQUIRE(v.size() == 3);
  REQUIRE(v[0] == 0);
  REQUIRE(v[1] == 1);
  REQUIRE(v[2] == 2);
}

TEST_CASE("initializer-list 0") {
  pangolin::init();
  Vector<int> v{};
  REQUIRE(v.size() == 0);
}