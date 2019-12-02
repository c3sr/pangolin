#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/dense/vector.cuh"
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

TEST_CASE("resize up with value") {
  pangolin::init();
  Vector<int> v(1, 1);

  v.resize(10, 2);
  REQUIRE(v.size() == 10);
  REQUIRE(v[0] == 1);
  for (size_t i = 1; i < v.size(); ++i) {
    REQUIRE(v[i] == 2);
  }
}

TEST_CASE("resize down with value") {
  pangolin::init();
  Vector<int> v(10, 1);

  v.resize(5, 2);
  REQUIRE(v.size() == 5);
  for (size_t i = 1; i < v.size(); ++i) {
    REQUIRE(v[i] == 1);
  }
}

TEMPLATE_TEST_CASE("initializer_list", "[gpu]", int, size_t) {
  pangolin::init();
  SECTION("{}") {
    Vector<TestType> v{};
    REQUIRE(v.size() == 0);
  }

  SECTION("{1}") {
    Vector<TestType> v{1};
    REQUIRE(v[0] == 1);
    REQUIRE(v.size() == 1);
  }

  SECTION("{1,3}") {
    Vector<TestType> v{1, 3};
    REQUIRE(v[0] == 1);
    REQUIRE(v[1] == 3);
    REQUIRE(v.size() == 2);
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

TEMPLATE_TEST_CASE("Vector size", "[gpu]", int, size_t) {
  pangolin::init();
  Vector<int> v;

  SECTION("vectors can be resized") {
    v.resize(1);
    REQUIRE(v.size() == 1);
    v[0] = 5;

    v.resize(10);
    REQUIRE(v.size() == 10);
    v[9] = 100;
    REQUIRE(v[9] == 100);
    REQUIRE(v[0] == 5);

    v.resize(1);
    REQUIRE(v.size() == 1);
    REQUIRE(v[0] == 5);
  }

  SECTION("vectors can be push_backed") {
    v.push_back(4);
    REQUIRE(v.size() == 1);
    REQUIRE(v[0] == 4);

    for (size_t i = 0; i < 99; ++i) {
      v.push_back(i);
    }
    REQUIRE(v[99] == 98);

    v.resize(5);
    REQUIRE(v.size() == 5);

    v.push_back(1);
    REQUIRE(v.size() == 6);
    REQUIRE(v[5] == 1);
  }
}
