#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/dense/array_view.hpp"
#include "pangolin/init.hpp"

using namespace pangolin;

TEMPLATE_TEST_CASE("array_view basics", "", int, uint64_t) {
  pangolin::init();
  logger::set_level(logger::Level::TRACE);

  SECTION("empty") {
    std::vector<TestType> v;
    ArrayView<TestType> a(v.data(), v.size());
    REQUIRE(a.begin() == a.end());
    REQUIRE(a.size() == 0);
  }
}

TEMPLATE_TEST_CASE("array_view more", "", int, uint64_t) {
  pangolin::init();
  logger::set_level(logger::Level::TRACE);
  std::vector<TestType> v(1);
  ArrayView<TestType> a(v.data(), v.size());

  SECTION("size 1") {
    REQUIRE(a.begin() + 1 == a.end());
    REQUIRE(a.size() == 1);
  }

  SECTION("assignment") {
    ArrayView<TestType> b;
    b = a;
    REQUIRE(a.begin() == b.begin());
    REQUIRE(a.size() == b.size());
  }

  SECTION("move") {
    ArrayView<TestType> b;
    b = std::move(a);
    REQUIRE(b.size() == 1);
  }

  SECTION("move ctor") {
    ArrayView<TestType> b = std::move(a);
    REQUIRE(b.size() == 1);
  }

  SECTION("copy ctor ctor") {
    ArrayView<TestType> b = a;
    REQUIRE(b.size() == 1);
  }

  SECTION("operator[]") {
    a[0] = 10;
    REQUIRE(a[0] == 10);
  }
}

TEST_CASE("array_view const", "") {
  pangolin::init();
  logger::set_level(logger::Level::TRACE);
  const std::vector<int> v(1, 7);
  ArrayView<const int> a(v.data(), v.size());
  REQUIRE(a[0] == 7);
}