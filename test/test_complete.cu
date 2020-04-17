#include <catch2/catch.hpp>

#include "pangolin/generator/complete.hpp"
#include "pangolin/init.hpp"
#include "pangolin/logger.hpp"

using namespace pangolin;

TEMPLATE_TEST_CASE("complete graph", "", int, size_t) {
  pangolin::init();
  pangolin::logger::set_level(pangolin::logger::Level::DEBUG);

  SECTION("0 nodes") {
    generator::Complete<TestType> c(0);

    REQUIRE(c.begin() == c.end());

    // expected number of edges
    size_t count = 0;
    for (auto _ : c) {
      (void)_;
      ++count;
    }
    REQUIRE(count == 0);
  };

  SECTION("1 node") {
    generator::Complete<TestType> c(1);

    // all edges are within graph
    for (auto e : c) {
      LOG(debug, "{} -> {}", e.src, e.dst);
      REQUIRE(e.src < 1);
      REQUIRE(e.dst < 1);
    }

    // expected number of edges
    size_t count = 0;
    for (auto _ : c) {
      (void)_;
      ++count;
    }
    REQUIRE(count == 0);
  };

  SECTION("2 nodes") {
    generator::Complete<TestType> c(2);

    // all edges are within graph
    for (auto e : c) {
      LOG(debug, "{} -> {}", e.src, e.dst);
      REQUIRE(e.src < 2);
      REQUIRE(e.dst < 2);
    }

    // expected number of edges
    size_t count = 0;
    for (auto _ : c) {
      (void)_;
      ++count;
    }
    REQUIRE(count == 2);
  };

  SECTION("3 nodes") {
    generator::Complete<TestType> c3(3);

    // all edges are within graph
    for (auto e : c3) {
      LOG(debug, "{} -> {}", e.src, e.dst);
      REQUIRE(e.src < 3);
      REQUIRE(e.dst < 3);
    }

    // expected number of edges
    size_t count = 0;
    for (auto _ : c3) {
      (void)_;
      ++count;
    }
    REQUIRE(count == 6);
  };
}