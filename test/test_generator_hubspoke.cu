#include <catch2/catch.hpp>

#include "pangolin/generator/hubspoke.hpp"
#include "pangolin/init.hpp"
#include "pangolin/logger.hpp"

using namespace pangolin;

TEMPLATE_TEST_CASE("hub and spoke graph", "", int, size_t) {
  pangolin::init();
  pangolin::logger::set_level(pangolin::logger::Level::DEBUG);

  SECTION("0 spokes") {
    generator::HubSpoke<TestType> c(0);

    REQUIRE(c.begin() == c.end());

    // expected number of edges
    size_t count = 0;
    for (auto _ : c) {
      (void)_;
      ++count;
    }
    REQUIRE(count == 0);
  };

  SECTION("1 spoke") {
    generator::HubSpoke<TestType> c(1);

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

  SECTION("2 spokes") {
    generator::HubSpoke<TestType> c(2);

    // all edges are within graph
    for (auto e : c) {
      LOG(debug, "{} -> {}", e.src, e.dst);
      REQUIRE(e.src < 3);
      REQUIRE(e.dst < 3);
    }

    // expected number of edges
    size_t count = 0;
    for (auto _ : c) {
      (void)_;
      ++count;
    }
    REQUIRE(count == 6);
  };

  SECTION("3 spokes") {
    LOG(debug, "3 spokes");
    generator::HubSpoke<TestType> c3(3);

    // all edges are within graph
    for (auto e : c3) {
      LOG(debug, "{} -> {}", e.src, e.dst);
      REQUIRE(e.src < 4);
      REQUIRE(e.dst < 4);
    }

    // expected number of edges
    size_t count = 0;
    for (auto _ : c3) {
      (void)_;
      ++count;
    }
    REQUIRE(count == 10);
  };

  SECTION("513 spokes") {
    LOG(debug, "513 spokes");
    generator::HubSpoke<TestType> c(513);

    // all edges are within graph
    for (auto e : c) {
      LOG(debug, "{} -> {}", e.src, e.dst);
      REQUIRE(e.src < 514);
      REQUIRE(e.dst < 514);
    }

    // expected number of edges
    size_t count = 0;
    for (auto _ : c) {
      (void)_;
      ++count;
    }
    REQUIRE(count == 513 * 2 + 512 * 2);
  };
}