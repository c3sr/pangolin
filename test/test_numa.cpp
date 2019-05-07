#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/init.hpp"
#include "pangolin/numa.hpp"
#include "pangolin/topology/topology.hpp"

using namespace pangolin;

TEST_CASE("available") {
  pangolin::init();
  pangolin::numa::set_strict();
  pangolin::numa::available();
}

TEST_CASE("node_of_cpu") {
  pangolin::init();
  INFO("set strict");
  pangolin::numa::set_strict();
  INFO("get cpus");
  auto cpus = pangolin::topology::get_cpus();
  INFO("get node of each cpu");
  for (const auto cpu : cpus) {
    pangolin::numa::node_of_cpu(cpu);
  }
}
