#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/flow.hpp"

using namespace pangolin;

TEST_CASE("") {
  FlowVector<int> v;
  v.with_producer(Component::CPU(0));
  v.with_consumer(Component::GPU(0)).with_consumer(Component::GPU(1));
}