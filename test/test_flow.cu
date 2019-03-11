#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/flow.hpp"

using namespace pangolin;

TEST_CASE("") {
  FlowVector<int> v;

  v.add_producer(Component::CPU(0, AccessKind::ManyOnce))
      .add_consumer(Component::GPU(0, AccessKind::ManyShared))
      .add_consumer(Component::GPU(1, AccessKind::ManyShared));
}