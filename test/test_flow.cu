#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/flow/flow.hpp"

using namespace pangolin;

TEST_CASE("", "[gpu][numa]") {

  numa_set_strict(1);
  LOG(debug, "set numa_set_strict(1)");
  numa_set_bind_policy(1);
  LOG(debug, "set numa_set_bind_policy(1)");
  numa_exit_on_warn = 1;
  LOG(debug, "set numa_exit_on_warn = 1");
  numa_exit_on_error = 1;
  LOG(debug, "set numa_exit_on_error = 1");

  FlowVector<int> v;

  v.add_producer(Component::CPU(0, AccessKind::OnceExclusive))
      .add_consumer(Component::GPU(0, AccessKind::ManyShared))
      .add_consumer(Component::GPU(1, AccessKind::ManyShared));
}

TEST_CASE("", "[gpu][numa]") {}