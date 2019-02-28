#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/pangolin.hpp"

using namespace pangolin;

TEST_CASE("Vector ctor(2)") {

  using NodeTy = int;

  FullyConnected<NodeTy> fc(2);
  auto i = fc.begin();
  auto e = *i;
  REQUIRE(e == EdgeTy<NodeTy>(0, 0));
  i++;
  REQUIRE(*i == EdgeTy<NodeTY>(0, 1));
}
