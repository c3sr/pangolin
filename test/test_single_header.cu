/*! Test that the single header include works
*/

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/pangolin.hpp"
#include "pangolin/pangolin.cuh"

using namespace pangolin;

TEST_CASE("do nothing") {
    // test passes if it compiles
    REQUIRE(true);
}