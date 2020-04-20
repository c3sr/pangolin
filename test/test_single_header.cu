/*! Test that the single header include works
*/

#include <catch2/catch.hpp>

#include "pangolin/pangolin.hpp"
#include "pangolin/pangolin.cuh"

using namespace pangolin;

TEST_CASE("do nothing") {
    // test passes if it compiles
    REQUIRE(true);
}