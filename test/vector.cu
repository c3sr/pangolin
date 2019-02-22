#include "pangolin/pangolin.hpp"

#include <catch.hpp>
using namespace pangolin;

TEST_CASE("Vector ctor(2)") {
    Vector<int> v(10, 1);
    REQUIRE(v.size() == 10);
    for (size_t i = 0; i < v.size(); ++i) {
        REQUIRE(v[9] == 1);
    }
}

TEST_CASE("Vector reserve") {
    Vector<int> v;
    v.reserve(10);
    REQUIRE(v.size() == 0);
    REQUIRE(v.capacity() >= 10);
}