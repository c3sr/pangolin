#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <numeric>

#include "pangolin/algorithm/load_balance.cuh"
#include "pangolin/init.hpp"

using namespace pangolin;

TEST_CASE("load_balance") {
  pangolin::init();

  SECTION("no work items") {
    std::vector<int> counts{0, 0, 0, 0};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<int> indices(numWorkItems);

    load_balance(indices.data(), numWorkItems, counts.data(), counts.size());
  }

  SECTION("1") {

    /*! counts
        1

        producer index
        0
     */

    std::vector<int> counts{1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<int> indices(numWorkItems);

    load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    REQUIRE(indices == std::vector<int>{0});
  }

  SECTION("1,1") {

    /*! counts
        1 1

        producer index
        0 1
     */

    std::vector<int> counts{1, 1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<int> indices(numWorkItems);

    load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    REQUIRE(indices == std::vector<int>{0, 1});
  }

  SECTION("1,1,1") {

    /*! counts
        1 1 1

        producer index
        0 1 2
     */

    std::vector<int> counts{1, 1, 1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<int> indices(numWorkItems);

    load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    REQUIRE(indices == std::vector<int>{0, 1, 2});
  }

  SECTION("1,0") {

    /*! counts
        1 0

        producer index
        0
     */

    std::vector<int> counts{1, 0};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<int> indices(numWorkItems);

    load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    REQUIRE(indices == std::vector<int>{0});
  }

  SECTION("2,0") {

    /*! counts
        2 0

        producer index
        0 0
     */

    std::vector<int> counts{2, 0};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<int> indices(numWorkItems);

    load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    REQUIRE(indices == std::vector<int>{0, 0});
  }

  SECTION("2,0,1") {

    /*! counts
        2 0 1

        producer index
        0 0 2
     */

    std::vector<int> counts{2, 0, 1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<int> indices(numWorkItems);

    load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    REQUIRE(indices == std::vector<int>{0, 0, 2});
  }

  SECTION("0, 2,0,1") {

    /*! counts
        0 2 0 1

        producer index
        1 1 3
     */

    std::vector<int> counts{0, 2, 0, 1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<int> indices(numWorkItems);

    load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    REQUIRE(indices == std::vector<int>{1, 1, 3});
  }
}
