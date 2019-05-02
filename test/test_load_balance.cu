#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <numeric>

#include "pangolin/algorithm/load_balance.cuh"
#include "pangolin/dense/vector.hu"
#include "pangolin/init.hpp"

using namespace pangolin;

TEST_CASE("load_balance", "[nogpu]") {
  pangolin::init();

  SECTION("no work items") {
    // here we just expect no crash

    std::vector<size_t> counts{0, 0, 0, 0};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<size_t> indices(numWorkItems);

    load_balance(indices.data(), numWorkItems, counts.data(), counts.size());
  }

  SECTION("1") {

    /*! counts
        1

        producer index
        0
     */

    std::vector<size_t> counts{1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<size_t> indices(numWorkItems);

    load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    REQUIRE(indices == std::vector<size_t>{0});
  }

  SECTION("1,1") {

    /*! counts
        1 1

        producer index
        0 1
     */

    std::vector<size_t> counts{1, 1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<size_t> indices(numWorkItems);

    load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    REQUIRE(indices == std::vector<size_t>{0, 1});
  }

  SECTION("1,1,1") {

    /*! counts
        1 1 1

        producer index
        0 1 2
     */

    std::vector<size_t> counts{1, 1, 1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<size_t> indices(numWorkItems);

    load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    REQUIRE(indices == std::vector<size_t>{0, 1, 2});
  }

  SECTION("1,0") {

    /*! counts
        1 0

        producer index
        0
     */

    std::vector<size_t> counts{1, 0};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<size_t> indices(numWorkItems);

    load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    REQUIRE(indices == std::vector<size_t>{0});
  }

  SECTION("2,0") {

    /*! counts
        2 0

        producer index
        0 0
     */

    std::vector<size_t> counts{2, 0};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<size_t> indices(numWorkItems);

    load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    REQUIRE(indices == std::vector<size_t>{0, 0});
  }

  SECTION("2,0,1") {

    /*! counts
        2 0 1

        producer index
        0 0 2
     */

    std::vector<size_t> counts{2, 0, 1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<size_t> indices(numWorkItems);

    load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    REQUIRE(indices == std::vector<size_t>{0, 0, 2});
  }

  SECTION("0,2,0,1") {

    /*! counts
        0 2 0 1

        producer index
        1 1 3
     */

    std::vector<size_t> counts{0, 2, 0, 1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<size_t> indices(numWorkItems);

    load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    REQUIRE(indices == std::vector<size_t>{1, 1, 3});
  }
}

TEST_CASE("device_load_balance", "[gpu]") {
  pangolin::init();

  SECTION("no work items") {
    // here we just expect no crash

    Vector<size_t> counts{0, 0, 0, 0};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    Vector<size_t> indices(numWorkItems);

    device_load_balance(indices.data(), numWorkItems, counts.data(), counts.size());
  }

  SECTION("1") {
    /*! counts
    1
    producer index
    0
    */

    Vector<size_t> counts{1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    Vector<size_t> indices(numWorkItems);
    device_load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    CUDA_RUNTIME(cudaDeviceSynchronize());

    // REQUIRE(*indices.data() == 0);
    REQUIRE(indices == Vector<size_t>{0});
  }

  SECTION("1,1") {

    /*! counts
    1 1
    producer index
    0 1
    */

    Vector<size_t> counts{1, 1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    Vector<size_t> indices(numWorkItems);

    device_load_balance(indices.data(), numWorkItems, counts.data(), counts.size());
    CUDA_RUNTIME(cudaDeviceSynchronize());

    REQUIRE(indices == Vector<size_t>{0, 1});
  }

  SECTION("1,1,1") {

    /*! counts
        1 1 1

        producer index
        0 1 2
     */

    Vector<size_t> counts{1, 1, 1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    Vector<size_t> indices(numWorkItems);

    device_load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    REQUIRE(indices == Vector<size_t>{0, 1, 2});
  }

  SECTION("1,0") {
    /*! counts
        1 0

        producer index
        0
     */

    Vector<size_t> counts{1, 0};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    Vector<size_t> indices(numWorkItems);

    device_load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    REQUIRE(indices == Vector<size_t>{0});
  }

  SECTION("2,0") {
    /*! counts
        2 0

        producer index
        0 0
     */

    Vector<size_t> counts{2, 0};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    Vector<size_t> indices(numWorkItems);

    device_load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    REQUIRE(indices == Vector<size_t>{0, 0});
  }

  SECTION("2,0,1") {
    /*! counts
        2 0 1

        producer index
        0 0 2
     */

    Vector<size_t> counts{2, 0, 1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    Vector<size_t> indices(numWorkItems);

    device_load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    REQUIRE(indices == Vector<size_t>{0, 0, 2});
  }

  SECTION("0,2,0,1") {
    /*! counts
        0 2 0 1

        producer index
        1 1 3
     */

    Vector<size_t> counts{0, 2, 0, 1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    Vector<size_t> indices(numWorkItems);

    device_load_balance(indices.data(), numWorkItems, counts.data(), counts.size());

    REQUIRE(indices == Vector<size_t>{1, 1, 3});
  }
}
