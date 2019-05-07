#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <numeric>

#include "pangolin/algorithm/load_balance.cuh"
#include "pangolin/dense/vector.hu"
#include "pangolin/init.hpp"

using namespace pangolin;

TEST_CASE("device_load_balance", "[gpu]") {
  pangolin::init();

  SECTION("no work items") {
    // here we just expect no crash
    pangolin::logger::set_level(logger::Level::DEBUG);

    Vector<size_t> counts{0, 0, 0, 0};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    Vector<size_t> indices(numWorkItems);

    SECTION("no ranks") {
      size_t *ranks = nullptr;
      device_load_balance(indices.data(), ranks, numWorkItems, counts.data(), counts.size());
    }

    SECTION("ranks") {
      Vector<size_t> ranks(numWorkItems);
      device_load_balance(indices.data(), ranks.data(), numWorkItems, counts.data(), counts.size());
    }
  }

  SECTION("1") {
    /*! counts
    1
    producer index
    0
    producer rank
    0
    */

    Vector<size_t> counts{1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    Vector<size_t> indices(numWorkItems);

    SECTION("ranks") {
      Vector<size_t> ranks(numWorkItems);
      device_load_balance(indices.data(), ranks.data(), numWorkItems, counts.data(), counts.size());
      CUDA_RUNTIME(cudaDeviceSynchronize());
      REQUIRE(indices == Vector<size_t>{0});
      REQUIRE(ranks == Vector<size_t>{0});
    }

    SECTION("no ranks") {
      size_t *ranks = nullptr;
      device_load_balance(indices.data(), ranks, numWorkItems, counts.data(), counts.size());
      CUDA_RUNTIME(cudaDeviceSynchronize());
      REQUIRE(indices == Vector<size_t>{0});
    }
  }

  SECTION("1,1") {

    /*! counts
    1 1
    producer index
    0 1
    producer ranks
    0 0
    */

    Vector<size_t> counts{1, 1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    Vector<size_t> indices(numWorkItems);

    SECTION("ranks") {
      Vector<size_t> ranks(numWorkItems);
      device_load_balance(indices.data(), ranks.data(), numWorkItems, counts.data(), counts.size());
      CUDA_RUNTIME(cudaDeviceSynchronize());
      REQUIRE(indices == Vector<size_t>{0, 1});
      REQUIRE(ranks == Vector<size_t>{0, 0});
    }

    SECTION("no ranks") {
      size_t *ranks = nullptr;
      device_load_balance(indices.data(), ranks, numWorkItems, counts.data(), counts.size());
      CUDA_RUNTIME(cudaDeviceSynchronize());
      REQUIRE(indices == Vector<size_t>{0, 1});
    }
  }

  SECTION("1,1,1") {

    /*! counts
        1 1 1
        producer index
        0 1 2
        producer rank
        0 0 0
     */

    Vector<size_t> counts{1, 1, 1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    Vector<size_t> indices(numWorkItems);

    SECTION("ranks") {
      Vector<size_t> ranks(numWorkItems);
      device_load_balance(indices.data(), ranks.data(), numWorkItems, counts.data(), counts.size());
      CUDA_RUNTIME(cudaDeviceSynchronize());
      REQUIRE(indices == Vector<size_t>{0, 1, 2});
      REQUIRE(ranks == Vector<size_t>{0, 0, 0});
    }

    SECTION("no ranks") {
      size_t *ranks = nullptr;
      device_load_balance(indices.data(), ranks, numWorkItems, counts.data(), counts.size());
      CUDA_RUNTIME(cudaDeviceSynchronize());
      REQUIRE(indices == Vector<size_t>{0, 1, 2});
    }
  }

  SECTION("1,0") {
    /*! counts
        1 0
        producer index
        0
        producer ranks
        0
     */

    Vector<size_t> counts{1, 0};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    Vector<size_t> indices(numWorkItems);

    SECTION("ranks") {
      Vector<size_t> ranks(numWorkItems);
      device_load_balance(indices.data(), ranks.data(), numWorkItems, counts.data(), counts.size());
      CUDA_RUNTIME(cudaDeviceSynchronize());
      REQUIRE(indices == Vector<size_t>{0});
      REQUIRE(ranks == Vector<size_t>{0});
    }

    SECTION("no ranks") {
      size_t *ranks = nullptr;
      device_load_balance(indices.data(), ranks, numWorkItems, counts.data(), counts.size());
      CUDA_RUNTIME(cudaDeviceSynchronize());
      REQUIRE(indices == Vector<size_t>{0});
    }
  }

  SECTION("2,0") {
    /*! counts
        2 0
        producer index
        0 0
        producer rank
        0 1
     */

    Vector<size_t> counts{2, 0};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    Vector<size_t> indices(numWorkItems);

    SECTION("ranks") {
      Vector<size_t> ranks(numWorkItems);
      device_load_balance(indices.data(), ranks.data(), numWorkItems, counts.data(), counts.size());
      CUDA_RUNTIME(cudaDeviceSynchronize());
      REQUIRE(indices == Vector<size_t>{0, 0});
      REQUIRE(ranks == Vector<size_t>{0, 1});
    }

    SECTION("no ranks") {
      size_t *ranks = nullptr;
      device_load_balance(indices.data(), ranks, numWorkItems, counts.data(), counts.size());
      CUDA_RUNTIME(cudaDeviceSynchronize());
      REQUIRE(indices == Vector<size_t>{0, 0});
    }
  }

  SECTION("2,0,1") {
    /*! counts
        2 0 1
        producer index
        0 0 2
        producer rank
        0 1 0
     */

    Vector<size_t> counts{2, 0, 1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    Vector<size_t> indices(numWorkItems);

    SECTION("ranks") {
      Vector<size_t> ranks(numWorkItems);
      device_load_balance(indices.data(), ranks.data(), numWorkItems, counts.data(), counts.size());
      CUDA_RUNTIME(cudaDeviceSynchronize());
      REQUIRE(indices == Vector<size_t>{0, 0, 2});
      REQUIRE(ranks == Vector<size_t>{0, 1, 0});
    }

    SECTION("no ranks") {
      size_t *ranks = nullptr;
      device_load_balance(indices.data(), ranks, numWorkItems, counts.data(), counts.size());
      CUDA_RUNTIME(cudaDeviceSynchronize());
      REQUIRE(indices == Vector<size_t>{0, 0, 2});
    }
  }

  SECTION("0,2,0,1") {
    /*! counts
        0 2 0 1
        producer index
        1 1 3
        producer rank
        0 1 0
     */

    Vector<size_t> counts{0, 2, 0, 1};
    size_t numWorkItems = std::accumulate(counts.begin(), counts.end(), 0);
    Vector<size_t> indices(numWorkItems);

    SECTION("ranks") {
      Vector<size_t> ranks(numWorkItems);
      device_load_balance(indices.data(), ranks.data(), numWorkItems, counts.data(), counts.size());
      CUDA_RUNTIME(cudaDeviceSynchronize());
      REQUIRE(indices == Vector<size_t>{1, 1, 3});
      REQUIRE(ranks == Vector<size_t>{0, 1, 0});
    }

    SECTION("no ranks") {
      size_t *ranks = nullptr;
      device_load_balance(indices.data(), ranks, numWorkItems, counts.data(), counts.size());
      CUDA_RUNTIME(cudaDeviceSynchronize());
      REQUIRE(indices == Vector<size_t>{1, 1, 3});
    }
  }
}
