
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "pangolin/cuda_cxx/rc_stream.hpp"
#include "pangolin/init.hpp"

using namespace pangolin;

TEST_CASE("ctor", "[gpu]") {
  pangolin::init();
  logger::set_level(logger::Level::TRACE);

  RcStream stream;

  REQUIRE(stream.stream());
  REQUIRE(stream.count() == 1);

  SECTION("move ctor", "[gpu]") {
    RcStream s2(std::move(stream));
    REQUIRE(s2.count() == 1);
    REQUIRE(s2.stream());
  }

  SECTION("copy ctor", "[gpu]") {
    {
      RcStream s2(stream);
      REQUIRE(s2.count() == 2);
      REQUIRE(stream.count() == 2);
      REQUIRE(s2 == stream);
    }
    REQUIRE(stream.count() == 1);
    REQUIRE(stream.stream());
  }
}

TEST_CASE("ref ctor", "[gpu]") {
  pangolin::init();
  logger::set_level(logger::Level::TRACE);

  int device;
  cudaStream_t raw;
  CUDA_RUNTIME(cudaGetDevice(&device));
  CUDA_RUNTIME(cudaStreamCreate(&raw));
  {
    RcStream stream(device, raw);
    REQUIRE(stream.stream() == raw);
  }
  REQUIRE(raw); // stream should not have been freed

  CUDA_RUNTIME(cudaStreamDestroy(raw));
}