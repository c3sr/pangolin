#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <sstream>

#include "pangolin/file/bmtx_stream.hpp"
#include "pangolin/init.hpp"

using namespace pangolin;

TEST_CASE("0") {
  pangolin::init();
  pangolin::logger::set_level(pangolin::logger::Level::TRACE);

  int64_t ints[6] = {1, 7, 3, 1, 7, 0};
  std::string data((char *)ints, sizeof(ints));
  auto stream = std::stringstream(data);

  BmtxStream<uint64_t, std::stringstream> bmtx(std::move(stream));
  REQUIRE(bmtx.num_rows() == 1);
  REQUIRE(bmtx.num_cols() == 7);
  REQUIRE(bmtx.nnz() == 3);

  EdgeTy<uint64_t> edge;
  REQUIRE(bmtx.readEdge(edge));
  REQUIRE(edge.first == 0);
  REQUIRE(edge.second == 6);
}
