#include <catch2/catch.hpp>

#include <sstream>

#include "pangolin/file/bmtx_stream.hpp"
#include "pangolin/init.hpp"

using namespace pangolin;

static int64_t bitsAsInt(double d) {
  int64_t i;
  std::memcpy(&i, &d, sizeof(i));
  return i;
}

TEST_CASE("BmtxStream") {
  pangolin::init();
  pangolin::logger::set_level(pangolin::logger::Level::TRACE);

  int64_t ints[6] = {1, 7, 3, 1, 7, bitsAsInt(0.0)};
  std::string data((char *)ints, sizeof(ints));
  auto stream = std::make_shared<std::stringstream>(data);

  BmtxStream<std::stringstream> bmtx(stream);
  REQUIRE(bmtx.num_rows() == 1);
  REQUIRE(bmtx.num_cols() == 7);
  REQUIRE(bmtx.nnz() == 3);

  decltype(bmtx)::edge_type edge;
  REQUIRE(bmtx.readEdge(edge));
  REQUIRE(edge.src == 0);
  REQUIRE(edge.dst == 6);
  REQUIRE(edge.val == 0.0);
}
