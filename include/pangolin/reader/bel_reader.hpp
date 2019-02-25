#pragma once

/*
Reader for Binary edge list format
each edge is 8 bytes integer little-endian of dst, src, weight
*/

#include <cassert>
#include <cstring>

#include "pangolin/reader/edge_list_reader.hpp"

namespace pangolin {

class BELReader : public EdgeListReader {

private:
  FILE *fp_;
  std::string path_;

public:
  BELReader(const std::string &path);
  ~BELReader() override;

  virtual EdgeListReader *clone() override;

  virtual size_t read(Edge *ptr, const size_t num) override;

  size_t size();
};

} // namespace pangolin