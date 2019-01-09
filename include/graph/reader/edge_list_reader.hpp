#pragma once

#include <iterator>
#include <string>

#include "graph/edge_list.hpp"

namespace graph
{

class EdgeListReader
{

public:
  virtual ~EdgeListReader() {}

  // read some number of edges from the file
  virtual size_t read(Edge *ptr, const size_t num) = 0;
  void read(EdgeList &edgeList, const size_t num)
  {
    edgeList.resize(num);
    const size_t numRead = read(edgeList.data(), num);
    edgeList.resize(numRead);
  }

  // construct an edge list reader based on the file type of path
  static EdgeListReader *from_file(const std::string &path);
};

} // namespace graph