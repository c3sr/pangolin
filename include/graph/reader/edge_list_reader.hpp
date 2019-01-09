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

  // read some number of edges from the file into buffer
  virtual size_t read(Edge *ptr, const size_t num) = 0;
  EdgeList read(const size_t num)
  {
    EdgeList edgeList(num);
    const size_t numRead = read(edgeList.data(), num);
    edgeList.resize(numRead);
    return edgeList;
  }

  // read all edges from the file
  EdgeList read()
  {
    const size_t bufSize = 10;
    EdgeList edgeList, buf(bufSize);
    while (true)
    {
      const size_t numRead = read(buf.data(), 10);
      if (0 == numRead)
      {
        break;
      }
      edgeList.insert(edgeList.end(), buf.begin(), buf.begin() + numRead);
    }
    return edgeList;
  }

  // construct an edge list reader based on the file type of path
  static EdgeListReader *from_file(const std::string &path);
};

} // namespace graph