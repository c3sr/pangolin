#pragma once

#include <iterator>
#include <string>

#include "graph/edge_list.hpp"

namespace graph
{

// typedef std::iterator<std::input_iterator_tag, Edge> EdgeIterator;

template <typename ITERATOR>
class EdgeListReader
{

public:
  virtual ITERATOR begin() = 0;
  virtual ITERATOR end() = 0;

  static EdgeListReader *from_file(const std::string &path);
};

} // namespace graph