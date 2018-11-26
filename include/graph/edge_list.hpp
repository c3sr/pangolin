#pragma once

#include <vector>
#include <string>

#include "graph/types.hpp"

template <typename T>
class EdgeBase
{
public:
  EdgeBase(const T &src, const T &dst) : src_(src), dst_(dst) {}
  T src_;
  T dst_;
};

typedef EdgeBase<Int> Edge;

class EdgeList
{

private:
  std::vector<Edge> edges_;

  static EdgeList read_tsv(const std::string &path);

  void push_back(const Edge &e)
  {
    edges_.push_back(e);
  }

  size_t size() const
  {
    return edges_.size();
  }
};
