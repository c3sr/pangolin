#pragma once

#include <vector>
#include <string>
#include <istream>

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
public:
  typedef std::vector<Edge> container_type;
  typedef container_type::value_type value_type;
  typedef container_type::iterator iterator;
  typedef container_type::const_iterator const_iterator;

  iterator begin() noexcept
  {
    return edges_.begin();
  }
  const_iterator begin() const noexcept
  {
    return edges_.begin();
  }

  iterator end() noexcept
  {
    return edges_.end();
  }
  const_iterator end() const noexcept
  {
    return edges_.end();
  }

  void push_back(const Edge &e)
  {
    edges_.push_back(e);
  }

  size_t size() const noexcept
  {
    return edges_.size();
  }

  // create edgelist from a tsv file
  static EdgeList read_tsv(const std::string &path);
  // create edgelist from an istream, reading until end
  static EdgeList read_tsv(std::istream &is, std::istream::streampos end = -1);

private:
  std::vector<Edge> edges_;
};
