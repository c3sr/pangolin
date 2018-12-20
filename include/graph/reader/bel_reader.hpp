#pragma once

/*
Reader for Binary edge list format
each edge is 8 bytes integer little-endian of dst, src, weight
*/

#include <iterator>
#include <cstring>
#include <sstream>
#include <fstream>
#include <cassert>

#include "graph/reader/edge_list_reader.hpp"

class BELIterator : public std::iterator<std::input_iterator_tag, Edge>
{
private:
  std::istream *is_;
  Edge value_;

  void read_next_value()
  {
    char buffer[24];
    std::string line;
    assert(is_ != nullptr);
    is_->read(buffer, 24);
    assert(is_->gcount() == 8);

    int64_t src64, dst64;
    std::memcpy(&src64, &buffer[0], 8);
    std::memcpy(&dst64, &buffer[8], 8);

    // no characters extracted or parsing error
    if (is_->gcount() != 8)
    {
      is_ = nullptr;
    }
    else
    {
      value_ = Edge(src64, dst64);
    }
  }

public:
  BELIterator() : is_(nullptr) {}
  BELIterator(std::istream &is) : is_(&is)
  {
    // immediately read a value so *iterator.begin() returns the first Edge
    assert(is_ != nullptr);
    read_next_value();
  }
  const Edge &operator*() const
  {
    assert(is_ != nullptr);
    return value_;
  }
  const Edge *operator->() const
  {
    assert(is_ != nullptr);
    return &value_;
  }

  BELIterator &operator++()
  {
    assert(is_ != nullptr);
    if (is_->eof())
    {
      is_ = nullptr;
      return *this;
    }

    read_next_value();

    return *this;
  }

  bool operator!=(const BELIterator &other) const
  {
    return is_ != other.is_;
  }

  bool operator==(const BELIterator &other) const
  {
    return !(*this != other);
  }
};

class BELReader : public EdgeListReader<BELIterator>
{

private:
  std::string path_;
  std::ifstream is_;

public:
  BELReader(const std::string &path);

  virtual BELIterator begin() override;
  virtual BELIterator end() override;

  size_t size();
};
