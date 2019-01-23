#pragma once

/*
Reader for GraphChallenge format TSV files.
Each line should be ASCII tab-separated dst, src, weight,
sorted with increasing src, and within each src increasing dst
*/

#include <iterator>
#include <string>
#include <sstream>
#include <fstream>
#include <cassert>
#include <cstdio>

#include "pangolin/reader/edge_list_reader.hpp"

namespace pangolin
{
class TSVIterator : public std::iterator<std::input_iterator_tag, Edge>
{
private:
  std::istream *is_;
  Edge value_;

  void read_next_value()
  {
    std::string line;
    assert(is_ != nullptr);
    std::getline(*is_, line);
    std::istringstream iss(line);

    int64_t src64, dst64;
    iss >> dst64;
    iss >> src64;

    // no characters extracted or parsing error
    if (iss.fail())
    {
      is_ = nullptr;
    }
    else
    {
      value_ = Edge(src64, dst64);
    }
  }

public:
  TSVIterator() : is_(nullptr) {}
  TSVIterator(std::istream &is) : is_(&is)
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

  TSVIterator &operator++()
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

  bool operator!=(const TSVIterator &other) const
  {
    return is_ != other.is_;
  }

  bool operator==(const TSVIterator &other) const
  {
    return !(*this != other);
  }
};

class GraphChallengeTSVReader : public EdgeListReader
{

private:
  FILE *fp_;
  std::string path_;
  std::ifstream is_;

public:
  GraphChallengeTSVReader(const std::string &path);
  ~GraphChallengeTSVReader() override;

  virtual EdgeListReader *clone() override;

  // read_edges(0, N/2)
  // read_edges(N/2, N)
  /*
    Read edges starting from the first complete edge after start to the
    last complete edge after end. So, a Reader on a file with length N
    would be completely covered like this:
    read_edges(0, N/2)
    read_edges(N/2, N)
    If start is 0, start there
    if end is past EOF, read until EOF
    */
  EdgeList read_edges(size_t start, size_t end);

  // Read all edges
  EdgeList read_edges();

  TSVIterator begin();
  TSVIterator end();

  size_t read(Edge *ptr, const size_t num) override;

  long size();
};

} // namespace pangolin