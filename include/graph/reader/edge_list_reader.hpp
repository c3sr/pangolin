#pragma once

#include <iterator>
#include <string>
#include <cassert>

#include "graph/edge_list.hpp"

namespace graph
{

class EdgeListReader
{

public:
  class iterator
  {
    friend class EdgeListReader;

  private:
    EdgeListReader *reader_;
    Edge edge_;

  public:
    iterator(EdgeListReader *reader) : reader_(reader) {}
    iterator(const iterator &i) : edge_(i.edge_)
    {
      reader_ = reader_->clone();
    }

    const Edge &operator*()
    {
      assert(reader_);
      return edge_;
    }

    const Edge *operator->()
    {
      assert(reader_);
      return &edge_;
    }

    iterator operator++(int) // postfix++
    {
      iterator i(*this);
      ++i;
      return i;
    }
    iterator &operator++() // ++prefix
    {
      reader_->read(&edge_, 1);
      return *this;
    }
  };

  iterator begin()
  {
    return iterator(this);
  }
  iterator end()
  {
    return iterator(nullptr);
  }
  virtual ~EdgeListReader() {}

  // return a deep copy of the edge list reader
  virtual EdgeListReader *clone() = 0;

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