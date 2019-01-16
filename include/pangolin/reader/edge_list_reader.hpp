#pragma once

#include "pangolin/edge_list.hpp"

class EdgeIterator
{
  public:
    virtual Edge operator*() = 0;
    virtual bool operator==(const EdgeIterator &other) = 0;
};

class EdgeListReader
{

  public:
    typedef EdgeIterator iterator;
    virtual iterator begin() = 0;
    virtual iterator end() = 0;
    virtual EdgeList read_edges() = 0;
};