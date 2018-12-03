#pragma once

#include <set>
#include <vector>
#include <string>
#include <fstream>

#include "graph/edge_list.hpp"
#include "graph/logger.hpp"

#define __TRI_SANITY_CHECK

class ParGraph
{
  public:
    std::vector<Int> rowStarts_;
    std::vector<Int> nonZeros_;
    // use unsigned char instead of bool so it's easy to copy to GPU
    std::vector<unsigned char> isLocalNonZero_;

  public:
    ParGraph() {}

    size_t num_rows() const noexcept
    {
        if (rowStarts_.empty())
        {
            return 0;
        }
        else
        {
            assert(rowStarts_.size() > 1);
            return rowStarts_.size() - 1;
        }
    }

    size_t nnz() const noexcept
    {
#ifdef __TRI_SANITY_CHECK
        if (!rowStarts_.empty())
        {
            assert(rowStarts_.back() == nonZeros_.size());
        }
#endif
        return nonZeros_.size();
    }

    static ParGraph from_edges(const EdgeList &local, const EdgeList &remote);
    static ParGraph from_edges(const std::set<Edge> &local, const std::set<Edge> &remote);
    std::vector<ParGraph> partition_nonzeros(const size_t numParts) const;
};

#undef __TRI_SANITY_CHECK