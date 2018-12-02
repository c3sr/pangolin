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
    std::vector<bool> isLocalNonZero_;

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

    size_t num_nodes() const noexcept
    {
        std::set<Int> ids;
        for (Int i = 0; i < rowStarts_.size() - 1; ++i)
        {
            auto start = rowStarts_[i];
            auto end = rowStarts_[i + 1];
            LOG(trace, "node {} has offsets {} {}", i, start, end);
            if (start != end)
            {
                ids.insert(i);
            }
            else
            {
                assert(false);
            }

            assert(start <= nnz());
            assert(end <= nnz());
            assert(end >= start);
            for (Int j = start; j != end; ++j)
            {
                ids.insert(nonZeros_[j]);
            }
        }
        return ids.size();
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
};

#undef __TRI_SANITY_CHECK