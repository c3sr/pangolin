#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <algorithm> // std::sort

#include "graph/edge_list.hpp"
#include "graph/logger.hpp"

// Node - the type used to store Node IDs
// Index - the type used to index into the Node IDs
class DAG2019
{
  public:
    std::vector<Int> edgeSrc_; // the node that edge I starts with
    std::vector<Int> edgeDst_; // the node that edge I ends with
    std::vector<Int> nodes_;   // where node i's edges start in edgeSrc and edgeDst

  public:
    DAG2019() {}

    size_t num_nodes() const
    {
        if (nodes_.empty())
        {
            return 0;
        }
        else
        {
            return nodes_.size() - 1;
        }
    }

    size_t num_edges() const
    {
        return edgeSrc_.size();
    }

    static DAG2019 from_edgelist(EdgeList &l);
};