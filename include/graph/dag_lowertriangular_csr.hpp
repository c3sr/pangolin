#pragma once

#include <vector>
#include <string>

#include "graph/edge_list.hpp"
#include "graph/logger.hpp"

// Lower Triangular adjacency matrix in CSR format
class DAGLowerTriangularCSR
{
  public:
    // Array of size nvertices+1, where i element equals to the number of the
    // first edge for this vertex in the list of all outgoing edges in the
    // destination_indices array. Last element stores total number of edges
    std::vector<Int> sourceOffsets_;
    // Array of size nedges, where each value designates destination vertex
    // for an edge.
    std::vector<Int> destinationIndices_;

  public:
    DAGLowerTriangularCSR() {}

    size_t num_nodes() const
    {
        if (sourceOffsets_.empty())
        {
            return 0;
        }
        else
        {
            return sourceOffsets_.size() - 1;
        }
    }

    size_t num_edges() const
    {
        return destinationIndices_.size();
    }

    static DAGLowerTriangularCSR from_edgelist(EdgeList &l);
};