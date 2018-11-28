#pragma once

#include <vector>
#include <string>

#include "graph/edge_list.hpp"
#include "graph/logger.hpp"

struct VertexGroup
{
    virtual size_t count(const Int &e) const = 0;
    virtual ~VertexGroup() {}
    virtual Int begin() const = 0;
    virtual Int end() const = 0;
};

struct VertexRange : public VertexGroup
{
    Int min_;
    Int max_;

    VertexRange(Int min, Int max) : min_(min), max_(max) {}

    size_t count(const Int &e) const override
    {
        return (e >= min_ && e < max_) ? 1 : 0;
    }
    virtual Int begin() const override { return min_; }
    virtual Int end() const override { return max_; }
};

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

    // get all edges that nodes along edges from srcGroup -> dstGroup participat in.
    // this is more than just srcGroup -> dstGroup edges
    EdgeList get_node_edges(const VertexGroup &srcGroup, const VertexGroup &dstGroup) const;
    std::vector<DAGLowerTriangularCSR> partition(const std::vector<VertexGroup *> &vertexGroups);
};