#include "graph/dag_lowertriangular_csr.hpp"

DAGLowerTriangularCSR DAGLowerTriangularCSR::from_edgelist(EdgeList &l)
{
    DAGLowerTriangularCSR dag;

    if (l.size() == 0)
    {
        return dag;
    }

    // sort the edge list by src, with dst sorted within each src
    // the file should come in this way
    // std::stable_sort(l.begin(), l.end(), [](const Edge &a, const Edge &b) -> bool {
    //     return a.dst_ < b.dst_;
    // });
    // std::stable_sort(l.begin(), l.end(), [](const Edge &a, const Edge &b) -> bool {
    //     return a.src_ < b.src_;
    // });

    // ensure node IDs are 0 - whatever
    const auto smallest = l.begin()->src_;
    LOG(debug, "smallest node was {}", smallest);
    for (auto &e : l)
    {
        e.src_ -= smallest;
        e.dst_ -= smallest;
    }

    for (const auto edge : l)
    {

        // a new source node or the first source node.
        // assume this come in in order
        if (dag.sourceOffsets_.size() != size_t(edge.src_ + 1))
        {
            assert(edge.src_ > dag.sourceOffsets_.back());
            assert(edge.src_ == dag.sourceOffsets_.size());
            // mark where the source node's destination indices start
            dag.sourceOffsets_.push_back(dag.destinationIndices_.size());
        }

        // convert to directed graph by only saving one direction of edges
        if (edge.src_ > edge.dst_)
        {
            dag.destinationIndices_.push_back(edge.dst_);
        }
    }
    dag.sourceOffsets_.push_back(dag.destinationIndices_.size());

    return dag;
}