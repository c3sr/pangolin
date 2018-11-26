#include "graph/dag2019.hpp"

DAG2019 DAG2019::from_edgelist(EdgeList &l)
{
    DAG2019 dag;

    if (l.size() == 0)
    {
        return dag;
    }

    // sort the edge list by src
    std::sort(l.begin(), l.end(), [](const Edge &a, const Edge &b) -> bool {
        return a.src_ < b.src_;
    });

    // ensure node IDs are 0 - whatever
    const auto smallest = l.begin()->src_;
    LOG(trace, "smallest node was {}", smallest);
    for (auto &e : l)
    {
        e.src_ -= smallest;
        e.dst_ -= smallest;
    }

    Int currentSrc = l.begin()->src_;

    for (const auto edge : l)
    {

        if (dag.nodes_.empty() || (currentSrc != edge.src_))
        {
            LOG(trace, "node {} edges start at {}", edge.src_, dag.edgeSrc_.size());
            dag.nodes_.push_back(dag.edgeSrc_.size());
        }

        // convert to directed graph by only saving one direction of edges
        if (edge.src_ < edge.dst_)
        {
            dag.edgeSrc_.push_back(edge.src_);
            dag.edgeDst_.push_back(edge.dst_);
            LOG(trace, "added edge {} ({} -> {})", dag.num_edges() - 1, edge.src_, edge.dst_);
        }

        currentSrc = edge.src_;
    }
    dag.nodes_.push_back(dag.edgeSrc_.size());
    LOG(trace, "final node idx {} points to {} ", dag.nodes_.size() - 1, dag.edgeSrc_.size());

    // check that all nodes point to an edge or one past the end of the edge arrays
    for (const auto n : dag.nodes_)
    {
        assert(n <= dag.edgeSrc_.size());
        assert(n >= 0);
    }

    //check that all edges have a src and dst which is a valid node
    for (const auto s : dag.edgeSrc_)
    {
        assert(s < dag.nodes_.size());
        assert(s >= 0);
    }
    for (const auto d : dag.edgeDst_)
    {
        if (d >= dag.nodes_.size())
        {
            LOG(critical, "edge dst {} is larger than the largest node {}", d, dag.nodes_.size());
        }
        assert(d < dag.nodes_.size());
        assert(d >= 0);
    }

    return dag;
}