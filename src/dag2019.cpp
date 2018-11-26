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

    Int currentSrc = l.begin()->src_;

    for (const auto edge : l)
    {

        if (dag.nodes.empty() || (currentSrc != edge.src_))
        {
            LOG(trace, "new src {} starts at {}", edge.src_, dag.edgeSrc.size());
            dag.nodes.push_back(dag.edgeSrc.size());
        }

        // convert to directed graph by only saving one direction of edges
        if (edge.src_ < edge.dst_)
        {
            dag.edgeSrc.push_back(edge.src_);
            dag.edgeDst.push_back(edge.dst_);
            LOG(trace, "added edge {} ({} -> {})", dag.num_edges(), edge.src_, edge.dst_);
        }

        currentSrc = edge.src_;
    }

    dag.nodes.push_back(dag.edgeSrc.size());

    return dag;
}