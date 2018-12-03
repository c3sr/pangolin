#include "graph/dag2019.hpp"

DAG2019 DAG2019::from_edgelist(EdgeList &l)
{
    DAG2019 dag;

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
    const auto smallest = l.begin()->first;
    LOG(debug, "smallest node was {}", smallest);
    for (auto &e : l)
    {
        e.first -= smallest;
        e.second -= smallest;
    }

    for (const auto edge : l)
    {

        assert(edge.first >= 0);
        assert(edge.second >= 0);

        if (dag.nodes_.size() != size_t(edge.first + 1))
        {
            assert(edge.first == dag.nodes_.size());
            // LOG(trace, "node {} edges start at {}", edge.src_, dag.edgeSrc_.size());
            dag.nodes_.push_back(dag.edgeSrc_.size());
        }

        // convert to directed graph by only saving one direction of edges
        if (edge.first < edge.second)
        {
            dag.edgeSrc_.push_back(edge.first);
            dag.edgeDst_.push_back(edge.second);
            // LOG(trace, "added edge {} ({} -> {})", dag.num_edges() - 1, edge.src_, edge.dst_);
        }
    }
    dag.nodes_.push_back(dag.edgeSrc_.size());
    // LOG(trace, "final node idx {} points to {} ", dag.nodes_.size() - 1, dag.edgeSrc_.size());

    // // check that all nodes point to an edge or one past the end of the edge arrays
    // for (const auto n : dag.nodes_)
    // {
    //     assert(n <= dag.edgeSrc_.size());
    //     assert(n >= 0);
    // }

    // //check that all edges have a src and dst which is a valid node
    // for (const auto s : dag.edgeSrc_)
    // {
    //     assert(s < dag.nodes_.size());
    //     assert(s >= 0);
    // }
    // for (const auto d : dag.edgeDst_)
    // {
    //     if (d >= dag.nodes_.size())
    //     {
    //         LOG(critical, "edge dst {} is larger than the largest node {}", d, dag.nodes_.size());
    //     }
    //     assert(d < dag.nodes_.size());
    //     assert(d >= 0);
    // }

    return dag;
}