#include "graph/par_graph.hpp"

#include <algorithm>
#include <map>

#define __TRI_SANITY_CHECK

ParGraph ParGraph::from_edges(const EdgeList &local, const EdgeList &remote)
{
    // sort local and remove
    std::vector<Edge> sortedLocal = local.edges_;
    std::vector<Edge> sortedRemote = remote.edges_;

    // rename node ids to be consecutive 0 -> n-1
    // order by src in local and remote and then dst in local and remote
    LOG(debug, "building rename map");
    std::map<Int, Int> rename;
    Int nextId = 0;
    for (const auto &e : local)
    {
        if (0 == rename.count(e.src_))
        {
            rename[e.src_] = nextId++;
        }
    }
    for (const auto &e : remote)
    {
        if (0 == rename.count(e.src_))
        {
            rename[e.src_] = nextId++;
        }
    }
    for (const auto &e : local)
    {
        if (0 == rename.count(e.dst_))
        {
            rename[e.dst_] = nextId++;
        }
    }
    for (const auto &e : remote)
    {
        if (0 == rename.count(e.dst_))
        {
            rename[e.dst_] = nextId++;
        }
    }

    // apply rename operation
    LOG(debug, "renaming");
    for (auto &e : sortedLocal)
    {
        e.src_ = rename[e.src_];
        e.dst_ = rename[e.dst_];
    }
    for (auto &e : sortedRemote)
    {
        e.src_ = rename[e.src_];
        e.dst_ = rename[e.dst_];
    }

    // sort local and remote by src id
    LOG(debug, "sorting");
    std::sort(sortedLocal.begin(), sortedLocal.end(), [&](const Edge &a, const Edge &b) {
        if (a.src_ == b.src_)
        {
            return (a.dst_ < b.dst_);
        }
        return a.src_ < b.src_;
    });
    std::sort(sortedRemote.begin(), sortedRemote.end(), [&](const Edge &a, const Edge &b) {
        if (a.src_ == b.src_)
        {
            return (a.dst_ < b.dst_);
        }
        return a.src_ < b.src_;
    });

    ParGraph graph;

    // build graph from local and remote edges
    // proceed from both lists in src edge order
    auto li = sortedLocal.begin();
    auto ri = sortedRemote.begin();
    const auto le = sortedLocal.end();
    const auto re = sortedRemote.end();
    Int maxDst = -1; // there may be nodes that have no outgoing edges, so we have to track them to fill out the row
    while ((li != le) || (ri != re))
    {
        bool edgeIsLocal;
        Edge edge;
        if (li == le) // no more local edges
        {
            edge = *ri;
            edgeIsLocal = false;
            ++ri;
        }
        else if (ri == re) // no more remote edges
        {
            edge = *li;
            edgeIsLocal = true;
            ++li;
        }
        else if (*li < *ri) // local edge comes first
        {
            edge = *li;
            edgeIsLocal = true;
            ++li;
        }
        else // remote edge is next
        {
            edge = *ri;
            edgeIsLocal = false;
            ++ri;
        }

        maxDst = std::max(edge.dst_, maxDst);

        LOG(trace, "edge {} -> {} local={}", edge.src_, edge.dst_, edgeIsLocal);
        if (graph.rowStarts_.size() != edge.src_ + 1)
        {
            LOG(trace, "new row {} at {}", edge.src_, graph.nonZeros_.size());
            assert(graph.rowStarts_.size() == edge.src_);
            graph.rowStarts_.push_back(graph.nonZeros_.size());
        }

        graph.nonZeros_.push_back(edge.dst_);
        graph.isLocalNonZero_.push_back(edgeIsLocal);
    }

    // fill up to maxDst
    while (graph.rowStarts_.size() < maxDst + 1)
    {
        LOG(trace, "adding node {} with 0 out degree", graph.rowStarts_.size());
        graph.rowStarts_.push_back(graph.nonZeros_.size());
    }

    graph.rowStarts_.push_back(graph.nonZeros_.size());
    LOG(trace, "final rowStarts length is {}", graph.rowStarts_.size());

#ifdef __TRI_SANITY_CHECK
    assert(graph.isLocalNonZero_.size() == graph.nonZeros_.size());
#endif

    return graph;
}

#undef __TRI_SANITY_CHECK