#include "graph/dag_lowertriangular_csr.hpp"

#include <map>

DAGLowerTriangularCSR DAGLowerTriangularCSR::from_edgelist(EdgeList &l)
{
    DAGLowerTriangularCSR dag;

    LOG(debug, "Building DAGLowerTriangularCSR from EdgeList with {} edges", l.size());

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

    for (const auto &edge : l)
    {
        // LOG(debug, "{} {}", edge.src_, edge.dst_);
        // a new source node or the first source node.
        // assume this come in in order
        if (dag.sourceOffsets_.size() != size_t(edge.src_ + 1))
        {
            // LOG(debug, "new source node {} starts at offset {}", edge.src_, dag.destinationIndices_.size());
            // node ids should cover all numbers and be increasing
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

EdgeList DAGLowerTriangularCSR::get_node_edges(const VertexGroup &srcGroup, const VertexGroup &dstGroup) const
{

    // for (Int uIdx = 0; uIdx < num_nodes(); ++uIdx)
    // {
    //     const Int u = uIdx;
    //     const Int vStart = sourceOffsets_[u];
    //     const Int vEnd = sourceOffsets_[u + 1];
    //     for (Int vOff = vStart; vOff < vEnd; ++vOff)
    //     {
    //         const Int v = destinationIndices_[vOff];
    //         LOG(trace, "edge {} {}", u, v);
    //     }
    // }

    EdgeList l;

    // loop over u -> v edges
    for (Int uIdx = 0; uIdx < num_nodes(); ++uIdx)
    {
        const Int u = uIdx;
        if (srcGroup.count(u))
        {
            const Int vStart = sourceOffsets_[u];
            const Int vEnd = sourceOffsets_[u + 1];
            for (Int vOff = vStart; vOff < vEnd; ++vOff)
            {
                const Int v = destinationIndices_[vOff];

                // u -> v | u in src && v in dst
                // v is in dst
                if (dstGroup.count(v))
                {
                    LOG(trace, "adding {0} -> {1} because {0} in src -> {1} in dst", u, v);
                    l.push_back(Edge(u, v));
                }
                else if (srcGroup.count(v))
                {

                    // u-> v | u in src v has a tail in dst
                    const int dstStart = sourceOffsets_[v];
                    const int dstEnd = sourceOffsets_[v + 1];
                    for (Int dstOff = dstStart; dstOff != dstEnd; ++dstOff)
                    {
                        const Int dst = destinationIndices_[dstOff];
                        if (dstGroup.count(dst))
                        {
                            LOG(trace, "adding {0} -> {1} because both in src and {1} has a tail in dst", u, v);
                            l.push_back(Edge(u, v));
                            break;
                        }
                    }
                }

                // want all edges v -> w where w is a tail from src
                const int wStart = sourceOffsets_[v];
                const int wEnd = sourceOffsets_[v + 1];
                for (Int wOff = wStart; wOff != wEnd; ++wOff)
                {
                    const Int w = destinationIndices_[wOff];
                    bool found = false;
                    for (auto head = srcGroup.begin(); head != srcGroup.end(); ++head)
                    {
                        if (found)
                            break;
                        const int dstStart = sourceOffsets_[head];
                        const int dstEnd = sourceOffsets_[head + 1];
                        for (Int dstOff = dstStart; dstOff != dstEnd; ++dstOff)
                        {
                            const Int dst = destinationIndices_[dstOff];
                            if (dst == w)
                            {
                                found = true;
                                LOG(trace, "adding {0} -> {1} because {0} is a tail from src and {1} is a tail from src", v, w);
                                l.push_back(Edge(v, w));
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    return l;
}

std::vector<DAGLowerTriangularCSR> DAGLowerTriangularCSR::partition(const std::vector<VertexGroup *> &vertexGroups)
{
    LOG(debug, "Partitioning into {} vertex groups", vertexGroups.size());
    std::vector<DAGLowerTriangularCSR> ret;

    for (const auto &srcGroup : vertexGroups)
    {
        for (const auto &dstGroup : vertexGroups)
        {

            EdgeList origEdges = get_node_edges(*srcGroup, *dstGroup);
            LOG(debug, "edgelist has {} edges", origEdges.size());

            // these edges are only the edges src -> dst where src > dst
            // we need to add back in the dst -> src edges since EdgeList is assumed to have both
            EdgeList flipped;
            for (const auto e : origEdges)
            {
                flipped.push_back(Edge(e.dst_, e.src_));
            }
            for (const auto e : flipped)
            {
                origEdges.push_back(Edge(e.src_, e.dst_));
            }

            // sort by dst, then src to ensure that the new list is in increasing order of src
            std::sort(origEdges.begin(), origEdges.end(),
                      [](const Edge &a, const Edge &b) {
                          return a.dst_ < b.dst_;
                      });
            std::stable_sort(origEdges.begin(), origEdges.end(),
                             [](const Edge &a, const Edge &b) {
                                 return a.src_ < b.src_;
                             });

            // these edges do not necessarily have nodes with contiguous ids of 0 to whatever
            // rename the vertex ids.
            // the edges will already have src > dst
            // we need to preserve that during the renaming or we'll drop edges
            // generate rename by going through dsts first to ensure they will be less than the srcs
            std::map<Int, Int> rename;
            Int nextName = 0;
            for (const auto &e : origEdges)
            {
                if (rename.count(e.dst_) == 0)
                {
                    rename[e.dst_] = nextName++;
                }
            }
            for (const auto &e : origEdges)
            {
                if (rename.count(e.src_) == 0)
                {
                    rename[e.src_] = nextName++;
                }
            }

            // apply the rename to the list
            for (auto &e : origEdges)
            {
                e.src_ = rename[e.src_];
                e.dst_ = rename[e.dst_];
            }

            // sort by dst, then src to ensure that the new list is in increasing order of src
            std::sort(origEdges.begin(), origEdges.end(),
                      [](const Edge &a, const Edge &b) {
                          return a.dst_ < b.dst_;
                      });
            std::stable_sort(origEdges.begin(), origEdges.end(),
                             [](const Edge &a, const Edge &b) {
                                 return a.src_ < b.src_;
                             });

            // for (const auto e : origEdges)
            // {
            //     LOG(debug, "{} {}", e.src_, e.dst_);
            // }

            // build the new dag
            DAGLowerTriangularCSR dag = from_edgelist(origEdges);

            ret.push_back(dag);
        }
    }

    return ret;
}