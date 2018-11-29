#include "graph/dag_lowertriangular_csr.hpp"

#include <map>
#include <set>

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

    // collect all nodes in edges between nodes in srcGroup and nodes in dstGroup
    std::set<Int> nodes;
    for (Int u = 0; u < num_nodes(); ++u)
    {

        const Int vStart = sourceOffsets_[u];
        const Int vEnd = sourceOffsets_[u + 1];
        for (Int vOff = vStart; vOff < vEnd; ++vOff)
        {
            const Int v = destinationIndices_[vOff];
            if (srcGroup.count(u) && dstGroup.count(v))
            {
                nodes.insert(u);
                nodes.insert(v);
            }
        }
    }

    // collect all edges that are between those nodes
    EdgeList edges;
    for (Int u = 0; u < num_nodes(); ++u)
    {
        if (nodes.count(u))
        {
            const Int vStart = sourceOffsets_[u];
            const Int vEnd = sourceOffsets_[u + 1];
            for (Int vOff = vStart; vOff < vEnd; ++vOff)
            {
                const Int v = destinationIndices_[vOff];
                if (nodes.count(v))
                {
                    edges.push_back(Edge(u, v));
                }
            }
        }
    }

    // also add edges vSrc -> vn ->vDst for vSrc != vDst
    if (&srcGroup != &dstGroup)
    { // FIXME, actually compare probably

        for (Int u = 0; u < num_nodes(); ++u)
        {
            if (srcGroup.count(u)) // vSrc -> ...
            {
                const Int nStart = sourceOffsets_[u];
                const Int nEnd = sourceOffsets_[u + 1];
                for (Int nOff = nStart; nOff < nEnd; ++nOff)
                {
                    const Int n = destinationIndices_[nOff]; // vSrc -> vn

                    const Int vStart = sourceOffsets_[n];
                    const Int vEnd = sourceOffsets_[n + 1];
                    for (Int vOff = vStart; vOff < vEnd; ++vOff)
                    {
                        const Int v = destinationIndices_[vOff]; // vSrc -> vn -> vDst
                        if (srcGroup.count(u) && dstGroup.count(v))
                        {
                            edges.push_back(Edge(u, n));
                            edges.push_back(Edge(n, v));
                        }
                    }
                }
            }
        }
    }

    return edges;
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
            LOG(debug, "adding flipped edges");
            EdgeList flipped;
            for (const auto e : origEdges)
            {
                assert(e.src_ > e.dst_);
                flipped.push_back(Edge(e.dst_, e.src_));
            }
            for (const auto e : flipped)
            {
                origEdges.push_back(Edge(e.src_, e.dst_));
            }

            // sort by dst, then src to ensure that the new list is in increasing order of src
            // LOG(debug, "sorting");
            // std::sort(origEdges.begin(), origEdges.end(),
            //           [](const Edge &a, const Edge &b) {
            //               return a.dst_ < b.dst_;
            //           });
            // std::stable_sort(origEdges.begin(), origEdges.end(),
            //                  [](const Edge &a, const Edge &b) {
            //                      return a.src_ < b.src_;
            //                  });

            // these edges do not necessarily have nodes with contiguous ids of 0 to whatever
            // rename the vertex ids.
            // the edges will already have src > dst
            // we need to preserve that during the renaming or we'll drop edges
            // generate rename by going through dsts first to ensure they will be less than the srcs

            LOG(debug, "building rename map");
            std::map<Int, Int> rename;
            Int nextName = 0;
            for (const auto &e : origEdges)
            {
                if (rename.count(e.src_) == 0)
                {
                    rename[e.src_] = nextName++;
                }
            }
            for (const auto &e : origEdges)
            {
                if (rename.count(e.dst_) == 0)
                {
                    rename[e.dst_] = nextName++;
                }
            }

            // apply the rename to the list
            LOG(debug, "renaming");
            for (auto &e : origEdges)
            {
                e.src_ = rename[e.src_];
                e.dst_ = rename[e.dst_];
            }

            // sort by dst, then src to ensure that the new list is in increasing order of src
            LOG(debug, "sorting");
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