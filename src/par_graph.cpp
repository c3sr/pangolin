#include "pangolin/par_graph.hpp"

#include <algorithm>
#include <map>

#define __TRI_SANITY_CHECK

namespace pangolin {

ParGraph ParGraph::from_edges(const std::set<Edge> &local, const std::set<Edge> &remote) {
  EdgeList localList, remoteList;
  for (const auto &e : local) {
    localList.push_back(e);
  }
  for (const auto &e : remote) {
    remoteList.push_back(e);
  }
  return from_edges(localList, remoteList);
}

ParGraph ParGraph::from_edges(const EdgeList &local, const EdgeList &remote) {
  // sort local and remove
  EdgeList sortedLocal = local;
  EdgeList sortedRemote = remote;

  // rename node ids to be consecutive 0 -> n-1
  // order by src in local and remote and then dst in local and remote
  LOG(debug, "building rename map");
  std::map<Int, Int> rename;
  Int nextId = 0;
  for (const auto &e : local) {
    if (0 == rename.count(e.first)) {
      rename[e.first] = nextId++;
    }
  }
  for (const auto &e : remote) {
    if (0 == rename.count(e.first)) {
      rename[e.first] = nextId++;
    }
  }
  for (const auto &e : local) {
    if (0 == rename.count(e.second)) {
      rename[e.second] = nextId++;
    }
  }
  for (const auto &e : remote) {
    if (0 == rename.count(e.second)) {
      rename[e.second] = nextId++;
    }
  }

  // apply rename operation
  LOG(debug, "renaming");
  for (auto &e : sortedLocal) {
    e.first = rename[e.first];
    e.second = rename[e.second];
  }
  for (auto &e : sortedRemote) {
    e.first = rename[e.first];
    e.second = rename[e.second];
  }

  // sort local and remote by src id
  LOG(debug, "sorting");
  std::sort(sortedLocal.begin(), sortedLocal.end(), [&](const Edge &a, const Edge &b) {
    if (a.first == b.first) {
      return (a.second < b.second);
    }
    return a.first < b.first;
  });
  std::sort(sortedRemote.begin(), sortedRemote.end(), [&](const Edge &a, const Edge &b) {
    if (a.first == b.first) {
      return (a.second < b.second);
    }
    return a.first < b.first;
  });

  ParGraph graph;

  // build graph from local and remote edges
  // proceed from both lists in src edge order
  auto li = sortedLocal.begin();
  auto ri = sortedRemote.begin();
  const auto le = sortedLocal.end();
  const auto re = sortedRemote.end();
  Uint maxDst = 0; // there may be nodes that have no outgoing edges, so we have
                   // to track them to fill out the row
  while ((li != le) || (ri != re)) {
    bool edgeIsLocal;
    Edge edge;
    if (li == le) // no more local edges
    {
      edge = *ri;
      edgeIsLocal = false;
      ++ri;
    } else if (ri == re) // no more remote edges
    {
      edge = *li;
      edgeIsLocal = true;
      ++li;
    } else if (*li < *ri) // local edge comes first
    {
      edge = *li;
      edgeIsLocal = true;
      ++li;
    } else // remote edge is next
    {
      edge = *ri;
      edgeIsLocal = false;
      ++ri;
    }

    maxDst = std::max(edge.second, maxDst);

    SPDLOG_TRACE(logger::console(), "edge {} -> {} local={}", edge.first, edge.second, edgeIsLocal);
    if (graph.rowStarts_.size() != edge.first + 1) {
      SPDLOG_TRACE(logger::console(), "new row {} at {}", edge.first, graph.nonZeros_.size());
      assert(graph.rowStarts_.size() == edge.first);
      graph.rowStarts_.push_back(graph.nonZeros_.size());
    }

    graph.nonZeros_.push_back(edge.second);
    graph.isLocalNonZero_.push_back(edgeIsLocal);
  }

  // fill up to maxDst
  while (graph.rowStarts_.size() < maxDst + 1) {
    SPDLOG_TRACE(logger::console(), "adding node {} with 0 out degree", graph.rowStarts_.size());
    graph.rowStarts_.push_back(graph.nonZeros_.size());
  }

  graph.rowStarts_.push_back(graph.nonZeros_.size());
  SPDLOG_TRACE(logger::console(), "final rowStarts length is {}", graph.rowStarts_.size());

#ifdef __TRI_SANITY_CHECK
  assert(graph.isLocalNonZero_.size() == graph.nonZeros_.size());
#endif

  return graph;
}

std::vector<ParGraph> ParGraph::partition_nonzeros(const size_t numParts) const {
  size_t targetNumNonZeros = (nnz() + numParts - 1) / numParts;
  LOG(debug, "partitioning into {} graphs with nnz ~= {}", numParts, targetNumNonZeros);
  std::vector<ParGraph> graphs;

  // Iterate over edges

  std::set<Edge> localSet, remoteSet;
  for (Int u = 0; u < rowStarts_.size(); ++u) {
    Int vStart = rowStarts_[u];
    Int vEnd = rowStarts_[u + 1];
    for (Int vOff = vStart; vOff < vEnd; ++vOff) {
      // Add the edge to the local set
      Int v = nonZeros_[vOff];

      localSet.insert(Edge(u, v));

      // all outgoing edges from u and v need to be in the remote set
      for (Int dstOff = vStart; dstOff < vEnd; ++dstOff) {
        Int dst = nonZeros_[dstOff];
        Edge remoteEdge(u, dst);
        if (!localSet.count(remoteEdge)) {
          remoteSet.insert(Edge(u, dst));
        }
      }
      Int dstStart = rowStarts_[v];
      Int dstEnd = rowStarts_[v + 1];
      for (Int dstOff = dstStart; dstOff < dstEnd; ++dstOff) {
        Int dst = nonZeros_[dstOff];
        remoteSet.insert(Edge(v, dst));
      }

      // if we've reached the target number of edges,
      // create a new graph
      if (localSet.size() == targetNumNonZeros) {
        // If any edge in the remote set is also in the local set, remove it
        for (const auto &e : localSet) {
          if (remoteSet.count(e)) {
            remoteSet.erase(e);
          }
        }

        LOG(debug, "local set with {} edges", localSet.size());
        LOG(debug, "remote set with {} edges", remoteSet.size());
        for (const auto &e : localSet) {
          SPDLOG_TRACE(logger::console(), "local edge {} -> {}", e.first, e.second);
        }
        for (const auto &e : remoteSet) {
          SPDLOG_TRACE(logger::console(), "remote edge {} -> {}", e.first, e.second);
        }

        graphs.push_back(ParGraph::from_edges(localSet, remoteSet));
        localSet.clear();
        remoteSet.clear();
      }
    }
  }

  // we've gone through all the edges, if there is something in the local set,
  // we need to add it
  if (!localSet.empty()) {
    // If any edge in the remote set is also in the local set, remove it
    for (const auto &e : localSet) {
      if (remoteSet.count(e)) {
        remoteSet.erase(e);
      }
    }

    LOG(debug, "local set with {} edges", localSet.size());
    LOG(debug, "remote set with {} edges", remoteSet.size());
    for (const auto &e : localSet) {
      SPDLOG_TRACE(logger::console(), "local edge {} -> {}", e.first, e.second);
    }
    for (const auto &e : remoteSet) {
      SPDLOG_TRACE(logger::console(), "remote edge {} -> {}", e.first, e.second);
    }

    graphs.push_back(ParGraph::from_edges(localSet, remoteSet));
  }

  assert(graphs.size() == numParts);
  return graphs;
}

#undef __TRI_SANITY_CHECK

} // namespace pangolin