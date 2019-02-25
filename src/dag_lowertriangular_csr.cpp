#include "pangolin/dag_lowertriangular_csr.hpp"

#include <map>
#include <set>

namespace pangolin {

DAGLowerTriangularCSR DAGLowerTriangularCSR::from_edgelist(EdgeList &l) {
  DAGLowerTriangularCSR dag;

  SPDLOG_DEBUG(logger::console,
               "Building DAGLowerTriangularCSR from EdgeList with {} edges",
               l.size());

  if (l.size() == 0) {
    return dag;
  }

  // sort the edge list by src, with dst sorted within each src
  // the file should come in this way
  // std::stable_sort(l.begin(), l.end(), [](const Edge &a, const Edge &b) ->
  // bool {
  //     return a.second < b.second;
  // });
  // std::stable_sort(l.begin(), l.end(), [](const Edge &a, const Edge &b) ->
  // bool {
  //     return a.src_ < b.src_;
  // });

  // ensure node IDs are 0 - whatever
  const auto smallest = l.begin()->first;
  SPDLOG_DEBUG(logger::console, "smallest node was {}", smallest);
  for (auto &e : l) {
    e.first -= smallest;
    e.second -= smallest;
  }

  for (const auto &edge : l) {
    // SPDLOG_DEBUG(logger::console, "{} {}", edge.src_, edge.second);
    // a new source node or the first source node.
    // assume this come in in order
    if (dag.sourceOffsets_.size() != size_t(edge.first + 1)) {
      // SPDLOG_DEBUG(logger::console, "new source node {} starts at offset {}",
      // edge.src_, dag.destinationIndices_.size()); node ids should cover all
      // numbers and be increasing
      assert(edge.first == dag.sourceOffsets_.size());
      // mark where the source node's destination indices start
      dag.sourceOffsets_.push_back(dag.destinationIndices_.size());
    }

    // convert to directed graph by only saving one direction of edges
    if (edge.first > edge.second) {
      dag.destinationIndices_.push_back(edge.second);
    }
  }
  dag.sourceOffsets_.push_back(dag.destinationIndices_.size());

  return dag;
}

} // namespace pangolin