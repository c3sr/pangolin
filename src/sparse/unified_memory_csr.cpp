#include "pangolin/sparse/unified_memory_csr.hpp"
#include "pangolin/logger.hpp"
#include <cassert>
#include <set>

namespace pangolin {

/*
Expect the incoming edge list to be sorted in increasing order of src.
Within each src, dst should be in increasing order
src should also be < dst
there should be no duplicate edges in local and remote
*/
UnifiedMemoryCSR UnifiedMemoryCSR::from_sorted_edgelist(const EdgeList &local, const EdgeList &remote) {
  UnifiedMemoryCSR csr;

  if (local.empty() && remote.empty()) {
    LOG(warn, "building empty UnifiedMemoryCSR");
    return csr;
  }

  // order two edges
  auto compareEdge = [](const Edge &a, const Edge &b) -> bool {
    if (a.first == b.first) {
      return a.second < b.second;
    }
    return a.first < b.first;
  };

  auto minEdge = [&](const Edge &a, const Edge &b) -> const Edge & { return compareEdge(a, b) ? a : b; };

  // smallest src edge
  Uint firstRow;
  if (local.empty()) {
    firstRow = remote.begin()->first;
  } else if (remote.empty()) {
    firstRow = local.begin()->first;
  } else {
    firstRow = minEdge(*local.begin(), *remote.begin()).first;
  }

  // add empty rows until firstRow
  LOG(debug, "smallest row was {}", firstRow);
  for (Uint i = 0; i < firstRow; ++i) {
    SPDLOG_TRACE(logger::console(), "added empty row {} before smallest row id", i);
    csr.rowOffsets_.push_back(0);
  }

  Uint maxNode = 0;

  auto li = local.begin();
  auto ri = remote.begin();
  auto le = local.end();
  auto re = remote.end();
  char edgeIsLocal;
  Edge e;
  while (li != local.end() || ri != remote.end()) {
    if (li == le) // no more local edges
    {
      edgeIsLocal = false;
      e = *ri++;
    } else if (ri == re) // no more remote edges
    {
      edgeIsLocal = true;
      e = *li++;
    } else if (compareEdge(*li, *ri)) // local is next
    {
      edgeIsLocal = true;
      e = *li++;
    } else // remote is next
    {
      edgeIsLocal = false;
      e = *ri++;
    }

    const Uint rowIdx = e.first;
    maxNode = std::max(e.first, maxNode);
    maxNode = std::max(e.second, maxNode);

    // add the starting offset of the new row
    while (csr.rowOffsets_.size() < rowIdx + 1) {
      csr.rowOffsets_.push_back(csr.data_.size());
    }

    // add the row destination
    csr.data_.push_back(e.second);
    csr.dataIsLocal_.push_back(edgeIsLocal);
  }

  // add final nodes with 0 out-degree
  LOG(debug, "max node id was {}", maxNode);
  while (csr.rowOffsets_.size() < maxNode + 1) {
    csr.rowOffsets_.push_back(csr.data_.size());
  }

  // add  the last entry to give a length on the last row
  csr.rowOffsets_.push_back(csr.data_.size());

  LOG(debug, "rowOffsets is length {}", csr.rowOffsets_.size());
  LOG(debug, "data is length {}", csr.data_.size());

  return csr;
}

std::vector<UnifiedMemoryCSR> UnifiedMemoryCSR::partition_nonzeros(const size_t numPartitions) const {
  LOG(debug, "paritioning into {} graphs", numPartitions);
  std::vector<std::set<Edge>> localEdges(numPartitions);  // local edges for each partition
  std::vector<std::set<Edge>> remoteEdges(numPartitions); // remote edges for each partitions

  const uint64_t nnzPerPartition = (nnz() + numPartitions - 1) / numPartitions;
  LOG(debug, "targeting {} nnz per partition", nnzPerPartition);

  // evenly distribute local edges
  size_t currentPartitionIdx = 0;
  for (size_t si = 0; si < rowOffsets_.size(); ++si) {
    Uint head = si;
    Uint tailOffsetBegin = rowOffsets_[si];
    Uint tailOffsetEnd = rowOffsets_[si + 1];
    for (Uint tailOffset = tailOffsetBegin; tailOffset < tailOffsetEnd; ++tailOffset) {
      Uint tail = data_[tailOffset];
      Edge e(head, tail);
      assert(currentPartitionIdx < localEdges.size());
      localEdges[currentPartitionIdx].insert(std::move(e));
      SPDLOG_TRACE(logger::console(), "added local edge {} {} to partition {}", e.first, e.second, currentPartitionIdx);
      if (localEdges[currentPartitionIdx].size() >= nnzPerPartition) {
        ++currentPartitionIdx;
      }
    }
  }

  // for each set of local edges, add all remote edges, which are neighbors of
  // the tail if that edge is not a local edge
  for (size_t i = 0; i < localEdges.size(); ++i) {
    auto &local = localEdges[i];
    auto &remote = remoteEdges[i];

    for (auto &e : local) {
      for (Uint off = rowOffsets_[e.second]; off < rowOffsets_[e.second + 1]; ++off) {
        Edge tailEdge(e.second, data_[off]);

        if (0 == local.count(tailEdge)) {
          SPDLOG_TRACE(logger::console(), "adding remote edge {} {} to parition {}", tailEdge.first, tailEdge.second,
                       i);
          remote.insert(tailEdge);
        } else {
          SPDLOG_TRACE(logger::console(), "edge {} {} is already local in parition {}", tailEdge.first, tailEdge.second,
                       i);
        }
      }
    }
  }

  std::vector<UnifiedMemoryCSR> ret(numPartitions); // partitions to be returned
  // convert to vectors and build CSR
  for (size_t i = 0; i < localEdges.size(); ++i) {
    auto &localSet = localEdges[i];
    auto &remoteSet = remoteEdges[i];

    SPDLOG_TRACE(logger::console(), "building CSR from {} local and {} remote edges", localSet.size(),
                 remoteSet.size());

    EdgeList localList(localSet.begin(), localSet.end());
    EdgeList remoteList(remoteSet.begin(), remoteSet.end());

    std::sort(localList.begin(), localList.end(), [](const Edge &a, const Edge &b) -> bool {
      if (a.first == b.first) {
        return a.second < b.second;
      }
      return a.first < b.first;
    });

    std::sort(remoteList.begin(), remoteList.end(), [](const Edge &a, const Edge &b) -> bool {
      if (a.first == b.first) {
        return a.second < b.second;
      }
      return a.first < b.first;
    });

    ret[i] = UnifiedMemoryCSR::from_sorted_edgelist(localList, remoteList);
  }
  return ret;
}

} // namespace pangolin