#pragma once

#include <set>

#include "csr.hpp"
#include "pangolin/logger.hpp"

namespace pangolin {

template <typename Index> CSR<Index>::CSR() {}

template <typename Index> PANGOLIN_HOST PANGOLIN_DEVICE uint64_t CSR<Index>::num_rows() const {
  if (rowPtr_.size() == 0) {
    return 0;
  } else {
    return rowPtr_.size() - 1;
  }
}

template <typename Index> uint64_t CSR<Index>::num_nodes() const {
  std::set<Index> nodes;
  // add all dsts
  for (Index ci = 0; ci < colInd_.size(); ++ci) {
    nodes.insert(colInd_[ci]);
  }
  // add non-zero sources
  for (Index i = 0; i < rowPtr_.size() - 1; ++i) {
    Index row_start = rowPtr_[i];
    Index row_end = rowPtr_[i + 1];
    if (row_start != row_end) {
      nodes.insert(i);
    }
  }
  return nodes.size();
}

template <typename Index>
template <typename EdgeIter>
CSR<Index> CSR<Index>::from_edges(EdgeIter begin, EdgeIter end, std::function<bool(EdgeTy<Index>)> f) {
  CSR<Index> csr;

  if (begin == end) {
    LOG(warn, "constructing from empty edge sequence");
    return csr;
  }

  for (auto ei = begin; ei != end; ++ei) {
    EdgeTy<Index> edge = *ei;
    const Index src = edge.first;
    const Index dst = edge.second;
    SPDLOG_TRACE(logger::console(), "handling edge {}->{}", edge.first, edge.second);

    // edge has a new src and should be in a new row
    // even if the edge is filtered out, we need to add empty rows
    while (csr.rowPtr_.size() != size_t(src + 1)) {
      // expecting inputs to be sorted by src, so it should be at least
      // as big as the current largest row we have recored
      assert(src >= csr.rowPtr_.size() && "are edges not ordered by source?");
      SPDLOG_TRACE(logger::console(), "node {} edges start at {}", edge.first, csr.rowPtr_.size());
      csr.rowPtr_.push_back(csr.colInd_.size());
    }

    if (f(edge)) {
      csr.colInd_.push_back(dst);
    } else {
      continue;
    }
  }

  // add the final length of the non-zeros to the offset array
  csr.rowPtr_.push_back(csr.colInd_.size());
  return csr;
}

template <typename Index> CSRView<Index> CSR<Index>::view() const {
  CSRView<Index> view;
  view.nnz_ = nnz();
  view.num_rows_ = num_rows();
  view.rowPtr_ = rowPtr_.data();
  view.colInd_ = colInd_.data();
  return view;
}

template <typename Index> void CSR<Index>::read_mostly(const int dev) {
  rowPtr_.read_mostly(dev);
  colInd_.read_mostly(dev);
}

template <typename Index> void CSR<Index>::accessed_by(const int dev) {
  rowPtr_.accessed_by(dev);
  colInd_.accessed_by(dev);
}

template <typename Index> void CSR<Index>::prefetch_async(const int dev) {
  rowPtr_.prefetch_async(dev);
  colInd_.prefetch_async(dev);
}

} // namespace pangolin
