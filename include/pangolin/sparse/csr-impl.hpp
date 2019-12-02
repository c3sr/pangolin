#pragma once

#include <set>

#include "csr.hpp"
#include "pangolin/logger.hpp"

namespace pangolin {

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

  bool acceptedEdges = false;

  for (auto ei = begin; ei != end; ++ei) {
    EdgeTy<Index> edge = *ei;
    if (f(edge)) {
      acceptedEdges = true;
      csr.add_next_edge(edge);
    }
  }

  if (acceptedEdges) {
    csr.finish_edges();
  }

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

template <typename Index> PANGOLIN_HOST void CSR<Index>::read_mostly() {
  rowPtr_.read_mostly();
  colInd_.read_mostly();
}

template <typename Index> PANGOLIN_HOST void CSR<Index>::accessed_by(const int dev) {
  rowPtr_.accessed_by(dev);
  colInd_.accessed_by(dev);
}

template <typename Index> PANGOLIN_HOST void CSR<Index>::prefetch_async(const int dev, cudaStream_t stream) {
  rowPtr_.prefetch_async(dev, stream);
  colInd_.prefetch_async(dev, stream);
}

template <typename Index> void CSR<Index>::add_next_edge(const EdgeTy<Index> &e) {
  const Index src = e.first;
  const Index dst = e.second;

  SPDLOG_TRACE(logger::console(), "handling edge {}->{}", src, dst);

  maxCol_ = std::max(src, maxCol_);
  maxCol_ = std::max(dst, maxCol_);

  // edge has a new src and should be in a new row
  // even if the edge is filtered out, we need to add empty rows
  while (rowPtr_.size() != size_t(src + 1)) {
    // expecting inputs to be sorted by src, so it should be at least
    // as big as the current largest row we have recored
    assert(src >= rowPtr_.size() && "are edges not ordered by source?");
    SPDLOG_TRACE(logger::console(), "node {} edges start at {}", src, rowPtr_.size());
    rowPtr_.push_back(colInd_.size());
  }

  colInd_.push_back(dst);
}

template <typename Index> void CSR<Index>::finish_edges() {

  // add empty nodes until we reach maxNode
  SPDLOG_TRACE(logger::console(), "adding empty nodes from {} to {}", rowPtr_.size(), maxCol_);
  while (rowPtr_.size() <= size_t(maxCol_) + 1) {
    rowPtr_.push_back(colInd_.size());
  }
}

} // namespace pangolin
