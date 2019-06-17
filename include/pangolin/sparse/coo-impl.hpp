#pragma once

#include <set>

#include "coo.hpp"
#include "pangolin/logger.hpp"

#ifdef __CUDACC__
#define PANGOLIN_HOST __host__
#define DEVICE __device__
#else
#define PANGOLIN_HOST
#define DEVICE
#endif

namespace pangolin {

template <typename Index> COO<Index>::COO() {}

template <typename Index> PANGOLIN_HOST DEVICE uint64_t COO<Index>::num_rows() const {
  if (rowPtr_.size() == 0) {
    return 0;
  } else {
    return rowPtr_.size() - 1;
  }
}

template <typename Index> uint64_t COO<Index>::num_nodes() const { return num_rows(); }

template <typename Index> COO<Index> COO<Index>::from_edgelist(const EdgeList &es, bool (*edgeFilter)(const Edge &)) {
  COO<Index> csr;

  if (es.size() == 0) {
    LOG(warn, "constructing from empty edge list");
    return csr;
  }

  for (const auto &edge : es) {

    const Index src = static_cast<Index>(edge.first);
    const Index dst = static_cast<Index>(edge.second);

    // edge has a new src and should be in a new row
    // even if the edge is filtered out, we need to add empty rows
    while (csr.rowPtr_.size() != size_t(src + 1)) {
      // expecting inputs to be sorted by src, so it should be at least
      // as big as the current largest row we have recored
      assert(src >= csr.rowPtr_.size());
      // SPDLOG_TRACE(logger::console(), "node {} edges start at {}", edge.src_,
      // csr.edgeSrc_.size());
      csr.rowPtr_.push_back(csr.colInd_.size());
    }

    // filter or add the edge
    if (nullptr != edgeFilter && edgeFilter(edge)) {
      continue;
    } else {
      csr.rowInd_.push_back(src);
      csr.colInd_.push_back(dst);
    }
  }

  // add the final length of the non-zeros to the offset array
  csr.rowPtr_.push_back(csr.colInd_.size());

  assert(csr.rowInd_.size() == csr.colInd_.size());
  return csr;
}

/*! Build a COO from a sequence of edges

*/
template <typename Index>
template <typename EdgeIter>
COO<Index> COO<Index>::from_edges(EdgeIter begin, EdgeIter end, std::function<bool(EdgeTy<Index>)> f) {
  COO<Index> coo;

  if (begin == end) {
    LOG(warn, "constructing from empty edge sequence");
    return coo;
  }

  // track the largest node seen so far.
  // there may be edges to nodes that have 0 out-degree.
  // if so, at the end, we need to add empty rows up until that node id
  Index largestNode = 0;
  size_t acceptedEdges = 0;

  for (auto ei = begin; ei != end; ++ei) {
    EdgeTy<Index> edge = *ei;
    const Index src = edge.first;
    const Index dst = edge.second;

    SPDLOG_TRACE(logger::console(), "handling edge {}->{}", edge.first, edge.second);

    if (f(edge)) {
      ++acceptedEdges;
      largestNode = max(largestNode, src);
      largestNode = max(largestNode, dst);
      while (coo.rowPtr_.size() != size_t(src + 1)) {
        // expecting inputs to be sorted by src, so it should be at least
        // as big as the current largest row we have recored
        assert(src >= coo.rowPtr_.size() && "are edges not sorted by source?");
        SPDLOG_TRACE(logger::console(), "node {} edges start at {}", edge.first, coo.rowPtr_.size());
        coo.rowPtr_.push_back(coo.colInd_.size());
      }

      coo.rowInd_.push_back(src);
      coo.colInd_.push_back(dst);
    } else {
      continue;
    }
  }

  if (acceptedEdges > 0) {
    // add empty nodes until we reach largestNode
    SPDLOG_TRACE(logger::console(), "adding empty nodes from {} to {}", coo.rowPtr_.size(), largestNode);
    while (coo.rowPtr_.size() <= size_t(largestNode)) {
      coo.rowPtr_.push_back(coo.colInd_.size());
    }
    // the nbr list starts at coo.rowPtr_[largestNode]
    if (coo.rowPtr_.size() != size_t(largestNode) + 1) {
      LOG(error, "the largest observed node {} does not have a rowPtr", largestNode);
      exit(1);
    }

    // add the final length of the non-zeros to the offset array
    coo.rowPtr_.push_back(coo.colInd_.size());
  }

  if (coo.rowInd_.size() != coo.colInd_.size()) {
    LOG(error, "rowInd and colInd sizes do not match");
    exit(1);
  }
  return coo;
}

template <typename Index> void COO<Index>::add_next_edge(const EdgeTy<Index> &e) {
  const Index src = e.first;
  const Index dst = e.second;
  SPDLOG_TRACE(logger::console(), "handling edge {}->{}", src, dst);

  // edge has a new src and should be in a new row
  // even if the edge is filtered out, we need to add empty rows
  while (rowPtr_.size() != size_t(src + 1)) {
    // expecting inputs to be sorted by src, so it should be at least
    // as big as the current largest row we have recored
    assert(src >= rowPtr_.size() && "are edges not ordered by source?");
    SPDLOG_TRACE(logger::console(), "node {} edges start at {}", src, rowPtr_.size());
    rowPtr_.push_back(colInd_.size());
  }

  rowInd_.push_back(src);
  colInd_.push_back(dst);
}

template <typename Index> void COO<Index>::finish_edges(const Index &maxNode) {

  // add empty nodes until we reach maxNode
  SPDLOG_TRACE(logger::console(), "adding empty nodes from {} to {}", rowPtr_.size(), maxNode);
  while (rowPtr_.size() <= size_t(maxNode)) {
    rowPtr_.push_back(colInd_.size());
  }

  // add the final length of the non-zeros to the offset array
  rowPtr_.push_back(colInd_.size());
}

template <typename Index> COOView<Index> COO<Index>::view() const {
  COOView<Index> view;
  view.nnz_ = nnz();
  view.num_rows_ = num_rows();
  view.rowPtr_ = rowPtr_.data();
  view.colInd_ = colInd_.data();
  view.rowInd_ = rowInd_.data();
  return view;
}

template <typename Index> void COO<Index>::read_mostly() {
  rowPtr_.read_mostly();
  rowInd_.read_mostly();
  colInd_.read_mostly();
}

template <typename Index> void COO<Index>::accessed_by(const int dev) {
  rowPtr_.accessed_by(dev);
  rowInd_.accessed_by(dev);
  colInd_.accessed_by(dev);
}

template <typename Index> void COO<Index>::prefetch_async(const int dev, cudaStream_t stream) {
  rowPtr_.prefetch_async(dev, stream);
  rowInd_.prefetch_async(dev, stream);
  colInd_.prefetch_async(dev, stream);
}

} // namespace pangolin

#undef PANGOLIN_HOST
#undef DEVICE
