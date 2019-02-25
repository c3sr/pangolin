#pragma once

#include <set>

#include "gpu_csr.hpp"
#include "pangolin/logger.hpp"

#ifdef __CUDACC__
#define PANGOLIN_CUDA_MEMBER __host__ __device__
#else
#define PANGOLIN_CUDA_MEMBER
#endif

namespace pangolin {

template <typename Index> GPUCSR<Index>::GPUCSR() : maxCol_(0) {}

template <typename Index>
PANGOLIN_CUDA_MEMBER uint64_t GPUCSR<Index>::num_rows() const {
  if (rowOffset_.size() == 0) {
    return 0;
  } else {
    return rowOffset_.size() - 1;
  }
}

template <typename Index> uint64_t GPUCSR<Index>::num_nodes() const {
  std::set<Index> nodes;
  // add all dsts
  for (Index ci = 0; ci < col_.size(); ++ci) {
    nodes.insert(col_[ci]);
  }
  // add non-zero sources
  for (Index i = 0; i < rowOffset_.size() - 1; ++i) {
    Index row_start = rowOffset_[i];
    Index row_end = rowOffset_[i + 1];
    if (row_start != row_end) {
      nodes.insert(i);
    }
  }
  return nodes.size();
}

template <typename Index>
GPUCSR<Index> GPUCSR<Index>::from_edgelist(const EdgeList &es,
                                           bool (*edgeFilter)(const Edge &)) {
  GPUCSR<Index> csr;

  if (es.size() == 0) {
    LOG(warn, "constructing from empty edge list");
    return csr;
  }

  for (const auto &edge : es) {

    const Index src = static_cast<Index>(edge.first);
    const Index dst = static_cast<Index>(edge.second);

    // edge has a new src and should be in a new row
    // even if the edge is filtered out, we need to add empty rows
    while (csr.rowOffset_.size() != size_t(src + 1)) {
      // expecting inputs to be sorted by src, so it should be at least
      // as big as the current largest row we have recored
      assert(src >= csr.rowOffset_.size());
      // SPDLOG_TRACE(logger::console, "node {} edges start at {}", edge.src_,
      // csr.edgeSrc_.size());
      csr.rowOffset_.push_back(csr.col_.size());
    }

    // filter or add the edge
    if (nullptr != edgeFilter && edgeFilter(edge)) {
      continue;
    } else {
      csr.col_.push_back(dst);
      csr.maxCol_ = std::max(dst, csr.maxCol_);
    }
  }

  // add the final length of the non-zeros to the offset array
  csr.rowOffset_.push_back(csr.col_.size());

  return csr;
}

template <typename Index> GPUCSRView<Index> GPUCSR<Index>::view() const {
  GPUCSRView<Index> view;
  view.nnz_ = nnz();
  view.num_rows_ = num_rows();
  view.rowOffset_ = rowOffset_.data();
  view.col_ = col_.data();
  return view;
}

} // namespace pangolin

#undef PANGOLIN_CUDA_MEMBER
