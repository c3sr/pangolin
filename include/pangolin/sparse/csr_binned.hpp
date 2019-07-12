#pragma once

#include <functional>

#include "pangolin/dense/vector.cuh"
#include "pangolin/edge_list.hpp"
#include "pangolin/types.hpp"

#ifdef __CUDACC__
#define PANGOLIN_HOST __host__
#define PANGOLIN_DEVICE __device__
#else
#define PANGOLIN_HOST
#define PANGOLIN_DEVICE
#endif

namespace pangolin {

template <typename NodeIndex, typename EdgeIndex> class CSRBinned;

/*! \brief a read-only view of a CSR suitable for passing to a GPU kernel by
value.

Any modifications to the underlying CSR may invalidate this view.
*/
template <typename NodeIndex, typename EdgeIndex> class CSRBinnedView {
  friend class CSRBinned<NodeIndex, EdgeIndex>;

public:
  typedef EdgeIndex edge_index_type;
  typedef NodeIndex node_index_type;
  typedef EdgeTy<NodeIndex> edge_type;

private:
  uint64_t nnz_;     //!< number of non-zeros
  uint64_t numRows_; //!< length of rowOffset - 1

public:
  const EdgeIndex *rowStart_; //<! offset in col_ where the partition starts
  const EdgeIndex *rowStop_;  //<1 offset in colInd where the partition ends
  const NodeIndex *colInd_;   //!< non-zero column indices

  PANGOLIN_HOST PANGOLIN_DEVICE __forceinline__ uint64_t nnz() const noexcept { return nnz_; }
  PANGOLIN_HOST PANGOLIN_DEVICE __forceinline__ uint64_t num_rows() const noexcept { return numRows_; }
  PANGOLIN_HOST PANGOLIN_DEVICE __forceinline__ uint64_t num_nodes() const noexcept { return numRows_; }
};

/*! \brief


\tparam NodeIndex the integer type needed to address the nodes
\tparam EdgeIndex the integer type needed to address the edges
*/
template <typename NodeIndex, typename EdgeIndex> class CSRBinned {
  static constexpr NodeIndex NUM_PARTS = 8;
  static_assert(NUM_PARTS >= 1, "expect at least one partition");

private:
  NodeIndex partitionSize_;

public:
  typedef EdgeIndex edge_index_type;
  typedef NodeIndex node_index_type;
  typedef EdgeTy<NodeIndex> edge_type;

  std::vector<Vector<EdgeIndex>>
      rowPtrs_; //!< NUM_PARTS+1 offsets in colInd where each partition starts at. Final array is where row ends
  Vector<NodeIndex> colInd_; //!< non-zero column indices

  /*! Empty CSR
   */
  explicit PANGOLIN_HOST CSRBinned(const NodeIndex maxExpectedNode)
      : rowPtrs_(NUM_PARTS + 1 /* last one is end of the rows */) {
    partitionSize_ = (maxExpectedNode + 1 + NUM_PARTS - 1) / NUM_PARTS; // ensure partition size is at least 1
  }

  PANGOLIN_HOST uint64_t nnz() const noexcept {
    return colInd_.size();
  } //!< number of non-zeros                                                //!< number of unique row/col indices

  /*!< number of matrix rows
   */
  PANGOLIN_HOST uint64_t num_rows() const noexcept {
    assert(rowPtrs_.size() && "no rowPtr arrays");
    for (const auto &rowPtr : rowPtrs_) {
      assert(rowPtr.size() == rowPtrs_[0].size() && "not all rowPtrs are the same length");
    }
    return rowPtrs_[0].size();
  }

  PANGOLIN_HOST NodeIndex num_partitions() const noexcept { return NUM_PARTS; }

  /*!
   */
  void add_next_edge(const Edge &edge) {

    const NodeIndex src = edge.first;
    const NodeIndex dst = edge.second;
    SPDLOG_TRACE(logger::console(), "handling edge {}->{}", edge.first, edge.second);

    // edge has a new src and should be in a new row
    while (num_rows() != size_t(src + 1)) {
      // expecting inputs to be sorted by src, so it should be at least
      // as big as the current largest row we have recored
      assert(src >= num_rows() && "are edges not ordered by source?");
      SPDLOG_TRACE(logger::console(), "node {} edges start at {}", edge.first, num_rows());

      // every partition starts at the beginning of the new row
      for (auto &rowPtr : rowPtrs_) {
        rowPtr.push_back(colInd_.size());
      }
    }

    colInd_.push_back(dst);

    // every partition after the one this edge is in starts after this edge
    size_t edgePartIdx = min(NUM_PARTS - 1, dst / partitionSize_); // cap in case the estimated max node is wrong
    SPDLOG_TRACE(logger::console(), "edge {}->{} in partition {}", edge.first, edge.second, edgePartIdx);
    for (size_t incPartIdx = edgePartIdx + 1; incPartIdx < rowPtrs_.size(); ++incPartIdx) {
      auto &rowPtr = rowPtrs_[incPartIdx];
      assert(!rowPtr.empty() && "expecting there to be at least one row");
      (*(rowPtr.end() - 1))++;
    }
  }

  void finish_edges(const NodeIndex maxNode) {
    // add empty nodes until we reach largestNode
    SPDLOG_TRACE(logger::console(), "adding empty nodes through {}", maxNode);
    while (num_rows() <= size_t(maxNode)) {
      for (auto &rowPtr : rowPtrs_) {
        rowPtr.push_back(colInd_.size());
      }
    }
    SPDLOG_TRACE(logger::console(), "num_rows now {}", num_rows());

    for (const auto &rowPtr : rowPtrs_) {
      assert(rowPtr.size() == rowPtrs_[0].size() && "not all rowPtrs are the same length");
    }
  }

  /*! Build a CSR from a range of edges [first, last)

    Only include edges where f is true (default = all edges)
  */
  template <typename EdgeIter>
  static CSRBinned<NodeIndex, EdgeIndex>
  from_edges(EdgeIter begin, EdgeIter end,
             const NodeIndex estMaxNode, //<! estimate of the maximum node that will be seen
             std::function<bool(EdgeTy<NodeIndex>)> f = [](EdgeTy<NodeIndex> e) {
               (void)e;
               return true;
             }) {
    CSRBinned csr(estMaxNode);

    if (begin == end) {
      LOG(warn, "constructing from empty edge sequence");
      return csr;
    }

    NodeIndex largestNode = 0;
    size_t acceptedEdges = 0;

    for (auto ei = begin; ei != end; ++ei) {
      auto edge = *ei;
      const NodeIndex src = edge.first;
      const NodeIndex dst = edge.second;
      if (f(edge)) {
        ++acceptedEdges;
        largestNode = max(largestNode, src);
        largestNode = max(largestNode, dst);

        csr.add_next_edge(edge);
      }
    }
    if (acceptedEdges) {
      csr.finish_edges(largestNode);
    }

    return csr;

#if 0
    CSRBinned csr;
    const NodeIndex partSize = (estMaxNode + 1 + NUM_PARTS - 1) / NUM_PARTS; // ensure partition size is at least 1
    LOG(debug, "partition size is {}", partSize);

    if (begin == end) {
      LOG(warn, "constructing from empty edge sequence");
      return csr;
    }

    // track the largest node seen so far.
    // there may be edges to nodes that have 0 out-degree.
    // if so, at the end, we need to add empty rows up until that node id
    NodeIndex largestNode = 0;
    size_t acceptedEdges = 0;

    for (auto ei = begin; ei != end; ++ei) {
      EdgeTy<NodeIndex> edge = *ei;
      const NodeIndex src = edge.first;
      const NodeIndex dst = edge.second;
      SPDLOG_TRACE(logger::console(), "handling edge {}->{}", edge.first, edge.second);

      if (f(edge)) {
        ++acceptedEdges;
        largestNode = max(largestNode, src);
        largestNode = max(largestNode, dst);

        // edge has a new src and should be in a new row
        while (csr.num_rows() != size_t(src + 1)) {
          // expecting inputs to be sorted by src, so it should be at least
          // as big as the current largest row we have recored
          assert(src >= csr.num_rows() && "are edges not ordered by source?");
          SPDLOG_TRACE(logger::console(), "node {} edges start at {}", edge.first, csr.num_rows());

          // every partition starts at the beginning of the new row
          for (auto &rowPtr : csr.rowPtrs_) {
            rowPtr.push_back(csr.colInd_.size());
          }
        }

        csr.colInd_.push_back(dst);

        // every partition after the one this edge is in starts after this edge
        size_t edgePartIdx = min(NUM_PARTS - 1, dst / partSize); // cap in case the estimated max node is wrong
        SPDLOG_TRACE(logger::console(), "edge {}->{} in partition {}", edge.first, edge.second, edgePartIdx);
        for (size_t incPartIdx = edgePartIdx + 1; incPartIdx < csr.rowPtrs_.size(); ++incPartIdx) {
          auto &rowPtr = csr.rowPtrs_[incPartIdx];
          assert(!rowPtr.empty() && "expecting there to be at least one row");
          (*(rowPtr.end() - 1))++;
        }

      } else {
        continue;
      }
    }

    if (acceptedEdges > 0) {
      // add empty nodes until we reach largestNode
      SPDLOG_TRACE(logger::console(), "adding empty nodes through {}", largestNode);
      while (csr.num_rows() <= size_t(largestNode)) {
        for (auto &rowPtr : csr.rowPtrs_) {
          rowPtr.push_back(csr.colInd_.size());
        }
      }
      SPDLOG_TRACE(logger::console(), "num_rows now {}", csr.num_rows());

      // add the final length of the non-zeros to the offset array
      // csr.rowPtr_.push_back(csr.colInd_.size());
    }

    for (const auto &rowPtr : csr.rowPtrs_) {
      assert(rowPtr.size() == csr.rowPtrs_[0].size() && "not all rowPtrs are the same length");
    }

    return csr;
#endif
  }

  /*! array of offsets in colInd_ where rows start
   */
  const EdgeIndex *row_start() const noexcept { return rowPtrs_[0].data(); }

  /*! array of offsets in colInd_ where rows stop
   */
  const EdgeIndex *row_stop() const noexcept { return rowPtrs_[NUM_PARTS].data(); }

  /*!create a CSRView for this BinnedCSR
   */
  CSRBinnedView<NodeIndex, EdgeIndex> view() const {
    CSRBinnedView<NodeIndex, EdgeIndex> result;
    result.rowStart_ = rowPtrs_[0].data();
    result.rowStop_ = rowPtrs_[NUM_PARTS].data();
    result.colInd_ = colInd_.data();
    result.numRows_ = num_rows();
    result.nnz_ = nnz();
    return result;
  }

  /*! Create a CSRBinnedView for partition part of this BinnerCSR
   */
  CSRBinnedView<NodeIndex, EdgeIndex> view(size_t part) const {
    assert(part < NUM_PARTS && "Requested partition ID larger than NUM_PARTS");
    CSRBinnedView<NodeIndex, EdgeIndex> result;
    result.rowStart_ = rowPtrs_[0].data();
    result.rowStop_ = rowPtrs_[part + 1].data();
    result.colInd_ = colInd_.data();
    result.numRows_ = num_rows();
    result.nnz_ = nnz();
    return result;
  }

  /*! call cudaMemAdvise(..., cudaMemAdviseSetReadMostly, 0) on all data
   */
  PANGOLIN_HOST void read_mostly() {
    for (auto &rowPtr : rowPtrs_) {
      rowPtr.read_mostly();
    }
    colInd_.read_mostly();
  }
  /*! call cudaMemAdvise(..., cudaMemAdviseSetAccessedBy, dev) on all data
   */
  PANGOLIN_HOST void accessed_by(const int dev) {
    for (auto &rowPtr : rowPtrs_) {
      rowPtr.accessed_by(dev);
    }
    colInd_.accessed_by(dev);
  }
  /*! call cudaMemPrefetchAsync(..., dev) on all data
   */
  PANGOLIN_HOST void prefetch_async(const int dev, cudaStream_t stream = 0) {
    for (auto &rowPtr : rowPtrs_) {
      rowPtr.prefetch_async(dev, stream);
    }
    colInd_.prefetch_async(dev, stream);
  }

  /*! Call shrink_to_fit on the underlying containers
   */
  PANGOLIN_HOST void shrink_to_fit() {
    for (auto &rowPtr : rowPtrs_) {
      rowPtr.shrink_to_fit();
    }
    colInd_.shrink_to_fit();
  }

  /*! pre-allocate space for numRows rows and nnz non-zeros
   */
  PANGOLIN_HOST void reserve(size_t numRows, size_t nnz) {
    for (auto &rowPtr : rowPtrs_) {
      rowPtr.reserve(numRows);
    }
    colInd_.reserve(nnz);
  }

  /*! The total capacity of the underlying containers in bytes
   */
  PANGOLIN_HOST PANGOLIN_DEVICE uint64_t capacity_bytes() const noexcept {
    uint64_t cap = 0;
    for (const auto &rowPtr : rowPtrs_) {
      cap += rowPtr.capacity() * sizeof(typename decltype(rowPtr)::value_type);
    }
    cap += colInd_.capacity() * sizeof(typename decltype(colInd_)::value_type);
    return cap;
  }

  /*! The total size of the underlying containers in bytes
   */
  PANGOLIN_HOST PANGOLIN_DEVICE uint64_t size_bytes() const noexcept {
    uint64_t sz = 0;
    for (const auto &rowPtr : rowPtrs_) {
      sz += rowPtr.size() * sizeof(typename decltype(rowPtr)::value_type);
    }
    sz += colInd_.size() * sizeof(typename decltype(colInd_)::value_type);
    return sz;
  }
}; // namespace pangolin

} // namespace pangolin

#undef PANGOLIN_HOST
#undef PANGOLIN_DEVICE