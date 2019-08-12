#pragma once

#include <functional>

#include "pangolin/dense/array_view.hpp"
#include "pangolin/dense/vector.cuh"
#include "pangolin/edge_list.hpp"
#include "pangolin/logger.hpp"
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
  const EdgeIndex *rowStart_;       //!< offset in colInd_ where the row starts
  const EdgeIndex *rowStop_;        //!< offset in colInd_ where the row ends
  const EdgeIndex *partitionStart_; //!< offset in colInd_ where the partition starts
  const EdgeIndex *partitionStop_;  //!< offset in colInd_ where the partition ends
  const NodeIndex *colInd_;         //!< non-zero column indices

  PANGOLIN_HOST PANGOLIN_DEVICE __forceinline__ uint64_t nnz() const noexcept { return nnz_; }
  PANGOLIN_HOST PANGOLIN_DEVICE __forceinline__ uint64_t num_rows() const noexcept { return numRows_; }
  PANGOLIN_HOST PANGOLIN_DEVICE __forceinline__ uint64_t num_nodes() const noexcept { return numRows_; }

  PANGOLIN_HOST EdgeIndex part_nnz() const noexcept {
    EdgeIndex nnz = 0;
    for (NodeIndex row = 0; row < numRows_; ++row) {
      nnz += partitionStop_[row] - partitionStart_[row];
    }
    return nnz;
  }

  /*! Return an ArrayView of row i of the CSR
   */
  PANGOLIN_HOST PANGOLIN_DEVICE __forceinline__ const ArrayView<NodeIndex> row(NodeIndex i) const noexcept {
    const NodeIndex rowStart = rowStart_[i];
    const NodeIndex rowStop = rowStop_[i];
    return ArrayView<NodeIndex>(&colInd_[rowStart], size_t(rowStop - rowStart));
  }

  /*! Return an ArrayView of partition p of row i of the CSR
   */
  PANGOLIN_HOST PANGOLIN_DEVICE __forceinline__ const ArrayView<NodeIndex> row_part(NodeIndex i) const noexcept {
    const NodeIndex rowStart = partitionStart_[i];
    const NodeIndex rowStop = partitionStop_[i];
    return ArrayView<NodeIndex>(&colInd_[rowStart], size_t(rowStop - rowStart));
  }
};

/*! \brief a read-only view of a CSR suitable for passing to a GPU kernel by
value.

Any modifications to the underlying CSR may invalidate this view.
*/
template <typename NodeIndex, typename EdgeIndex> class TwoColView {
  friend class CSRBinned<NodeIndex, EdgeIndex>;

public:
  typedef EdgeIndex edge_index_type;
  typedef NodeIndex node_index_type;
  typedef EdgeTy<NodeIndex> edge_type;

private:
  uint64_t nnz_;
  uint64_t numRows_;       //!< length of rowOffset - 1
  uint64_t partitionSize_; //!< the number of rows/cols in a partition

public:
  const EdgeIndex *jStartPtrs_; //!< offsets in colInd_ where the partition starts
  const EdgeIndex *jStopPtrs_;  //!< offsets in colInd_ where the partition ends
  const EdgeIndex *kStartPtrs_; //!< offsets in colInd_ where the partition starts
  const EdgeIndex *kStopPtrs_;  //!< offsets in colInd_ where the partition ends
  NodeIndex *colInd_;           //!< non-zero column indices

  PANGOLIN_HOST PANGOLIN_DEVICE __forceinline__ uint64_t partition_size() const noexcept { return partitionSize_; }
  PANGOLIN_HOST PANGOLIN_DEVICE __forceinline__ uint64_t nnz() const noexcept { return nnz_; }
  PANGOLIN_HOST PANGOLIN_DEVICE __forceinline__ uint64_t num_rows() const noexcept { return numRows_; }

  /*! Return an ArrayView of partition p of row i of the CSR
   */
  PANGOLIN_HOST PANGOLIN_DEVICE __forceinline__ ArrayView<NodeIndex> row_j(NodeIndex i) const noexcept {
    assert(i < num_rows());
    const NodeIndex rowStart = jStartPtrs_[i];
    const NodeIndex rowStop = jStopPtrs_[i];
    assert(rowStop >= rowStart);
    return ArrayView<NodeIndex>(&colInd_[rowStart], size_t(rowStop - rowStart));
  }

  PANGOLIN_HOST PANGOLIN_DEVICE __forceinline__ ArrayView<NodeIndex> row_k(NodeIndex i) const noexcept {
    assert(i < num_rows());
    const NodeIndex rowStart = kStartPtrs_[i];
    const NodeIndex rowStop = kStopPtrs_[i];
    // if (!(rowStop >= rowStart)) {
    //   LOG(critical, "i={}: partition ends before it begins {} !>= {}", i, rowStop, rowStart);
    //   exit(1);
    // }
    assert(rowStop >= rowStart);
    return ArrayView<NodeIndex>(&colInd_[rowStart], size_t(rowStop - rowStart));
  }
};

/*! \brief


\tparam NodeIndex the integer type needed to address the nodes
\tparam EdgeIndex the integer type needed to address the edges
*/
template <typename NodeIndex, typename EdgeIndex> class CSRBinned {

private:
  NodeIndex numParts_;
  NodeIndex partitionSize_;
  NodeIndex maxNode_;

public:
  typedef EdgeIndex edge_index_type;
  typedef NodeIndex node_index_type;
  typedef EdgeTy<NodeIndex> edge_type;

  std::vector<Vector<EdgeIndex>>
      rowPtrs_; //!< NUM_PARTS+1 offsets in colInd where each partition starts at. Final array is where row ends
  Vector<NodeIndex> colInd_; //!< non-zero column indices

  /*! Empty CSR
   */
  explicit PANGOLIN_HOST CSRBinned(const NodeIndex numRows, const EdgeIndex nnz)
      : numParts_(8), rowPtrs_(numParts_ + 1), maxNode_(0) {
    partitionSize_ = (numRows + numParts_ - 1) / numParts_;
    for (size_t i = 0; i < numParts_; ++i) {
      LOG(debug, "CSRBinned parition {}: cols {}-{}", i, i * partitionSize_, (i + 1) * partitionSize_);
    }
    colInd_.reserve(nnz);
  }

  PANGOLIN_HOST
  CSRBinned(CSRBinned &&rhs) : partitionSize_(rhs.partitionSize_) {
    rowPtrs_ = std::move(rhs.rowPtrs_);
    colInd_ = std::move(colInd_);
  }

  /*! number of non-zeros in the whole csr
   */
  PANGOLIN_HOST uint64_t nnz() const noexcept { return colInd_.size(); }

  /*!< number of matrix rows
   */
  PANGOLIN_HOST uint64_t num_rows() const noexcept {
    assert(rowPtrs_.size() && "no rowPtr arrays");
    for (const auto &rowPtr : rowPtrs_) {
      assert(rowPtr.size() == rowPtrs_[0].size() && "not all rowPtrs are the same length");
    }
    return rowPtrs_[0].size();
  }

  PANGOLIN_HOST uint64_t num_partitions() const noexcept { return numParts_; }

  /*!
   */
  void add_next_edge(const Edge &edge) {

    const NodeIndex src = edge.first;
    const NodeIndex dst = edge.second;
    SPDLOG_TRACE(logger::console(), "handling edge {}->{}", edge.first, edge.second);

    // for an edge with src 0, we should have no more than 1 row
    assert(src + 1 >= num_rows() && "edges must be sorted by src");

    // edge has a new src and should be in a new row
    while (num_rows() <= src) {
      // expecting inputs to be sorted by src, so it should be at least
      // as big as the current largest row we have recored
      SPDLOG_TRACE(logger::console(), "node {} edges start at {}", edge.first, num_rows());

      // every partition starts at the beginning of the new row
      for (auto &rowPtr : rowPtrs_) {
        rowPtr.push_back(colInd_.size());
      }
    }

    colInd_.push_back(dst);

    // every partition after the one this edge is in starts after this edge
    size_t edgePartIdx = min(dst / partitionSize_, numParts_ - 1); // cap in case the estimated max node is wrong
    SPDLOG_TRACE(logger::console(), "edge {}->{} in partition {}", edge.first, edge.second, edgePartIdx);
    // LOG(debug, "edge {}->{} in partition {}", edge.first, edge.second, edgePartIdx);
    for (size_t incPartIdx = edgePartIdx + 1; incPartIdx < rowPtrs_.size(); ++incPartIdx) {
      auto &rowPtr = rowPtrs_[incPartIdx];
      assert(!rowPtr.empty() && "expecting there to be at least one row");
      (*(rowPtr.end() - 1))++;
    }

    maxNode_ = max(maxNode_, src);
    maxNode_ = max(maxNode_, dst);
  }

  void finish_edges() {
    if (nnz() > 0) {
      // add empty nodes until we reach largestNode
      SPDLOG_TRACE(logger::console(), "adding empty nodes through {}", maxNode_);
      while (num_rows() <= maxNode_) {
        for (auto &rowPtr : rowPtrs_) {
          rowPtr.push_back(colInd_.size());
        }
      }
      SPDLOG_TRACE(logger::console(), "num_rows now {}", num_rows());
    }

    // for (const auto &rowPtr : rowPtrs_) {
    //   assert(rowPtr.size() == rowPtrs_[0].size() && "not all rowPtrs are the same length");
    // }

    // for (int64_t i = 0; i < int64_t(rowPtrs_.size()) - 1; ++i) {
    //   assert(rowPtrs_[i + 1][0] >= rowPtrs_[i][0]);
    // }

    // for (int64_t i = 0; i < int64_t(rowPtrs_.size()) - 1; ++i) {
    //   for (size_t j = 0; j < rowPtrs_[0].size(); ++j) {
    //     assert(rowPtrs_[i + 1][j] >= rowPtrs_[i][j]);
    //   }
    // }
#if 0
    // check that all rows are contiguous
    LOG(debug, "checking csr for consistency...");
    for (size_t i = 0; i < rowPtrs_.size() - 1; ++i) {
      LOG(debug, "comparing rowPtrs_ {} and {}", i, i + 1);
      auto &p1 = rowPtrs_[i];
      auto &p2 = rowPtrs_[i + 1];
      // all rowPtrs should be the same size
      if (p1.size() != p2.size()) {
        LOG(critical, "row pointer size mismatch");
        exit(1);
      }
      if (p1.size() != num_rows()) {
        LOG(critical, "num_rows is wrong");
        exit(1);
      }
      for (NodeIndex row = 0; row < num_rows(); ++row) {
        // partition 2 should start after partition 1
        if (p1[row] > p2[row]) {
          LOG(critical, "partitions are not non-decreasing");
          exit(1);
        }
      }

      // the beginning of the first partition of a row is the end of the last partition of the preceeding row
      if (rowPtrs_[0][i + 1] != rowPtrs_[NUM_PARTS][i]) {
        LOG(critical, "partitions in different rows are not adjacent");
        exit(1);
      }
    }
    for (size_t i = 0; i < rowPtrs_.size(); ++i) {
      LOG(debug, "checking partition {}", i);
      auto &p1 = rowPtrs_[i];
      for (NodeIndex row = 0; row < num_rows() - 1; ++row) {
        // all later row pointer entries should point past earlier row pointer entries
        if (p1[row + 1] < p1[row]) {
          LOG(critical, "row pointer array is decreasing");
        }
      }
    }
    // the total across all parts should be end - beginning

    for (NodeIndex row = 0; row < num_rows(); ++row) {
      uint64_t total = 0;
      for (size_t i = 0; i < rowPtrs_.size() - 1; ++i) {
        auto &p1 = rowPtrs_[i];
        auto &p2 = rowPtrs_[i + 1];
        total += p2[row] - p1[row];
      }

      if (total != rowPtrs_[num_partitions()][row] - rowPtrs_[0][row]) {
        LOG(critical, "partitions in row {} doesnt have the right nnz", row);
      }
    }
#endif
  }

  /*! Build a CSR from a range of edges [first, last)

    Only include edges where f is true (default = all edges)
  */
  template <typename EdgeIter>
  static CSRBinned<NodeIndex, EdgeIndex>
  from_edges(EdgeIter begin, EdgeIter end,
             const NodeIndex numNodes,   //!< estimate of the maximum node that will be seen
             const EdgeIndex numEntries, //!< estimate of the number of edges
             std::function<bool(EdgeTy<NodeIndex>)> f = [](EdgeTy<NodeIndex> e) {
               (void)e;
               return true;
             }) {
    CSRBinned csr(numNodes, numEntries);

    if (begin == end) {
      LOG(warn, "constructing from empty edge sequence");
      return csr;
    }

    for (auto ei = begin; ei != end; ++ei) {
      auto edge = *ei;
      const NodeIndex src = edge.first;
      const NodeIndex dst = edge.second;
      if (f(edge)) {
        csr.add_next_edge(edge);
      }
    }
    csr.finish_edges();

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

  /*! Return an ArrayView of row i of the CSR
   */
  ArrayView<NodeIndex> row(NodeIndex i) const noexcept {
    const EdgeIndex rowStart = rowPtrs_[0][i];
    const EdgeIndex rowStop = rowPtrs_.back()[i];
    return ArrayView<NodeIndex>(&colInd_[rowStart], rowStop - rowStart);
  }

  /*! Return an ArrayView of partition p of row i of the CSR
   */
  ArrayView<NodeIndex> row_part(NodeIndex i, size_t p) const noexcept {
    const EdgeIndex rowStart = rowPtrs_[p][i];
    const EdgeIndex rowStop = rowPtrs_[p + 1][i];
    return ArrayView<NodeIndex>(&colInd_[rowStart], rowStop - rowStart);
  }

  /*! array of offsets in colInd_ where rows start
   */
  const EdgeIndex *row_start() const noexcept { return rowPtrs_[0].data(); }

  /*! array of offsets in colInd_ where rows stop
   */
  const EdgeIndex *row_stop() const noexcept { return rowPtrs_[numParts_].data(); }

  /*!create a CSRView for this BinnedCSR
   */
  CSRBinnedView<NodeIndex, EdgeIndex> view() const {
    CSRBinnedView<NodeIndex, EdgeIndex> result;
    result.rowStart_ = rowPtrs_[0].data();
    result.rowStop_ = rowPtrs_[numParts_].data();
    result.partitionStart_ = rowPtrs_[0].data();
    result.partitionStop_ = rowPtrs_[numParts_].data();
    result.colInd_ = colInd_.data();
    result.numRows_ = num_rows();
    result.nnz_ = nnz();
    return result;
  }

  /*! Create a CSRBinnedView for partition part of this BinnedCSR
   */
  CSRBinnedView<NodeIndex, EdgeIndex> view(size_t part) const { return view(part, part + 1); }

  /*! Create a CSRBinnedView for a span of partitions of this BinnedCSR
   */
  CSRBinnedView<NodeIndex, EdgeIndex> view(size_t partStart, size_t partStop) const {
    assert(partStart < numParts_ && "Requested partition Start >= numParts_");
    assert(partStop <= numParts_ && "Requested partition stop > numParts_");
    CSRBinnedView<NodeIndex, EdgeIndex> result;
    result.rowStart_ = rowPtrs_[0].data();
    result.rowStop_ = rowPtrs_[numParts_].data();
    result.partitionStart_ = rowPtrs_[partStart].data();
    result.partitionStop_ = rowPtrs_[partStop].data();
    result.colInd_ = colInd_.data();
    result.numRows_ = num_rows();
    result.nnz_ = nnz();
    return result;
  }

  TwoColView<NodeIndex, EdgeIndex> two_col_view(size_t j, size_t k) {
    assert(j < numParts_ && "Requested j >= numParts_");
    // assert(j <= numParts_ && "Requested j > numParts_");
    assert(k < numParts_ && "Requested k >= numParts_");
    // assert(k <= numParts_ && "Requested k > numParts_");
    TwoColView<NodeIndex, EdgeIndex> result;
    result.nnz_ = nnz();
    result.numRows_ = num_rows();
    result.colInd_ = colInd_.data();
    result.partitionSize_ = partitionSize_;

    result.jStartPtrs_ = rowPtrs_[j].data();
    result.jStopPtrs_ = rowPtrs_[j + 1].data();
    result.kStartPtrs_ = rowPtrs_[k].data();
    result.kStopPtrs_ = rowPtrs_[k + 1].data();

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