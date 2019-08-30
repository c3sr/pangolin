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

/*!
  \tparam NodeI Index type for nodes
  \tparam EdgeI Index type for edges
  \tparam ValT type of Values
  \tparam EdgeA type of allocator for rowPtr_
  \tparam NodeA type of allocator for colInd_
  \tparam ValA type of allocator for vals_
*/
template <typename NodeI, typename EdgeI, typename ValT, typename NodeA = allocator::CUDAManaged<NodeI>,
          typename EdgeA = allocator::CUDAManaged<EdgeI>, typename ValA = allocator::CUDAManaged<ValT>>
class CSR;

/*! \brief
 */
template <typename NodeI, typename EdgeI, typename ValT, typename NodeA, typename EdgeA, typename ValA> class CSR {
public:
  typedef NodeI node_index_type;
  typedef EdgeI edge_index_type;
  typedef ValT value_type;

private:
  node_index_type maxNode_;

public:
  Vector<EdgeI, EdgeA> rowPtr_; //!< offset in col_ that each row starts at
  Vector<NodeI, NodeA> colInd_; //!< non-zero column indices
  Vector<ValT, ValA> vals_;     //<! CSR values

  /*! default constructor
   */
  CSR() : maxNode_(0) {}

  void add_next_edge(const WeightedDiEdge<NodeI, ValT> &e) {
    SPDLOG_TRACE(logger::console(), "handling edge {}->{}", e.src, e.dst);

    maxNode_ = std::max(e.src, maxNode_);
    maxNode_ = std::max(e.dst, maxNode_);

    // edge has a new src and should be in a new row
    // even if the edge is filtered out, we need to add empty rows
    while (rowPtr_.size() != size_t(e.src + 1)) {
      // expecting inputs to be sorted by src, so it should be at least
      // as big as the current largest row we have recored
      assert(e.src >= rowPtr_.size() && "are edges not ordered by source?");
      SPDLOG_TRACE(logger::console(), "node {} edges start at {}", e.src, rowPtr_.size());
      rowPtr_.push_back(colInd_.size());
    }

    colInd_.push_back(e.dst);
    vals_.push_back(e.val);
  }

  void finish_edges() {

    if (nnz() > 0) {
      // add empty nodes until we reach maxNode
      SPDLOG_TRACE(logger::console(), "adding empty nodes from {} to {}", rowPtr_.size(), maxNode_);
      while (rowPtr_.size() <= size_t(maxNode_)) {
        rowPtr_.push_back(colInd_.size());
      }

      // add the final length of the non-zeros to the offset array
      rowPtr_.push_back(colInd_.size());
    }
  }

  /*! Build a CSR from a range of edges [first, last)

    Only include edges where f is true (default = all edges)
  */
  template <typename EdgeIter>
  static CSR from_edges(EdgeIter begin, EdgeIter end,
                        std::function<bool(WeightedDiEdge<NodeI, ValT>)> f = [](WeightedDiEdge<NodeI, ValT> e) {
                          (void)e;
                          return true;
                        }) {
    CSR m;
    for (auto i = begin; i != end; ++i) {
      auto e = *i;
      if (f(e)) {
        m.add_next_edge(*i);
      }
    }
    m.finish_edges();
    return m;
  }

  /*! number of non-zeros
   */
  PANGOLIN_HOST PANGOLIN_DEVICE uint64_t nnz() const {
    assert(colInd_.size() == vals_.size());
    return colInd_.size();
  }

  PANGOLIN_HOST PANGOLIN_DEVICE uint64_t num_rows() const {
    assert(colInd_.size() == vals_.size());
    if (rowPtr_.empty()) {
      return 0;
    } else {
      return rowPtr_.size() - 1;
    }
  }

  /*! call cudaMemAdvise(..., cudaMemAdviseSetReadMostly, 0) on all data
   */
  PANGOLIN_HOST void read_mostly() {
    rowPtr_.read_mostly();
    colInd_.read_mostly();
    vals_.read_mostly();
  }
  /*! call cudaMemAdvise(..., cudaMemAdviseSetAccessedBy, dev) on all data
   */
  PANGOLIN_HOST void accessed_by(const int dev) {
    rowPtr_.accessed_by(dev);
    colInd_.accessed_by(dev);
    vals_.accessed_by(dev);
  }
  /*! call cudaMemPrefetchAsync(..., dev) on all data
   */
  PANGOLIN_HOST void prefetch_async(const int dev, cudaStream_t stream = 0) {
    rowPtr_.prefetch_async(dev, stream);
    colInd_.prefetch_async(dev, stream);
    vals_.prefetch_async(dev, stream);
  }

  /*! Call shrink_to_fit on the underlying containers
   */
  PANGOLIN_HOST void shrink_to_fit() {
    rowPtr_.shrink_to_fit();
    colInd_.shrink_to_fit();
    vals_.shrink_to_fit();
  }

  /*! pre-allocate space for numRows rows and nnz non-zeros
   */
  PANGOLIN_HOST void reserve(size_t numRows, size_t numNonZeros) {
    rowPtr_.reserve(numRows + 1);
    colInd_.reserve(numNonZeros);
    vals_.reserve(numNonZeros);
  }

  /*! The total capacity of the underlying containers in bytes
   */
  PANGOLIN_HOST PANGOLIN_DEVICE uint64_t capacity_bytes() const noexcept {
    return rowPtr_.capacity() * sizeof(typename decltype(rowPtr_)::value_type) +
           colInd_.capacity() * sizeof(typename decltype(colInd_)::value_type) +
           vals_.capacity() * sizeof(typename decltype(vals_)::value_type);
  }
  /*! The total size of the underlying containers in bytes
   */
  PANGOLIN_HOST PANGOLIN_DEVICE uint64_t size_bytes() const noexcept {
    return rowPtr_.size() * sizeof(typename decltype(rowPtr_)::value_type) +
           colInd_.size() * sizeof(typename decltype(colInd_)::value_type) +
           vals_.size() * sizeof(typename decltype(vals_)::value_type);
  }

  /*! raw pointer to row pointer array
   */
  const edge_index_type *row_ptr() const noexcept { return rowPtr_.data(); }
  const node_index_type *col_ind() const noexcept { return colInd_.data(); }
  const value_type *data() const noexcept { return vals_.data(); }
};

} // namespace pangolin

#undef PANGOLIN_HOST
#undef PANGOLIN_DEVICE