#pragma once

#include "pangolin/dense/vector.hu"
#include "pangolin/edge_list.hpp"
#include "pangolin/types.hpp"

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

namespace pangolin {

template <typename Index> class COO;

/*! \brief a read-only view of a COO suitable for passing to a GPU kernel by
value.

Any modifications to the underlying COO may invalidate this view.
*/
template <typename Index> class COOView {
  friend class COO<Index>;

private:
  uint64_t nnz_;      //!< number of non-zeros
  uint64_t num_rows_; //!< length of rowOffset - 1

public:
  const Index *rowPtr_; //!< offset in col_ that each row starts at
  const Index *rowInd_; //!< non-zero row indices
  const Index *colInd_; //!< non-zero column indices

  HOST DEVICE uint64_t nnz() const { return nnz_; }
  HOST DEVICE uint64_t num_rows() const { return num_rows_; }

  const Index *row_ptr() const { return rowPtr_; } //!< row offset array
  const Index *col_ind() const { return colInd_; } //!< column index array
  const Index *row_ind() const { return rowInd_; } //<! row index array
};

/*! \brief A COO matrix backed by CUDA Unified Memory, with a CSR rowPtr

Copying to a GPU kernel by value will cause the underling memory to be copied as
well. For read-only GPU access, use the view() method to get a lightweight
reference to the data.
*/
template <typename Index> class COO {
private:
  Index maxCol_;

public:
  COO();                 //!< empty CSR
  Vector<Index> rowPtr_; //!< offset in col_ that each row starts at
  Vector<Index> colInd_; //!< non-zero column indices
  Vector<Index> rowInd_; //!< non-zero row indices
  HOST DEVICE uint64_t nnz() const {
    assert(colInd_.size() == rowInd_.size());
    return colInd_.size();
  }                                      //!< number of non-zeros
  uint64_t num_nodes() const;            //!< number of unique row/col indices
  HOST DEVICE uint64_t num_rows() const; //!< number of matrix rows

  /*! Build a COO from an EdgeList

  Do not include edges where edgeFilter(edge) returns true
  */
  static COO<Index> from_edgelist(const EdgeList &es,
                                  bool (*edgeFilter)(const Edge &) = nullptr);
  COOView<Index> view() const; //!< create a COOView for this COO

  const Index *row_ptr() { return rowPtr_.data(); } //!< row offset array
  const Index *col_ind() { return colInd_.data(); } //!< column index array
  const Index *row_ind() { return rowInd_.data(); } //<! row index array
};

} // namespace pangolin

#undef HOST
#undef DEVICE

#include "coo-impl.hpp"