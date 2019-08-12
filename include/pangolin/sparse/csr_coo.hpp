#pragma once

#include <functional>

#include "pangolin/dense/vector.cuh"
#include "pangolin/edge_list.hpp"
#include "pangolin/types.hpp"

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#define INLINE __forceinline__
#else
#define HOST
#define DEVICE
#define INLINE
#endif

namespace pangolin {

template <typename Index, typename Vector> class CSRCOO;

/*! \brief a read-only view of a CSRCOO suitable for passing to a GPU kernel by
value.

Any modifications to the underlying CSRCOO may invalidate this view.
*/
template <typename Index, typename Vector> class CSRCOOView {
  friend class CSRCOO<Index, Vector>;

private:
  uint64_t nnz_;      //!< number of non-zeros
  uint64_t num_rows_; //!< length of rowOffset - 1

public:
  typedef Index index_type;
  typedef EdgeTy<Index> edge_type;
  const Index *rowPtr_; //!< offset in col_ that each row starts at
  const Index *rowInd_; //!< non-zero row indices
  const Index *colInd_; //!< non-zero column indices

  HOST DEVICE INLINE uint64_t nnz() const { return nnz_; }
  HOST DEVICE INLINE uint64_t num_rows() const { return num_rows_; }

  INLINE const Index *row_ptr() const { return rowPtr_; }                    //!< row offset array
  INLINE const Index *col_ind() const { return colInd_; }                    //!< column index array
  INLINE const Index *row_ind() const { return rowInd_; }                    //!< row index array
  HOST DEVICE INLINE const Index *device_row_ptr() const { return rowPtr_; } //!< row offset array
  HOST DEVICE INLINE const Index *device_col_ind() const { return colInd_; } //!< column index array
  HOST DEVICE INLINE const Index *device_row_ind() const { return rowInd_; } //!< row index array
};

/*! \brief A hybrid CSRCOO matrix

Copying to a GPU kernel by value will cause the underling memory to be copied as
well. For read-only GPU access, use the view() method to get a lightweight
reference to the data.

\tparam Vector the vector type used for rowPtr, rowInd, and colInd
*/
template <typename Index, typename Vector = Vector<Index>> class CSRCOO {

public:
  typedef Index index_type;
  typedef EdgeTy<Index> edge_type;
  CSRCOO() {}     //!< empty matrix
  Vector rowPtr_; //!< offset in colInd_/rowInd_ that each row starts at
  Vector colInd_; //!< non-zero column indices
  Vector rowInd_; //!< non-zero row indices
  HOST DEVICE uint64_t nnz() const {
    assert(colInd_.size() == rowInd_.size());
    return colInd_.size();
  }                           //!< number of non-zeros
  uint64_t num_nodes() const; //!< number of unique row/col indices

  /*! number of matrix rows

  0 if rowPtr_.size() is 0, otherwise rowPtr_.size() - 1
   */
  HOST DEVICE uint64_t num_rows() const {
    if (rowPtr_.size() == 0) {
      return 0;
    } else {
      return rowPtr_.size() - 1;
    }
  }

  /*! Build a CSRCOO from an EdgeList

  Do not include edges where edgeFilter(edge) returns true
  */
  static CSRCOO<Index, Vector> from_edgelist(const EdgeList &es, bool (*edgeFilter)(const Edge &) = nullptr);

  /*! Build a CSRCOO from a sequence of edges

    Only include edges where f is true (default = all edges)

  */
  template <typename EdgeIter>
  static CSRCOO<Index, Vector> from_edges(EdgeIter begin, EdgeIter end,
                                          std::function<bool(EdgeTy<Index>)> f = [](EdgeTy<Index> e) { return true; });

  /*! Add a single edge to the CSRCOO.

  The edge should either
    start a new row
    be in the current row and have a NZ column index larger than the previous one in the row

  */
  void add_next_edge(const EdgeTy<Index> &e);

  /*!
    Should be called after all calls to add_next_edge

    Must be provided a maxNode count to pad rows out to.
    While edges were being added, the final rows may be empty, so they never get added.
  */
  void finish_edges(const Index &maxNode //!< [in] add empty rows out to maxNode
  );

  CSRCOOView<Index, Vector> view() const {
    CSRCOOView<Index, Vector> ret;
    ret.nnz_ = nnz();
    ret.num_rows_ = num_rows();
    ret.rowPtr_ = rowPtr_.data();
    ret.colInd_ = colInd_.data();
    ret.rowInd_ = rowInd_.data();
    return ret;
  }

  /*! call cudaMemAdvise(..., cudaMemAdviseSetReadMostly, 0) on all data
   */
  HOST void read_mostly() {
    rowPtr_.read_mostly();
    rowInd_.read_mostly();
    colInd_.read_mostly();
  }

  /*! call cudaMemAdvise(..., cudaMemAdviseSetAccessedBy, dev) on all data
   */
  HOST void accessed_by(const int dev) {
    rowPtr_.accessed_by(dev);
    rowInd_.accessed_by(dev);
    colInd_.accessed_by(dev);
  }

  /*! call cudaMemPrefetchAsync(..., dev) on all data
   */
  HOST void prefetch_async(const int dev, cudaStream_t stream = 0) {
    rowPtr_.prefetch_async(dev, stream);
    rowInd_.prefetch_async(dev, stream);
    colInd_.prefetch_async(dev, stream);
  }

  /*! Call shrink_to_fit on the underlying containers
   */
  HOST void shrink_to_fit() {
    rowPtr_.shrink_to_fit();
    rowInd_.shrink_to_fit();
    colInd_.shrink_to_fit();
  }

  /*! The total capacity of the underlying containers in bytes
   */
  HOST uint64_t capacity_bytes() const noexcept {
    return rowPtr_.capacity() * sizeof(typename decltype(rowPtr_)::value_type) +
           rowInd_.capacity() * sizeof(typename decltype(rowInd_)::value_type) +
           colInd_.capacity() * sizeof(typename decltype(colInd_)::value_type);
  }
  /*! The total size of the underlying containers in bytes
   */
  HOST uint64_t size_bytes() const noexcept {
    return rowPtr_.size() * sizeof(typename decltype(rowPtr_)::value_type) +
           rowInd_.size() * sizeof(typename decltype(rowInd_)::value_type) +
           colInd_.size() * sizeof(typename decltype(colInd_)::value_type);
  }

  const Index *row_ptr() const { return rowPtr_.data(); } //!< row offset array
  const Index *col_ind() const { return colInd_.data(); } //!< column index array
  const Index *row_ind() const { return rowInd_.data(); } //!< row index array

  HOST DEVICE const Index *device_row_ptr() const { return rowPtr_.data(); } //!< row offset array
  HOST DEVICE const Index *device_col_ind() const { return colInd_.data(); } //!< column index array
  HOST DEVICE const Index *device_row_ind() const { return rowInd_.data(); } //!< row index array
};

} // namespace pangolin

#undef HOST
#undef DEVICE
#undef INLINE

#include "csr_coo-impl.hpp"