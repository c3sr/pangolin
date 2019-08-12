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

template <typename Index> class CSR;

/*! \brief a read-only view of a CSR suitable for passing to a GPU kernel by
value.

Any modifications to the underlying CSR may invalidate this view.
*/
template <typename Index> class CSRView {
  friend class CSR<Index>;

private:
  uint64_t nnz_;      //!< number of non-zeros
  uint64_t num_rows_; //!< length of rowOffset - 1

public:
  typedef Index index_type;
  const Index *rowPtr_; //!< offset in col_ that each row starts at
  const Index *colInd_; //!< non-zero column indices

  PANGOLIN_HOST PANGOLIN_DEVICE __forceinline__ uint64_t nnz() const noexcept { return nnz_; }
  PANGOLIN_HOST PANGOLIN_DEVICE __forceinline__ uint64_t num_rows() const noexcept { return num_rows_; }
  PANGOLIN_HOST PANGOLIN_DEVICE __forceinline__ uint64_t num_nodes() const noexcept { return num_rows(); }

  __forceinline__ const Index *row_ptr() const noexcept { return rowPtr_; } //!< row offset array
  __forceinline__ const Index *col_ind() const noexcept { return colInd_; } //!< column index array
  PANGOLIN_HOST PANGOLIN_DEVICE __forceinline__ const Index *device_row_ptr() const {
    return rowPtr_;
  } //!< row offset array
  PANGOLIN_HOST PANGOLIN_DEVICE __forceinline__ const Index *device_col_ind() const {
    return colInd_;
  } //!< column index array
};

/*! \brief A CSR matrix backed by CUDA Unified Memory

Copying to a GPU kernel by value will cause the underling memory to be copied as
well. For read-only GPU access, use the view() method to get a lightweight
reference to the data.
*/
template <typename Index> class CSR {
private:
  Index maxCol_;

public:
  typedef Index index_type;
  CSR();                 //!< empty CSR
  Vector<Index> rowPtr_; //!< offset in col_ that each row starts at
  Vector<Index> colInd_; //!< non-zero column indices
  PANGOLIN_HOST PANGOLIN_DEVICE uint64_t nnz() const { return colInd_.size(); } //!< number of non-zeros
  uint64_t num_nodes() const;                                                   //!< number of unique row/col indices
  PANGOLIN_HOST PANGOLIN_DEVICE uint64_t num_rows() const;                      //!< number of matrix rows

  /*! Build a CSR from a range of edges [first, last)

    Only include edges where f is true (default = all edges)
  */
  template <typename EdgeIter>
  static CSR<Index> from_edges(EdgeIter begin, EdgeIter end,
                               std::function<bool(EdgeTy<Index>)> f = [](EdgeTy<Index> e) {
                                 (void)e;
                                 return true;
                               });

  CSRView<Index> view() const; //!< create a CSRView for this CSR

  /*! call cudaMemAdvise(..., cudaMemAdviseSetReadMostly, 0) on all data
   */
  PANGOLIN_HOST void read_mostly();
  /*! call cudaMemAdvise(..., cudaMemAdviseSetAccessedBy, dev) on all data
   */
  PANGOLIN_HOST void accessed_by(const int dev);
  /*! call cudaMemPrefetchAsync(..., dev) on all data
   */
  PANGOLIN_HOST void prefetch_async(const int dev, cudaStream_t stream = 0);

  /*! Call shrink_to_fit on the underlying containers
   */
  PANGOLIN_HOST void shrink_to_fit() {
    rowPtr_.shrink_to_fit();
    colInd_.shrink_to_fit();
  }

  /*! pre-allocate space for numRows rows and nnz non-zeros
   */
  PANGOLIN_HOST void reserve(size_t numRows, size_t numNonZeros) {
    rowPtr_.reserve(numRows + 1);
    colInd_.reserve(numNonZeros);
  }

  /*! The total capacity of the underlying containers in bytes
   */
  PANGOLIN_HOST PANGOLIN_DEVICE uint64_t capacity_bytes() const noexcept {
    return rowPtr_.capacity() * sizeof(typename decltype(rowPtr_)::value_type) +
           colInd_.capacity() * sizeof(typename decltype(colInd_)::value_type);
  }
  /*! The total size of the underlying containers in bytes
   */
  PANGOLIN_HOST PANGOLIN_DEVICE uint64_t size_bytes() const noexcept {
    return rowPtr_.size() * sizeof(typename decltype(rowPtr_)::value_type) +
           colInd_.size() * sizeof(typename decltype(colInd_)::value_type);
  }

  const Index *row_ptr() const { return rowPtr_.data(); } //!< row offset array
  const Index *col_ind() const { return colInd_.data(); } //!< column index array

  PANGOLIN_HOST PANGOLIN_DEVICE const Index *device_row_ptr() const { return rowPtr_.data(); } //!< row offset array
  PANGOLIN_HOST PANGOLIN_DEVICE const Index *device_col_ind() const { return colInd_.data(); } //!< column index array
};

} // namespace pangolin

#include "csr-impl.hpp"

#undef PANGOLIN_HOST
#undef PANGOLIN_DEVICE