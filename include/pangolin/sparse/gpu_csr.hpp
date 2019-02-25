#pragma once

#include "pangolin/dense/vector.hu"
#include "pangolin/edge_list.hpp"
#include "pangolin/types.hpp"

#ifdef __CUDACC__
#define PANGOLIN_CUDA_MEMBER __host__ __device__
#else
#define PANGOLIN_CUDA_MEMBER
#endif

namespace pangolin {

template <typename Index> class GPUCSR;

/*! \brief a read-only view of a GPUCSRm suitable for passing to a GPU kernel by
value.

Any modifications to the underlying GPUCSR may invalidate this view.
*/
template <typename Index> class GPUCSRView {
  friend class GPUCSR<Index>;

private:
  uint64_t nnz_;      //!< number of non-zeros
  uint64_t num_rows_; //!< length of rowOffset - 1
  Index maxCol_;      //!< largest observed column during constructiono of CSR

public:
  const Index *rowOffset_; //!< offset in col_ that each row starts at
  const Index *col_;       //!< non-zero column indices

  PANGOLIN_CUDA_MEMBER uint64_t nnz() const { return nnz_; }
  PANGOLIN_CUDA_MEMBER uint64_t num_rows() const { return num_rows_; }
  PANGOLIN_CUDA_MEMBER uint64_t max_col() const {
    return maxCol_;
  } //!< largest observed column during construction of CSR

  const Index *deviceRowPtr() {
    return rowOffset_;
  }                                            //!< rowPtr index valid on device
  const Index *deviceColInd() { return col_; } //!< colInd array valid on device
};

/*! \brief A CSR matrix backed by CUDA Unified Memory

Copying to a GPU kernel by value will cause the underling memory to be copied as
well. For read-only GPU access, use the view() method to get a lightweight
reference to the CSR data.
*/
template <typename Index> class GPUCSR {
private:
  Index maxCol_;

public:
  GPUCSR();                 //!< empty CSR
  Vector<Index> rowOffset_; //!< offset in col_ that each row starts at
  Vector<Index> col_;       //!< non-zero column indices
  PANGOLIN_CUDA_MEMBER uint64_t nnz() const {
    return col_.size();
  }                           //!< number of non-zeros
  uint64_t num_nodes() const; //!< number of unique row/col indices
  PANGOLIN_CUDA_MEMBER uint64_t num_rows() const; //!< number of matrix rows
  PANGOLIN_CUDA_MEMBER uint64_t max_col() const {
    return maxCol_;
  } //!< number of matrix columns

  /*! Build a GPUCSR from an EdgeList

  Do not include edges where edgeFilter(edge) returns true
  */
  static GPUCSR<Index>
  from_edgelist(const EdgeList &es, bool (*edgeFilter)(const Edge &) = nullptr);
  GPUCSRView<Index> view() const; //!< create a GPUCSRView for this GPUCSR

  const Index *deviceRowPtr() {
    return rowOffset_.data();
  } //!< row offset array, valid on device
  const Index *deviceColInd() {
    return col_.data();
  } //!< column index array, valid on device
};

} // namespace pangolin

#undef PANGOLIN_CUDA_MEMBER

#include "gpu_csr-impl.hpp"