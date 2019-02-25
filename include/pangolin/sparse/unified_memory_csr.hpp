#pragma once

#include <cuda_runtime.h>

#include "pangolin/dense/cuda_managed_vector.hpp"
#include "pangolin/edge_list.hpp"
#include "pangolin/sparse/csr.hpp"

PANGOLIN_BEGIN_NAMESPACE()

class UnifiedMemoryCSR : public CSR<Uint> {
private:
  CUDAManagedVector<index_type>
      rowOffsets_; // offset that each row starts in data_ (num rows)
  CUDAManagedVector<index_type> data_;  // the non-zero columns (nnz)
  CUDAManagedVector<char> dataIsLocal_; // whether we own this column

public:
  virtual const index_type *row_offsets() const override {
    return rowOffsets_.data();
  }

  virtual const index_type *cols() const override { return data_.data(); }

  virtual const char *is_local_cols() const { return dataIsLocal_.data(); }

  virtual size_t num_rows() const {
    return rowOffsets_.empty() ? 0 : rowOffsets_.size() - 1;
  }

  virtual size_t num_nonzero_rows() const {
    size_t count = 0;
    for (size_t i = 0; i < rowOffsets_.size() - 1; ++i) {
      count += rowOffsets_[i + 1] - rowOffsets_[i];
    }
    return count;
  }

  virtual size_t nnz() const { return data_.size(); }

  virtual size_t bytes() const {
    size_t sz = 0;
    sz += rowOffsets_.capacity() * sizeof(rowOffsets_[0]);
    sz += data_.capacity() * sizeof(data_[0]);
    sz += dataIsLocal_.capacity() * sizeof(dataIsLocal_[0]);
    return sz;
  }

  inline std::pair<index_type, index_type> row(const size_t i) const {
    return std::make_pair(rowOffsets_[i], rowOffsets_[i + 1]);
  }

  // static UnifiedMemoryCSR from_sorted_edgelist(const EdgeList &local);
  static UnifiedMemoryCSR
  from_sorted_edgelist(const EdgeList &local,
                       const EdgeList &remote = EdgeList());
  std::vector<UnifiedMemoryCSR>
  partition_nonzeros(const size_t numPartitions) const;
};

PANGOLIN_END_NAMESPACE()