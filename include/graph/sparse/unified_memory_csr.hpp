#pragma once

#include <cuda_runtime.h>

#include "graph/sparse/csr.hpp"
#include "graph/dense/cuda_managed_vector.hpp"
#include "graph/edge_list.hpp"

class UnifiedMemoryCSR : public CSR<Uint>
{
  private:
    CUDAManagedVector<index_type> rowOffsets_; // offset that each row starts in data_ (num rows)
    CUDAManagedVector<index_type> data_;       // the non-zero columns (nnz)
    CUDAManagedVector<char> dataIsLocal_;      // whether we own this column

  public:
    virtual const index_type *
    row_offsets() const override
    {
        return rowOffsets_.data();
    }

    virtual const index_type *cols() const override
    {
        return data_.data();
    }

    virtual const char *is_local_cols() const
    {
        return dataIsLocal_.data();
    }

    virtual size_t num_rows() const
    {
        return rowOffsets_.empty() ? 0 : rowOffsets_.size() - 1;
    }

    virtual size_t num_nonzero_rows() const
    {
        size_t count;
        for (size_t i = 0; i < rowOffsets_.size() - 1; ++i)
        {
            count += rowOffsets_[i + 1] - rowOffsets_[i];
        }
        return count;
    }

    virtual size_t nnz() const
    {
        return data_.size();
    }

    inline std::pair<index_type, index_type> row(const size_t i) const
    {
        return std::make_pair(rowOffsets_[i], rowOffsets_[i + 1]);
    }

    static UnifiedMemoryCSR from_sorted_edgelist(const EdgeList &local);
    std::vector<UnifiedMemoryCSR> partition_nonzeros(const size_t numPartitions) const;
};