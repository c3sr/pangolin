#pragma once

#include <cuda_runtime.h>

#include "graph/sparse/csr.hpp"
#include "graph/dense/cuda_managed_vector.hpp"
#include "graph/edge_list.hpp"

class UnifiedMemoryCSR : public CSR<Int, Int>
{
  private:
    size_t startingRow_; // rowOffsets_[0] is actually this row
    CUDAManagedVector<index_type> rowOffsets_;
    CUDAManagedVector<scalar_type> data_;

  public:
    virtual index_type *
    row_offsets()
    {
        return rowOffsets_.data();
    }

    virtual scalar_type *data()
    {
        return data_.data();
    }

    virtual size_t num_nodes() const
    {
        return rowOffsets_.empty() ? 0 : rowOffsets_.size() - 1;
    }

    virtual size_t num_edges() const
    {
        return data_.size();
    }

    inline std::pair<index_type, index_type> row(const size_t i) const
    {
        return std::make_pair(rowOffsets_[i - startingRow_], rowOffsets_[i - startingRow_ + 1]);
    }

    static UnifiedMemoryCSR from_sorted_edgelist(const EdgeList &e, const size_t startingRow_ = 0);
};