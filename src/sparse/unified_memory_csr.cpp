#include "graph/sparse/unified_memory_csr.hpp"
#include "graph/logger.hpp"
#include <cassert>

/*
Expect the incoming edge list to be sorted in increasing order of src.
Within each src, dst should be in increasing order
src should also be < dst
*/
UnifiedMemoryCSR UnifiedMemoryCSR::from_sorted_edgelist(const EdgeList &local)
{
    UnifiedMemoryCSR csr;

    // smallest src edge
    const Uint firstRow = local.begin()->first;

    // add empty rows until firstRow
    LOG(debug, "smallest row was {}", firstRow);
    for (Uint i = 0; i < firstRow; ++i)
    {
        LOG(trace, "added empty row {} before smallest row id", i);
        csr.rowOffsets_.push_back(0);
    }

    Uint maxNode = 0;

    for (const auto &e : local)
    {
        const Uint rowIdx = e.first;
        maxNode = std::max(e.first, maxNode);
        maxNode = std::max(e.second, maxNode);

        // add the starting offset of the new row
        while (csr.rowOffsets_.size() < rowIdx + 1)
        {
            csr.rowOffsets_.push_back(csr.data_.size());
        }

        // add the row destination
        csr.data_.push_back(e.second);
        csr.dataIsLocal_.push_back(1);
    }

    // add final nodes with 0 out-degree
    LOG(debug, "max node id was {}", maxNode);
    while (csr.rowOffsets_.size() < maxNode + 1)
    {
        csr.rowOffsets_.push_back(csr.data_.size());
    }

    return csr;
}

std::vector<UnifiedMemoryCSR> UnifiedMemoryCSR::partition_nonzeros(const size_t numPartitions) const
{
    assert(0 && "unimplemented.");
}