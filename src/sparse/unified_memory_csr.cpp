#include "graph/sparse/unified_memory_csr.hpp"

#include <cassert>

/*
Expect the incoming edge list to be sorted in increasing order of src.
Within each src, dst should be in increasing order
src should also be < dst
*/
UnifiedMemoryCSR UnifiedMemoryCSR::from_sorted_edgelist(const EdgeList &edgeList, const size_t startingRow)
{
    UnifiedMemoryCSR csr;

    // if the starting row is not the 0th row
    csr.startingRow_ = startingRow;

    // smallest src edge
    const Int firstRow = edgeList.begin()->first;
    assert(firstRow >= startingRow);

    // add empty rows from startingRow until firstRow
    for (size_t i = 0; i < firstRow - startingRow; ++i)
    {
        csr.rowOffsets_.push_back(0);
    }

    Int maxNode = 0;

    for (const auto &e : edgeList)
    {
        const Int rowIdx = e.first - startingRow;

        // add the starting offset of the new row
        while (csr.rowOffsets_.size() < rowIdx + 1)
        {
            csr.rowOffsets_.push_back(csr.data_.size());
        }

        // add the row destination
        csr.data_.push_back(e.second);
    }

    // add final nodes with 0 out-degree
    while (csr.rowOffsets_.size() < maxNode)
    {
        csr.rowOffsets_.push_back(csr.data_.size());
    }

    return csr;
}