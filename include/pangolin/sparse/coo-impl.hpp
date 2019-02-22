#pragma once

#include <set>

#include "gpu_csr.hpp"
#include "pangolin/logger.hpp"

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif 

PANGOLIN_BEGIN_NAMESPACE()

template<typename Index>
COO<Index>::COO() { 
}

template<typename Index>
HOST DEVICE uint64_t COO<Index>::num_rows() const { 
    if (rowPtr_.size() == 0) {
        return 0;
    } else {
        return rowPtr_.size() - 1;
    }
}

template<typename Index>
uint64_t COO<Index>::num_nodes() const { 
    std::set<Index> nodes;
    // add all dsts
    for (Index ci = 0; ci < colInd_.size(); ++ci) {
        nodes.insert(colInd_[ci]);
    }
    // add non-zero sources
    for (Index i = 0; i < rowPtr_.size() - 1; ++i) {
        Index row_start = rowPtr_[i];
        Index row_end = rowPtr_[i+1];
        if (row_start != row_end) {
            nodes.insert(i);
        }
    }
    return nodes.size();
}

template<typename Index>
COO<Index> COO<Index>::from_edgelist(const EdgeList &es, bool (*edgeFilter)(const Edge &)) {
    COO<Index> csr;

    
    if (es.size() == 0) {
        LOG(warn, "constructing from empty edge list");
        return csr;
    }


    for (const auto &edge : es) {

        const Index src = static_cast<Index>(edge.first);
        const Index dst = static_cast<Index>(edge.second);

        // edge has a new src and should be in a new row
        // even if the edge is filtered out, we need to add empty rows
        while (csr.rowPtr_.size() != size_t(src + 1))
        {
            // expecting inputs to be sorted by src, so it should be at least
            // as big as the current largest row we have recored
            assert(src >= csr.rowPtr_.size());
            // SPDLOG_TRACE(logger::console, "node {} edges start at {}", edge.src_, csr.edgeSrc_.size());
            csr.rowPtr_.push_back(csr.colInd_.size());
        }

        // filter or add the edge
        if (nullptr != edgeFilter && edgeFilter(edge)) {
            continue;
        } else {
            csr.rowInd_.push_back(src);
            csr.colInd_.push_back(dst);
        }
    }

    // add the final length of the non-zeros to the offset array
    csr.rowPtr_.push_back(csr.colInd_.size());

    assert(csr.rowInd_.size() == csr.colInd_.size());
    return csr;
}

template<typename Index>
COOView<Index> COO<Index>::view() const {
    COOView<Index> view;
    view.nnz_ = nnz();
    view.num_rows_ = num_rows();
    view.rowPtr_ = rowPtr_.data();
    view.colInd_ = colInd_.data();
    view.rowInd_ = rowInd_.data();
    return view;
}


PANGOLIN_END_NAMESPACE()

#undef HOST
#undef DEVICE
