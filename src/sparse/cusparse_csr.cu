#include "pangolin/sparse/cusparse_csr.hu"
#include "pangolin/cusparse.hpp"

PANGOLIN_BEGIN_NAMESPACE()

CusparseCSR::CusparseCSR() : descr_(nullptr) {}

CusparseCSR CusparseCSR::from_edgelist(const EdgeList &es, bool (*edgeFilter)(const Edge &)) {

    CusparseCSR csr;

CUSPARSE(cusparseCreateMatDescr(&csr.descr_));
CUSPARSE(cusparseSetMatIndexBase(csr.descr_,CUSPARSE_INDEX_BASE_ZERO));
CUSPARSE(cusparseSetMatType(csr.descr_, CUSPARSE_MATRIX_TYPE_GENERAL ));

if (es.size() == 0) {
    LOG(warn, "constructing from empty edge list");
    return csr;
}


for (const auto &edge : es) {

    // edge has a new src and should be in a new row
    // even if the edge is filtered out, we need to add empty rows
    while (csr.csrRowPtr_.size() != size_t(edge.first + 1))
    {
        // expecting inputs to be sorted by src, so it should be at least
        // as big as the current largest row we have recored
        assert(edge.first >= csr.csrRowPtr_.size());
        // SPDLOG_TRACE(logger::console, "node {} edges start at {}", edge.src_, csr.edgeSrc_.size());
        csr.csrRowPtr_.push_back(csr.csrColInd_.size());
    }

    // filter or add the edge
    if (nullptr != edgeFilter && edgeFilter(edge)) {
        continue;
    } else {
        csr.csrColInd_.push_back(edge.second);
    }
}

// add the final length of the non-zeros to the offset array
csr.csrRowPtr_.push_back(csr.csrColInd_.size());

return csr;
}


PANGOLIN_END_NAMESPACE()