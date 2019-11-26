#pragma once

#include <nvToolsExt.h>

#include "pangolin/logger.hpp"
#include "pangolin/utilities.hpp"

namespace pangolin {

/*! Count triangles with CUSparse

Count triangle for a lower-triangular matrix A with (A x A .* A).

A x A = C
C .*= A

*/
class NVGraphTC {
private:
  int dev_;

  nvgraphHandle_t handle_;
  nvgraphGraphDescr_t graph_;
  nvgraphCSRTopology32I_t csr_;

public:
  NVGraphTC(int dev) : dev_(dev) {
    LOG(debug, "create CUSparse handle");
    csr_ = (nvgraphCSRTopology32I_t)malloc(sizeof(struct nvgraphCSRTopology32I_st));
    NVGRAPH(nvgraphCreate(&handle_));
    NVGRAPH(nvgraphCreateGraphDescr(handle_, &graph_));
  }

  NVGraphTC() : NVGraphTC(-1) {}

  /*!
  Use NVGraph to count triangles in `csr`
  */
  template <typename CSR> uint64_t count_sync(CSR &csr) {

    csr_->nvertices = csr.num_rows();
    csr_->nedges = csr.nnz();
    csr_->source_offsets = csr.row_ptr();
    csr_->destination_indices = csr.col_ind();

    NVGRAPH(nvgraphSetGraphStructure(handle_, graph_, (void *)csr_, NVGRAPH_CSR_32));
    uint64_t trcount;
    NVGRAPH(nvgraphTriangleCount(handle_, graph_, &trcount));
    return trcount;
  }

  int device() const { return dev_; }

  ~NVGraphTC() {
    free(csr_);
    NVGRAPH(nvgraphDestroyGraphDescr(handle_, graph_));
    NVGRAPH(nvgraphDestroy(handle_));
  }
};

} // namespace pangolin