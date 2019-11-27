#pragma once

#include "pangolin/algorithm/count.cuh"
#include "pangolin/logger.hpp"
#include "pangolin/utilities.hpp"

namespace pangolin {

template <typename NodeIndex, typename EdgeIndex>
static uint64_t count_vertex(const NodeIndex vertex, const EdgeIndex *__restrict__ rowPtr,
                             const NodeIndex *__restrict__ colInd) {
  uint64_t count = 0;

  EdgeIndex start = rowPtr[vertex];
  EdgeIndex stop = rowPtr[vertex + 1];

  SPDLOG_TRACE(logger::console(), "vertex {} ({} {})", vertex, start, stop);

  for (EdgeIndex nbrIdx = start; nbrIdx < stop; ++nbrIdx) {
    NodeIndex nbr = colInd[nbrIdx];
    EdgeIndex nbrStart = rowPtr[nbr];
    EdgeIndex nbrStop = rowPtr[nbr + 1];
    SPDLOG_TRACE(logger::console(), "compare {} {} and {} {}", start, stop, nbrStart, nbrStop);
    count += serial_sorted_count_binary(&colInd[start], stop - start, &colInd[nbrStart], nbrStop - nbrStart);
  }
  return count;
}

/*!

*/
class CPUVertexBinaryTC {
private:
  int num_threads_;

public:
  CPUVertexBinaryTC(int num_threads = 1) : num_threads_(num_threads) {}

  template <typename CSR> uint64_t count_sync(CSR &csr) {

    typedef typename CSR::index_type NodeIndex;

    uint64_t trcount = 0;
    for (NodeIndex i = 0; i < csr.num_rows(); ++i) {
      SPDLOG_TRACE(logger::console(), "row {}", i);
      trcount += count_vertex(i, csr.row_ptr(), csr.col_ind());
    }

    return trcount;
  }
};

} // namespace pangolin