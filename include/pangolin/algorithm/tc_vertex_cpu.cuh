

#pragma once


#include "pangolin/logger.hpp"
#include "count.cuh"
#include "search.cuh"



namespace pangolin {

class VertexCPUTC {
private:
  uint64_t count_;

public:
  VertexCPUTC(int dev) {}

  VertexCPUTC(int dev, cudaStream_t stream) {}

  VertexCPUTC(VertexCPUTC &&other) {}

  VertexCPUTC() : VertexCPUTC(0) {}

  ~VertexCPUTC() {}

  template <typename Csr>
  void count_async(const Csr &adj, const size_t rowOffset, const size_t numRows, const size_t dimBlock = 256) {
    LOG(debug, "count_async {} {}", rowOffset, numRows);
    typedef typename Csr::edge_index_type EdgeIndex;
    typedef typename Csr::node_index_type NodeIndex;
    count_ = 0;

    for (NodeIndex rowIdx = rowOffset; rowIdx < rowOffset + numRows; rowIdx += 1) {

      // the part of the row we are responsible for comparing for triangle counts
      const EdgeIndex srcPartStart = adj.partitionStart_[rowIdx];
      const EdgeIndex srcPartStop = adj.partitionStop_[rowIdx];
      const NodeIndex *srcPartBegin = &adj.colInd_[srcPartStart];
      const NodeIndex *srcPartEnd = &adj.colInd_[srcPartStop];

      // the whole row
      const EdgeIndex srcStart = adj.rowStart_[rowIdx];
      const EdgeIndex srcStop = adj.rowStop_[rowIdx];
      const NodeIndex *srcBegin = &adj.colInd_[srcStart];
      const NodeIndex *srcEnd = &adj.colInd_[srcStop];

      for (const NodeIndex *srcPtr = srcBegin; srcPtr < srcEnd; ++srcPtr) {
        NodeIndex src = *srcPtr;
        const EdgeIndex dstPartStart = adj.partitionStart_[src];
        const EdgeIndex dstPartStop = adj.partitionStop_[src];
        const NodeIndex *dstPartBegin = &adj.colInd_[dstPartStart];
        const NodeIndex *dstPartEnd = &adj.colInd_[dstPartStop];
        count_ += pangolin::serial_sorted_count_linear(srcPartBegin, srcPartEnd, dstPartBegin, dstPartEnd);
      }
    }
  }

  template <typename Csr> void count_async(const Csr &adj, const size_t dimBlock = 256) {
    count_async(adj, 0, adj.num_rows(), dimBlock);
  }

  template <typename Csr>
  uint64_t count_sync(const Csr &adj, const size_t rowOffset, const size_t n, const size_t dimBlock = 256) {
    count_async(adj, rowOffset, n, dimBlock);
    sync();
    return count();
  }

  template <typename Csr> uint64_t count_sync(const Csr &adj, const size_t dimBlock = 256) {
    count_async(adj, dimBlock);
    sync();
    return count();
  }

  void sync() {}

  uint64_t count() const { return count_; }
  int device() const { return 0; }

  double kernel_time() const { return 0; }
};

} // namespace pangolin