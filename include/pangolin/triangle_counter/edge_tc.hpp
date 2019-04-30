#pragma once

#include "pangolin/dense/cuda_managed_vector.hpp"
#include "pangolin/sparse/unified_memory_csr.hpp"
#include "pangolin/triangle_counter/cuda_triangle_counter.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

namespace pangolin {

class EdgeTC : public CUDATriangleCounter {
private:
  enum class Kernel { LINEAR, BINARY };

private:
  Kernel kernel_;

  // partitioned data structures
  std::vector<UnifiedMemoryCSR> graphs_;
  std::vector<CUDAManagedVector<Uint>> edgeSrc_;

  // per-block triangle counts for each partition
  std::vector<CUDAManagedVector<uint64_t>> triangleCounts_;

  // per-partition device pointers (csr structure)
  std::vector<const Uint *> rowOffsets_d_;
  std::vector<const Uint *> cols_d_;
  std::vector<const char *> isLocalCol_d_;

  // additional nnz-sized data for edge sources
  std::vector<const Uint *> rows_d_;

  std::vector<uint64_t *> triangleCounts_d_;

  size_t numEdges_; // edges in input graph
  size_t numNodes_; // nodes in input graph

public:
  EdgeTC(Config &c);
  virtual ~EdgeTC();
  virtual void read_data(const std::string &path) override;
  virtual void setup_data() override;
  virtual size_t count() override;
  virtual uint64_t num_edges() override { return numEdges_; }
  virtual size_t num_nodes() { return numNodes_; }
};

} // namespace pangolin