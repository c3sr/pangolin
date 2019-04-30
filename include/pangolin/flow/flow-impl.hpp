#pragma once

#include "pangolin/logger.hpp"

namespace pangolin {

inline void numa_bind_node(const int node) {

  if (-1 == node) {
    numa_bind(numa_all_nodes_ptr);
  } else if (node >= 0) {
    struct bitmask *nodemask = numa_allocate_nodemask();
    nodemask = numa_bitmask_setbit(nodemask, node);
    numa_bind(nodemask);
    numa_free_nodemask(nodemask);
  } else {
    LOG(critical, "expected node >= -1");
    exit(1);
  }
}

template <typename T, size_t MC> void FlowVector<T, MC>::reserve(size_t n) {

  if (true) { // strategy == numa/cuda
    if (n > capacity_) {

      // Look for a CPU component in producers
      for (const auto &p : producers_) {
        if (p.is_cpu()) {
          LOG(debug, "bind to node {}", p.id());
          numa_bind_node(p.id());
        }
      }

      value_type *newData = nullptr;
      CUDA_RUNTIME(cudaMallocManaged(&newData, sizeof(value_type) * n));
      std::memmove(newData, data_, sizeof(value_type) * size_);
      CUDA_RUNTIME(cudaFree(data_));
      data_ = newData;
      capacity_ = n;
    }
  }
}

} // namespace pangolin