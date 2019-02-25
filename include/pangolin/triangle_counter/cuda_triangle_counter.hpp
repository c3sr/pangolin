#pragma once

#include <map>
#include <set>
#include <vector>

#include <cuda_runtime.h>

#include "pangolin/config.hpp"
#include "pangolin/namespace.hpp"
#include "pangolin/triangle_counter/triangle_counter.hpp"

PANGOLIN_BEGIN_NAMESPACE()

class CUDATriangleCounter : public TriangleCounter {
protected:
  // treat each of these as a separate GPU (even if there are duplicates)
  std::vector<int> gpus_;

  // the unique gpus in gpus_
  std::set<int> unique_gpus_;

  // properties for each gpu
  std::map<int, cudaDeviceProp> cudaDeviceProps_;

public:
  CUDATriangleCounter(Config &c);

  // always available
  std::vector<int> &gpus() const;
  std::set<int> &unique_gpus() const;

  // available after read_data()
  virtual size_t num_nodes() = 0;
};

PANGOLIN_END_NAMESPACE()