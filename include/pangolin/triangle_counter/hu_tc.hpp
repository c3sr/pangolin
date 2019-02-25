#pragma once

#include "pangolin/config.hpp"
#include "pangolin/sparse/dag2019.hpp"
#include "pangolin/triangle_counter/cuda_triangle_counter.hpp"

PANGOLIN_BEGIN_NAMESPACE()

class Hu2018TC : public CUDATriangleCounter {
private:
  DAG2019 hostDAG_;
  size_t *triangleCounts_; // per-edge triangle counts
  Int *edgeSrc_d_;
  Int *edgeDst_d_;
  Int *nodes_d_;

public:
  Hu2018TC(Config &c);
  virtual ~Hu2018TC();
  virtual void read_data(const std::string &path) override;
  virtual void setup_data() override;
  virtual size_t count() override;
  virtual uint64_t num_edges() override { return hostDAG_.num_edges(); }
  virtual size_t num_nodes() override { return hostDAG_.num_nodes(); }
};

PANGOLIN_END_NAMESPACE()