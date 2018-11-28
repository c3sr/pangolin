#pragma once

#include <nvgraph.h>

#include "graph/triangle_counter.hpp"
#include "graph/dag_lowertriangular_csr.hpp"

class NvGraphTriangleCounter : public TriangleCounter
{
private:
  int numGPUs_;
  std::vector<DAGLowerTriangularCSR> dags_;
  std::vector<nvgraphCSRTopology32I_t> csrs_;

public:
  NvGraphTriangleCounter(Config &c);
  virtual void read_data(const std::string &path) override;
  virtual void setup_data() override;
  virtual size_t count() override;
};