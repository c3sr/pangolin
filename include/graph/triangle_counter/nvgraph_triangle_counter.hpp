#pragma once

#include <nvgraph.h>

#include "graph/triangle_counter/triangle_counter.hpp"
#include "graph/dag_lowertriangular_csr.hpp"

class NvGraphTriangleCounter : public TriangleCounter
{
private:
  int gpu_;
  DAGLowerTriangularCSR dag_;
  nvgraphCSRTopology32I_t csr_;

public:
  NvGraphTriangleCounter(Config &c);
  virtual void read_data(const std::string &path) override;
  virtual void setup_data() override;
  virtual size_t count() override;
  virtual size_t num_edges() override { return dag_.num_edges(); }
};