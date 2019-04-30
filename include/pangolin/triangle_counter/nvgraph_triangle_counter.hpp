#pragma once

#include <nvgraph.h>

#include "pangolin/dag_lowertriangular_csr.hpp"
#include "pangolin/triangle_counter/triangle_counter.hpp"

namespace pangolin {

/*! \brief Triangle Count using nvgraph

  Undefined behavior for CUDA 8 and below.

*/
class NvGraphTriangleCounter : public TriangleCounter {
private:
  int gpu_;
  DAGLowerTriangularCSR dag_;
  nvgraphCSRTopology32I_t csr_;

public:
  NvGraphTriangleCounter(Config &c);
  virtual void read_data(const std::string &path) override;
  virtual void setup_data() override;
  virtual size_t count() override;
  virtual uint64_t num_edges() override { return dag_.num_edges(); }
};

} // namespace pangolin