#pragma once

#include "graph/triangle_counter.hpp"
#include "graph/dag_lowertriangular_csr.hpp"

class VertexTC : public TriangleCounter
{
private:
  DAGLowerTriangularCSR dag_;
  Int *sourceOffsets_;
  Int *destinationIndices_;
  int *blockTriangleCounts_;

public:
  VertexTC();
  ~VertexTC();
  virtual void read_data(const std::string &path) override;
  virtual void setup_data() override;
  virtual size_t count() override;
  virtual size_t num_edges() override { return dag_.num_edges(); }
};