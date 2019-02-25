#pragma once

#include <cusparse.h>

#include "pangolin/namespace.hpp"
#include "pangolin/sparse/gpu_csr.hpp"
#include "pangolin/triangle_counter/triangle_counter.hpp"

PANGOLIN_BEGIN_NAMESPACE()

/*! Count triangles with CUSparse cusparseScsrgemm

Create a lower-triangular matrix A, and count triangles with (A x A .* A).

A x A = C
C .*= A


*/
class CusparseTC : public TriangleCounter {
private:
  int gpu_;

  cusparseHandle_t handle_;
  cusparseMatDescr_t descrA_;
  cusparseMatDescr_t descrC_;

  GPUCSR<int> A_;

public:
  CusparseTC(Config &c);
  ~CusparseTC();
  virtual void read_data(const std::string &path) override;
  virtual void setup_data() override;
  virtual size_t count() override;
  virtual uint64_t num_edges() override { return A_.nnz(); }
};

PANGOLIN_END_NAMESPACE()