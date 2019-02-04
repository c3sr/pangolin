#pragma once

#include <cusparse.h>

#include "pangolin/triangle_counter/triangle_counter.hpp"
#include "pangolin/sparse/gpu_csr.hpp"
#include "pangolin/namespace.hpp"

PANGOLIN_NAMESPACE_BEGIN

class CusparseTC : public TriangleCounter
{
private:
  int gpu_;
  cusparseHandle_t handle_;

  GPUCSR<int> A_;
  GPUCSR<int> B_;

public:
  CusparseTC(Config &c);
  ~CusparseTC();
  virtual void read_data(const std::string &path) override;
  virtual void setup_data() override;
  virtual size_t count() override;
  virtual uint64_t num_edges() override {return A_.nnz(); }
};

PANGOLIN_NAMESPACE_END