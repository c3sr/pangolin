#pragma once

#include <string>

#include "graph/config.hpp"

class TriangleCounter
{

public:
  virtual ~TriangleCounter();
  virtual void execute(const char *filename, const int omp_numthreads) = 0;

  virtual void read_data(const std::string &path) = 0;
  virtual size_t count() = 0;

  static TriangleCounter *CreateTriangleCounter(Config &config);
};