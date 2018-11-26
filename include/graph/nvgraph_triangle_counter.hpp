#pragma once

#include "graph/triangle_counter.hpp"

class NvGraphTriangleCounter : public TriangleCounter
{

  public:
    virtual void execute(const char *filename, int omp_numthreads);
};