#include "graph/triangle_counter.hpp"
#include "graph/cpu_triangle_counter.hpp"
#include "graph/gpu_triangle_counter.hpp"
#include "graph/nvgraph_triangle_counter.hpp"

TriangleCounter::~TriangleCounter() {}

TriangleCounter *TriangleCounter::CreateTriangleCounter(Config &c)
{
    if (c.type_ == "gpu")
    {
        return new GPUTriangleCounter();
    }
    else if (c.type_ == "cpu")
    {
        return new CPUTriangleCounter(c);
    }
    else
    {
        LOG(critical, "unhandled triangle counter type: {}", c.type_);
        exit(-1);
    }
}