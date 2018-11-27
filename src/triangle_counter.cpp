#include "graph/triangle_counter.hpp"
#include "graph/cpu_triangle_counter.hpp"
#include "graph/cudamemcpy_tc.hpp"
#include "graph/nvgraph_triangle_counter.hpp"
#include "graph/um_tc.hpp"
#include "graph/zc_tc.hpp"

TriangleCounter::~TriangleCounter() {}

void TriangleCounter::setup_data()
{
    LOG(info, "setup_data no-op");
}

TriangleCounter *TriangleCounter::CreateTriangleCounter(Config &c)
{
    if (c.type_ == "zc")
    {
        return new ZeroCopyTriangleCounter();
    }
    else if (c.type_ == "cudamemcpy")
    {
        return new CudaMemcpyTC();
    }
    else if (c.type_ == "um")
    {
        return new UMTC();
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